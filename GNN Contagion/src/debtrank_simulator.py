"""
DebtRank Contagion Simulator - Multi-Layer Financial Network Analysis

This module implements the DebtRank algorithm to simulate financial contagion
across multi-layer networks, generating ground truth labels for GNN training.

Architecture optimized for distributed processing with multiprocessing support.
Based on Battiston et al. (2012) methodology with multi-layer network extensions.

Project: Multi-Layer Financial Contagion Modeling with Graph Neural Networks
"""

import numpy as np
import pandas as pd
import logging
import time
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import yaml
from io import StringIO

from google.cloud import storage

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ShockScenario:
    """Configuration for a financial shock scenario."""
    scenario_type: str
    initial_bank_id: int
    shock_magnitude: float = 1.0
    scenario_id: str = ""


@dataclass
class ContagionResult:
    """Results from a single contagion simulation."""
    network_id: int
    shock_scenario: ShockScenario
    initial_shock_bank: int
    final_bank_states: np.ndarray
    binary_defaults: np.ndarray
    systemic_risk: float
    cascade_length: int
    total_defaults: int
    convergence_achieved: bool
    computation_time: float


# =============================================================================
# WORKER FUNCTIONS FOR DISTRIBUTED PROCESSING
# =============================================================================

def simulate_network_batch_worker(network_ids: List[int], 
                                 project_id: str, 
                                 bucket_name: str,
                                 config_dict: Dict[str, Any],
                                 nodes_data: Dict[str, Any]) -> List[Dict]:
    """
    Process a batch of networks in a worker process.
    
    This function operates independently of class state to ensure compatibility
    with multiprocessing serialization requirements.
    
    Args:
        network_ids: List of network identifiers to process
        project_id: Google Cloud Platform project identifier
        bucket_name: Cloud storage bucket name
        config_dict: Configuration parameters
        nodes_data: Serialized node data dictionary
        
    Returns:
        List of result dictionaries containing simulation outcomes
    """
    logger.info(f"Worker processing {len(network_ids)} networks")
    
    try:
        # Initialize cloud storage clients in worker process
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        
        # Reconstruct nodes DataFrame from serializable data
        nodes_df = pd.DataFrame(nodes_data)
        logger.debug(f"Reconstructed nodes_df with {len(nodes_df)} nodes")
        
        debtrank_config = config_dict.get('debtrank', {})
        
        results = []
        successful_networks = 0
        
        for network_id in network_ids:
            try:
                # Load network from cloud storage
                networks = load_network_from_cloud_worker(network_id, bucket, nodes_df)
                if networks is None:
                    logger.warning(f"Skipping network {network_id} - could not load")
                    continue
                
                # Generate shock scenarios for this network
                shock_scenarios = create_shock_scenarios_worker(network_id, nodes_df, debtrank_config)
                
                # Execute DebtRank simulation for each scenario
                for scenario in shock_scenarios:
                    result = simulate_debtrank_contagion_worker(
                        networks, scenario, nodes_df, debtrank_config
                    )
                    
                    # Convert to serializable dictionary format
                    result_dict = {
                        'network_id': network_id,
                        'scenario_type': scenario.scenario_type,
                        'initial_shock_bank': scenario.initial_bank_id,
                        'shock_magnitude': scenario.shock_magnitude,
                        'scenario_id': scenario.scenario_id,
                        'systemic_risk': float(result.systemic_risk),
                        'cascade_length': int(result.cascade_length),
                        'total_defaults': int(result.total_defaults),
                        'convergence_achieved': bool(result.convergence_achieved),
                        'computation_time': float(result.computation_time),
                        'final_bank_states': result.final_bank_states.tolist(),
                        'binary_defaults': result.binary_defaults.tolist()
                    }
                    results.append(result_dict)
                
                successful_networks += 1
                
            except Exception as e:
                logger.error(f"Error processing network {network_id}: {e}")
                continue
        
        logger.info(f"Worker completed: {successful_networks} networks, {len(results)} total results")
        return results
        
    except Exception as e:
        logger.error(f"Fatal error in worker process: {e}", exc_info=True)
        return []


def load_network_from_cloud_worker(network_id: int, bucket, nodes_df: pd.DataFrame) -> Optional[Dict[str, np.ndarray]]:
    """
    Load network edge list from cloud storage and convert to adjacency matrices.
    
    Args:
        network_id: Network identifier
        bucket: Google Cloud Storage bucket object
        nodes_df: DataFrame containing node metadata
        
    Returns:
        Dictionary mapping layer names to adjacency matrices, or None if load fails
    """
    blob_path = f"networks/edge_lists/edges_sim_{network_id:06d}.csv"
    
    try:
        blob = bucket.blob(blob_path)
        
        if not blob.exists():
            logger.warning(f"Network {network_id} not found at {blob_path}")
            return None
        
        # Download and parse CSV edge list
        csv_content = blob.download_as_text()
        edges_df = pd.read_csv(StringIO(csv_content))
        
        logger.debug(f"Network {network_id} loaded: {len(edges_df)} edges across {edges_df['layer'].nunique()} layers")
        
        return edges_to_adjacency_matrices_worker(edges_df, nodes_df)
        
    except Exception as e:
        logger.error(f"Error loading network {network_id} from cloud: {e}")
        return None


def edges_to_adjacency_matrices_worker(edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Convert edge list DataFrame to layer-specific adjacency matrices.
    
    Args:
        edges_df: DataFrame with columns [source, target, layer, weight]
        nodes_df: DataFrame containing node identifiers
        
    Returns:
        Dictionary mapping layer names to weighted adjacency matrices
    """
    n_nodes = len(nodes_df)
    node_id_map = {int(node_id): i for i, node_id in enumerate(nodes_df['node_id'])}
    
    networks = {}
    
    for layer in edges_df['layer'].unique():
        adj_matrix = np.zeros((n_nodes, n_nodes))
        layer_edges = edges_df[edges_df['layer'] == layer]
        
        valid_edges = 0
        invalid_edges = 0
        
        for _, edge in layer_edges.iterrows():
            source_idx = node_id_map.get(int(edge['source']))
            target_idx = node_id_map.get(int(edge['target']))
            
            if source_idx is not None and target_idx is not None:
                adj_matrix[source_idx, target_idx] = float(edge['weight'])
                valid_edges += 1
            else:
                invalid_edges += 1
                if invalid_edges < 5:
                    logger.warning(f"Invalid edge: source={edge['source']}, target={edge['target']}")
        
        networks[layer] = adj_matrix
        logger.debug(f"Layer '{layer}': {valid_edges} valid edges, {invalid_edges} invalid edges")
    
    return networks


def create_shock_scenarios_worker(network_id: int, nodes_df: pd.DataFrame, 
                                 debtrank_config: Dict) -> List[ShockScenario]:
    """
    Generate shock scenarios based on configuration parameters.
    
    Supports multiple shock types:
    - largest_bank: Target institution with highest total assets
    - random_bank: Randomly selected institution
    - tier_2_bank: Randomly selected tier 2 institution
    
    Args:
        network_id: Network identifier for scenario naming
        nodes_df: DataFrame containing node attributes
        debtrank_config: Configuration dictionary with shock_types list
        
    Returns:
        List of ShockScenario objects
    """
    scenarios = []
    shock_types = debtrank_config.get('shock_types', ['largest_bank'])
    
    for shock_type in shock_types:
        initial_bank_node_id = -1
        
        if shock_type == 'largest_bank':
            # Select institution with maximum total assets
            largest_bank_row = nodes_df.loc[nodes_df['total_assets'].idxmax()]
            initial_bank_node_id = int(largest_bank_row['node_id'])
            
        elif shock_type == 'random_bank':
            # Randomly select an institution
            random_idx = np.random.randint(0, len(nodes_df))
            initial_bank_node_id = int(nodes_df['node_id'].iloc[random_idx])
            
        elif shock_type == 'tier_2_bank':
            # Select from tier 2 institutions
            tier_2_banks = nodes_df[nodes_df['bank_tier'] == 2]
            if len(tier_2_banks) > 0:
                random_tier2_row = tier_2_banks.sample(n=1)
                initial_bank_node_id = int(random_tier2_row['node_id'].iloc[0])
            else:
                logger.warning("No tier 2 banks found for shock scenario")
                continue
        
        if initial_bank_node_id != -1:
            scenario = ShockScenario(
                scenario_type=shock_type,
                initial_bank_id=initial_bank_node_id,
                scenario_id=f"net_{network_id:06d}_{shock_type}_{initial_bank_node_id}"
            )
            scenarios.append(scenario)
    
    return scenarios


def simulate_debtrank_contagion_worker(networks: Dict[str, np.ndarray],
                                      shock_scenario: ShockScenario,
                                      nodes_df: pd.DataFrame,
                                      debtrank_config: Dict) -> ContagionResult:
    """
    Execute DebtRank algorithm to simulate financial contagion.
    
    Implements the iterative DebtRank algorithm where distressed institutions
    propagate losses through the network until convergence or maximum iterations.
    
    Algorithm states:
    - U (Undistressed): Initial state, h=0
    - D (Distressed): Currently propagating distress, 0<h≤1
    - I (Inactive): Previously distressed, no longer propagating
    
    Args:
        networks: Dictionary of layer-specific adjacency matrices
        shock_scenario: Initial shock configuration
        nodes_df: DataFrame containing node attributes (capital, assets)
        debtrank_config: Algorithm parameters
        
    Returns:
        ContagionResult object with simulation outcomes
    """
    start_time = time.time()
    
    # Extract algorithm parameters
    max_iterations = debtrank_config.get('max_iterations', 1000)
    failure_threshold = debtrank_config.get('failure_threshold', 0.8)
    layer_weights = debtrank_config.get('layer_propagation_weights', {'deposits_loans': 1.0})
    
    n_nodes = len(nodes_df)
    
    # Extract node attributes
    capital = nodes_df['capital'].values
    total_assets = nodes_df['total_assets'].values
    
    # Create node ID mapping
    node_id_map = {int(node_id): i for i, node_id in enumerate(nodes_df['node_id'])}
    
    # Construct aggregated interbank asset matrix
    A = np.zeros((n_nodes, n_nodes))
    for layer_name, W_layer in networks.items():
        weight = layer_weights.get(layer_name, 0.0)
        if weight > 0:
            A += weight * W_layer
    
    # Build impact matrix W: normalize by creditor capital
    W_impact = np.divide(A.T, capital[:, np.newaxis], 
                        out=np.zeros_like(A.T), 
                        where=capital[:, np.newaxis] > 0)
    W_impact = np.minimum(1.0, W_impact)
    
    # Initialize state variables
    h = np.zeros(n_nodes)  # Continuous distress level [0,1]
    s = np.full(n_nodes, 'U')  # Discrete state {U, D, I}
    
    # Apply initial shock
    shocked_bank_idx = node_id_map.get(shock_scenario.initial_bank_id)
    if shocked_bank_idx is None:
        raise ValueError(f"Invalid shock bank ID: {shock_scenario.initial_bank_id}")
    
    h[shocked_bank_idx] = shock_scenario.shock_magnitude
    s[shocked_bank_idx] = 'D'
    
    # Execute DebtRank iterative algorithm
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        if 'D' not in s:
            break
        
        h_prev = h.copy()
        
        # Identify currently distressed institutions
        distressed_indices = np.where(s == 'D')[0]
        
        # Propagate distress through network
        additional_distress = (W_impact @ (h * (s == 'D'))).T
        h = np.minimum(1.0, h + additional_distress)
        
        # Update discrete states
        s[distressed_indices] = 'I'  # Distressed → Inactive
        newly_distressed = (h > 0) & (s == 'U')
        s[newly_distressed] = 'D'  # Undistressed → Distressed
        
        # Check for convergence
        if np.allclose(h, h_prev, atol=1e-7):
            break
    
    # Calculate final metrics
    final_distress = h
    binary_defaults = (final_distress >= failure_threshold).astype(int)
    
    # Compute asset-weighted systemic risk
    bank_weights = total_assets / np.sum(total_assets) if np.sum(total_assets) > 0 else np.zeros(n_nodes)
    systemic_risk = np.sum(final_distress * bank_weights)
    
    computation_time = time.time() - start_time
    
    return ContagionResult(
        network_id=0,
        shock_scenario=shock_scenario,
        initial_shock_bank=shock_scenario.initial_bank_id,
        final_bank_states=final_distress,
        binary_defaults=binary_defaults,
        systemic_risk=systemic_risk,
        cascade_length=iteration,
        total_defaults=np.sum(binary_defaults),
        convergence_achieved=(iteration < max_iterations),
        computation_time=computation_time
    )


# =============================================================================
# MAIN SIMULATOR CLASS
# =============================================================================

class DebtRankSimulator:
    """
    DebtRank algorithm implementation for multi-layer financial network contagion.
    
    Architecture designed for distributed processing with multiprocessing support.
    Avoids storing unpickleable objects in class state for worker compatibility.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the DebtRank simulator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        logger.info(f"Initializing DebtRank simulator with config: {config_path}")
        self.config = self._load_config(config_path)
        self.debtrank_config = self.config.get('debtrank', {})
        
        # Algorithm parameters
        self.max_iterations = self.debtrank_config.get('max_iterations', 1000)
        
        # Node data
        self.nodes_df: Optional[pd.DataFrame] = None
        self.n_nodes = 0
        self.node_id_map: Dict[int, int] = {}

        # Cloud storage configuration (connection parameters only)
        self.project_id: Optional[str] = None
        self.bucket_name: Optional[str] = None
        
        logger.info("DebtRank simulator initialized successfully")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        return config

    def load_node_data(self, nodes_path: str = "data/processed/nodes.csv") -> bool:
        """
        Load node attributes and create identifier mapping.
        
        Args:
            nodes_path: Path to CSV file containing node data
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading node data from {nodes_path}")
        try:
            nodes_file = Path(nodes_path)
            if not nodes_file.exists():
                logger.error(f"Nodes file not found: {nodes_path}")
                return False
            
            self.nodes_df = pd.read_csv(nodes_file)
            self.n_nodes = len(self.nodes_df)
            
            # Create node ID to index mapping
            self.node_id_map = {node_id: i for i, node_id in enumerate(self.nodes_df['node_id'])}
            
            logger.info(f"Successfully loaded {self.n_nodes} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Error loading node data: {e}", exc_info=True)
            return False

    def init_cloud_storage(self, project_id: str, bucket_name: str):
        """
        Configure cloud storage connection parameters.
        
        Stores connection parameters only (not client objects) for
        multiprocessing compatibility.
        
        Args:
            project_id: Google Cloud Platform project ID
            bucket_name: Cloud Storage bucket name
        """
        logger.info(f"Setting cloud storage config - Project: {project_id}, Bucket: {bucket_name}")
        
        self.project_id = project_id
        self.bucket_name = bucket_name
        
        # Verify connection without storing client
        try:
            test_client = storage.Client(project=project_id)
            test_bucket = test_client.bucket(bucket_name)
            
            if test_bucket.exists():
                logger.info(f"Successfully connected to bucket: {bucket_name}")
            else:
                logger.warning(f"Bucket {bucket_name} may not exist or be accessible")
                
        except Exception as e:
            logger.error(f"Failed to test cloud storage connection: {e}")
            raise

    def load_network_from_local(self, network_id: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load network from local file system.
        
        Args:
            network_id: Network identifier
            
        Returns:
            Dictionary of layer adjacency matrices, or None if load fails
        """
        network_path = Path(f"data/networks/edge_lists/edges_sim_{network_id:06d}.csv")
        
        try:
            if not network_path.exists():
                logger.warning(f"Network file not found: {network_path}")
                return None
            
            edges_df = pd.read_csv(network_path)
            logger.info(f"Network {network_id} loaded locally: {len(edges_df)} edges")
            return edges_to_adjacency_matrices_worker(edges_df, self.nodes_df)
            
        except Exception as e:
            logger.error(f"Error loading network from {network_path}: {e}")
            return None

    def simulate_single_network(self, network_id: int, source: str = "cloud") -> List[ContagionResult]:
        """
        Execute DebtRank simulation for a single network.
        
        Args:
            network_id: Network identifier to simulate
            source: Data source, either "cloud" or "local"
            
        Returns:
            List of ContagionResult objects for all scenarios
        """
        try:
            # Load network from specified source
            if source == "cloud":
                if not self.project_id or not self.bucket_name:
                    raise ValueError("Cloud storage not configured. Call init_cloud_storage first.")
                
                storage_client = storage.Client(project=self.project_id)
                bucket = storage_client.bucket(self.bucket_name)
                networks = load_network_from_cloud_worker(network_id, bucket, self.nodes_df)
            else:
                networks = self.load_network_from_local(network_id)
            
            if networks is None:
                logger.warning(f"Could not load network {network_id}")
                return []
            
            # Generate shock scenarios
            shock_scenarios = create_shock_scenarios_worker(network_id, self.nodes_df, self.debtrank_config)
            
            # Execute simulations
            results = []
            for scenario in shock_scenarios:
                result = simulate_debtrank_contagion_worker(
                    networks, scenario, self.nodes_df, self.debtrank_config
                )
                result.network_id = network_id
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error simulating network {network_id}: {e}")
            return []

    async def run_distributed_simulation(self, network_ids: List[int], 
                                       batch_size: int, max_workers: int) -> List[ContagionResult]:
        """
        Execute distributed DebtRank simulation using multiprocessing.
        
        Divides work into batches and processes them in parallel using
        independent worker processes with fresh cloud storage clients.
        
        Args:
            network_ids: List of network identifiers to simulate
            batch_size: Number of networks per batch
            max_workers: Maximum number of parallel worker processes
            
        Returns:
            List of all ContagionResult objects
        """
        logger.info(f"Starting distributed simulation for {len(network_ids)} networks with {max_workers} workers")
        
        if not self.project_id or not self.bucket_name:
            raise ValueError("Cloud storage not configured. Call init_cloud_storage first.")
        
        if self.nodes_df is None:
            raise ValueError("Node data not loaded. Call load_node_data first.")
        
        # Prepare serializable data for worker processes
        nodes_data = {
            'node_id': self.nodes_df['node_id'].tolist(),
            'total_assets': self.nodes_df['total_assets'].tolist(),
            'capital': self.nodes_df['capital'].tolist(),
            'bank_tier': self.nodes_df['bank_tier'].tolist()
        }
        
        # Include all additional columns
        for col in self.nodes_df.columns:
            if col not in nodes_data:
                nodes_data[col] = self.nodes_df[col].tolist()
        
        config_dict = self.config.copy()
        
        # Divide work into batches
        batches = [network_ids[i:i + batch_size] for i in range(0, len(network_ids), batch_size)]
        logger.info(f"Split into {len(batches)} batches of size {batch_size}")
        
        all_results = []
        
        # Execute batches in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(
                    simulate_network_batch_worker,
                    batch,
                    self.project_id,
                    self.bucket_name,
                    config_dict,
                    nodes_data
                ): i for i, batch in enumerate(batches, 1)
            }
            
            # Collect results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_results = future.result()
                    
                    # Convert dictionaries back to ContagionResult objects
                    for result_dict in batch_results:
                        contagion_result = self._dict_to_contagion_result(result_dict)
                        all_results.append(contagion_result)
                    
                    completed_batches += 1
                    logger.info(f"Batch {batch_num}/{len(batches)} complete. "
                              f"Total results: {len(all_results)}")
                              
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
        
        logger.info(f"Distributed simulation complete: {len(all_results)} total results")
        return all_results

    def _dict_to_contagion_result(self, result_dict: Dict) -> ContagionResult:
        """
        Convert serialized dictionary back to ContagionResult object.
        
        Args:
            result_dict: Dictionary with result data
            
        Returns:
            ContagionResult object
        """
        scenario = ShockScenario(
            scenario_type=result_dict['scenario_type'],
            initial_bank_id=result_dict['initial_shock_bank'],
            shock_magnitude=result_dict.get('shock_magnitude', 1.0),
            scenario_id=result_dict.get('scenario_id', '')
        )
        
        return ContagionResult(
            network_id=result_dict['network_id'],
            shock_scenario=scenario,
            initial_shock_bank=result_dict['initial_shock_bank'],
            final_bank_states=np.array(result_dict['final_bank_states']),
            binary_defaults=np.array(result_dict['binary_defaults']),
            systemic_risk=result_dict['systemic_risk'],
            cascade_length=result_dict['cascade_length'],
            total_defaults=result_dict['total_defaults'],
            convergence_achieved=result_dict['convergence_achieved'],
            computation_time=result_dict['computation_time']
        )

    def save_results(self, results: List[ContagionResult],
                    output_dir: str = "data/results",
                    upload_to_cloud: bool = True):
        """
        Save simulation results to local storage and optionally to cloud.
        
        Generates both detailed JSON output and summary CSV for analysis.
        
        Args:
            results: List of ContagionResult objects to save
            output_dir: Local directory for output files
            upload_to_cloud: Whether to upload results to cloud storage
        """
        if not results:
            logger.warning("No results to save.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {len(results)} results to {output_path}")
        
        # Convert results to serializable format
        serializable_results = []
        for res in results:
            res_dict = {
                'network_id': res.network_id,
                'scenario_type': res.shock_scenario.scenario_type,
                'initial_shock_bank': res.initial_shock_bank,
                'shock_magnitude': res.shock_scenario.shock_magnitude,
                'scenario_id': res.shock_scenario.scenario_id,
                'systemic_risk': float(res.systemic_risk),
                'cascade_length': int(res.cascade_length),
                'total_defaults': int(res.total_defaults),
                'convergence_achieved': bool(res.convergence_achieved),
                'computation_time': float(res.computation_time),
                'final_bank_states': res.final_bank_states.tolist(),
                'binary_defaults': res.binary_defaults.tolist()
            }
            serializable_results.append(res_dict)

        # Save detailed results as JSON
        json_path = output_path / "debtrank_results.json"
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved detailed results to {json_path}")
        
        # Create summary DataFrame
        summary_data = []
        for res_dict in serializable_results:
            summary_dict = res_dict.copy()
            del summary_dict['final_bank_states']
            del summary_dict['binary_defaults']
            summary_data.append(summary_dict)
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = output_path / "debtrank_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")
        
        # Upload to cloud storage if configured
        if upload_to_cloud and self.project_id and self.bucket_name:
            try:
                storage_client = storage.Client(project=self.project_id)
                bucket = storage_client.bucket(self.bucket_name)
                
                # Create timestamped folder
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                cloud_folder = f"data/results/debtrank_{timestamp}"
                
                # Upload JSON results
                json_blob = bucket.blob(f"{cloud_folder}/debtrank_results.json")
                json_blob.upload_from_filename(str(json_path))
                logger.info(f"Uploaded detailed results to gs://{self.bucket_name}/{cloud_folder}/debtrank_results.json")
                
                # Upload CSV summary
                csv_blob = bucket.blob(f"{cloud_folder}/debtrank_summary.csv")
                csv_blob.upload_from_filename(str(csv_path))
                logger.info(f"Uploaded summary to gs://{self.bucket_name}/{cloud_folder}/debtrank_summary.csv")
                
                # Upload run metadata
                metadata = {
                    'timestamp': timestamp,
                    'total_results': len(results),
                    'networks_processed': len(set(res.network_id for res in results)),
                    'avg_systemic_risk': float(summary_df['systemic_risk'].mean()),
                    'avg_defaults': float(summary_df['total_defaults'].mean()),
                    'avg_cascade_length': float(summary_df['cascade_length'].mean())
                }
                
                metadata_blob = bucket.blob(f"{cloud_folder}/run_metadata.json")
                metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
                logger.info(f"Uploaded metadata to gs://{self.bucket_name}/{cloud_folder}/run_metadata.json")
                
                logger.info(f"All results uploaded to cloud folder: gs://{self.bucket_name}/{cloud_folder}/")
                
            except Exception as e:
                logger.error(f"Failed to upload results to cloud: {e}")
                logger.info("Results are still available locally")
        
        # Log summary statistics
        avg_defaults = summary_df['total_defaults'].mean()
        avg_systemic_risk = summary_df['systemic_risk'].mean()
        avg_cascade_length = summary_df['cascade_length'].mean()
        logger.info(f"Results summary: Avg defaults: {avg_defaults:.1f}, "
                    f"Avg systemic risk: {avg_systemic_risk:.4f}, "
                    f"Avg cascade length: {avg_cascade_length:.1f}")


async def main():
    """Main execution function for DebtRank simulation."""
    parser = argparse.ArgumentParser(
        description="Execute DebtRank Contagion Simulations on Multi-Layer Financial Networks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--start', type=int, default=0, 
                       help="Starting network ID")
    parser.add_argument('--end', type=int, default=9, 
                       help="Ending network ID (inclusive)")
    parser.add_argument('--workers', type=int, default=4, 
                       help="Number of parallel worker processes")
    parser.add_argument('--batch-size', type=int, default=25, 
                       help="Networks per batch for parallel processing")
    parser.add_argument('--config', type=str, default="config/model_config.yaml", 
                       help="Path to model configuration file")
    parser.add_argument('--cloud-config', type=str, default="config/cloud_config.yaml", 
                       help="Path to cloud configuration file")
    parser.add_argument('--source', type=str, default="cloud", choices=["cloud", "local"], 
                       help="Source of network data files")
    parser.add_argument('--log-level', type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging verbosity level")
    
    args = parser.parse_args()

    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Add file handler for persistent logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / f"debtrank_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("DEBTRANK CONTAGION SIMULATION STARTING")
    logger.info(f"Command line arguments: {vars(args)}")
    logger.info("=" * 70)
    
    try:
        # Initialize simulator
        simulator = DebtRankSimulator(config_path=args.config)
        
        # Load node data
        if not simulator.load_node_data():
            logger.error("Failed to load node data. Exiting.")
            return
        
        # Configure cloud storage if using cloud source
        if args.source == "cloud":
            with open(args.cloud_config, 'r') as f:
                cloud_conf = yaml.safe_load(f)['gcp']
            simulator.init_cloud_storage(
                project_id=cloud_conf['project_id'],
                bucket_name=cloud_conf['storage_bucket']
            )
        
        # Display execution parameters
        print(f"Network range: {args.start} to {args.end}")
        print(f"Workers: {args.workers}")
        print(f"Batch size: {args.batch_size}")
        print(f"Data source: {args.source}")
        
        # Generate network ID list
        network_ids = list(range(args.start, args.end + 1))
        
        # Execute distributed simulation
        results = await simulator.run_distributed_simulation(
            network_ids=network_ids,
            batch_size=args.batch_size,
            max_workers=args.workers
        )
        
        # Save results
        simulator.save_results(results)
    
        logger.info(f"Total results generated: {len(results)}")
        
        print("\nSimulation completed successfully!")
        print(f"Generated {len(results)} simulation results")
        print(f"Results saved to: data/results/")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    asyncio.run(main())