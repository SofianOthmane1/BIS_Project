"""
Masking Experiment for GNN Robustness

This script evaluates Graph Neural Network performance under partial information conditions
by systematically masking edges at various levels. The experiment uses a holdout set of
networks (IDs 17000-17999) to assess model robustness and prediction stability.

Key Features:
- Statistical rigor through increased sample sizes and multiple random seeds
- Network diversity sampling to ensure representative results
- Comprehensive variance measurement and uncertainty quantification
- Performance degradation tracking across masking levels
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import logging
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
from io import StringIO, BytesIO
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr
import gc
import re

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

from src.masking_gnn_training import MultiTaskFinancialGAT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMaskingExperiment:
    """
    Conducts systematic evaluation of GNN robustness under information uncertainty.
    
    This class implements a testing framework that measures model
    performance degradation as network visibility decreases, providing insights
    into model reliability and prediction stability.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml",
                 cloud_config_path: str = "config/cloud_config.yaml",
                 networks_per_level: int = 200,
                 masking_levels: int = 10):
        """
        Initialize the masking experiment framework.
        
        Args:
            config_path: Path to model configuration file
            cloud_config_path: Path to cloud storage configuration
            networks_per_level: Number of networks to test per masking level
            masking_levels: Granularity of masking (3=minimal, 5=standard, 10=comprehensive)
        """
        self.config = self._load_config(config_path)
        self.cloud_config = self._load_config(cloud_config_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        if GCS_AVAILABLE:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(
                self.cloud_config['gcp']['storage_bucket']
            )
        else:
            raise ImportError("Google Cloud Storage client not available")
        
        self.model = None
        self.nodes_df = None
        self.ground_truth_map = {}
        
        self.networks_per_level = networks_per_level
        if masking_levels == 3:
            self.masking_levels = [0.9, 0.5, 0.1]
        elif masking_levels == 5:
            self.masking_levels = [0.9, 0.7, 0.5, 0.3, 0.1]
        else:
            self.masking_levels = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        self.random_seeds = [42, 123, 456, 789, 999, 1337, 2021, 8888, 5555, 7777]
        self.results = []
        
        self.base_node_features = None
        self.node_id_map = None
        self.feature_scaler = None
        self.feature_columns = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return {}
    
    def load_best_model(self) -> bool:
        """Load the trained GNN model for evaluation."""
        try:
            model_base_dir = Path("models/masking_gnn")
            potential_models = list(model_base_dir.glob("**/best_model.pth"))
            
            if not potential_models:
                logger.error(f"No model checkpoint found in {model_base_dir}")
                return False
            
            latest_model_path = max(potential_models, key=lambda p: p.parent.stat().st_mtime)
            
            config_path = latest_model_path.parent / "config_used.yaml"
            if not config_path.exists():
                logger.error(f"Model configuration file not found: {config_path}")
                return False
            
            with open(config_path, 'r') as f:
                model_config = yaml.safe_load(f)
            
            checkpoint = torch.load(latest_model_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint['model_state_dict']
            
            input_dim = state_dict['input_projection.0.weight'].shape[1]
            edge_dim = state_dict['gat_layers.0.lin_edge.weight'].shape[1]
            
            self.model = MultiTaskFinancialGAT(
                input_dim=input_dim,
                edge_dim=edge_dim,
                **model_config['model_architecture']
            )
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {latest_model_path}")
            logger.info(f"Architecture: input_dim={input_dim}, edge_dim={edge_dim}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            return False
    
    def load_nodes_data(self) -> bool:
        """Load and preprocess node features for efficient access."""
        try:
            nodes_path = Path("data/processed/nodes.csv")
            if not nodes_path.exists():
                logger.error(f"Node data file not found: {nodes_path}")
                return False
            
            self.nodes_df = pd.read_csv(nodes_path)
            
            pure_feature_cols = ['total_assets', 'capital', 'capital_ratio', 'bank_tier', 'lgd', 'leverage_ratio']
            self.feature_columns = [col for col in pure_feature_cols if col in self.nodes_df.columns]
            
            if not self.feature_columns:
                logger.error("No valid feature columns found in node data")
                return False
            
            node_features_df = self.nodes_df[self.feature_columns].copy()
            
            for col in node_features_df.columns:
                if node_features_df[col].isnull().any():
                    median = node_features_df[col][node_features_df[col] > 0].median()
                    node_features_df[col].fillna(median if not pd.isna(median) else 0, inplace=True)
            
            self.feature_scaler = RobustScaler()
            self.base_node_features = self.feature_scaler.fit_transform(node_features_df)
            
            self.node_id_map = {int(node_id): i for i, node_id in enumerate(self.nodes_df['node_id'])}
            
            logger.info(f"Preprocessed {len(self.nodes_df)} nodes with {len(self.feature_columns)} features")
            return True
            
        except Exception as e:
            logger.error(f"Node data loading failed: {e}", exc_info=True)
            return False

    def load_ground_truth_data(self):
        """Load ground truth results for the holdout test set."""
        gcs_path = "gs://financial-networks-multilayer-unique123/final_test/results/debtrank_results.json"
        logger.info(f"Loading ground truth data from {gcs_path}")
        
        try:
            blob_name = gcs_path.split(self.bucket.name + '/')[-1]
            blob = self.bucket.blob(blob_name)
            all_ground_truth = json.loads(blob.download_as_text())
            
            holdout_data = [
                item for item in all_ground_truth
                if 17000 <= item.get('network_id', -1) <= 17999
            ]
            
            self.ground_truth_map = {
                (item['network_id'], item['scenario_id']): item
                for item in holdout_data
            }
            
            logger.info(f"Loaded {len(self.ground_truth_map)} ground truth entries")

        except Exception as e:
            logger.error(f"Ground truth data loading failed: {e}", exc_info=True)
            raise

    def get_diverse_networks_for_experiment(self) -> List[Tuple[str, int, str]]:
        """
        Sample a diverse set of networks for robust statistical analysis.
        
        Uses stratified sampling to ensure representative coverage of the network space.
        
        Returns:
            List of tuples containing (blob_name, network_id, scenario_id)
        """
        try:
            all_gt_entries = [(nid, sid) for (nid, sid) in self.ground_truth_map.keys()]
            
            if not all_gt_entries:
                logger.error("No ground truth entries available")
                return []
            
            network_groups = {}
            for nid, sid in all_gt_entries:
                if nid not in network_groups:
                    network_groups[nid] = []
                network_groups[nid].append(sid)
            
            available_networks = list(network_groups.keys())
            target_networks = min(self.networks_per_level, len(available_networks))
            
            np.random.seed(42)
            
            sorted_networks = sorted(available_networks)
            step_size = len(sorted_networks) // target_networks
            
            selected_network_ids = []
            for i in range(target_networks):
                idx = min(i * step_size, len(sorted_networks) - 1)
                selected_network_ids.append(sorted_networks[idx])
            
            remaining_networks = [nid for nid in sorted_networks if nid not in selected_network_ids]
            if len(selected_network_ids) < target_networks and remaining_networks:
                additional_count = min(target_networks - len(selected_network_ids), len(remaining_networks))
                additional_networks = np.random.choice(remaining_networks, size=additional_count, replace=False)
                selected_network_ids.extend(additional_networks)
            
            sampled_networks = []
            for nid in selected_network_ids:
                scenarios = network_groups[nid]
                selected_scenario = scenarios[0]
                blob_name = f"final_test/networks/edge_lists/edges_sim_{nid:06d}.csv"
                sampled_networks.append((blob_name, nid, selected_scenario))
            
            logger.info(f"Sampled {len(sampled_networks)} networks from {len(available_networks)} available")
            return sampled_networks
            
        except Exception as e:
            logger.error(f"Network sampling failed: {e}", exc_info=True)
            return []
    
    def load_network_from_gcs(self, blob_name: str) -> Optional[pd.DataFrame]:
        """Load network edge list from cloud storage."""
        try:
            blob = self.bucket.blob(blob_name)
            if not blob.exists():
                logger.warning(f"Network file not found: {blob_name}")
                return None
            
            csv_content = blob.download_as_text()
            return pd.read_csv(StringIO(csv_content))
            
        except Exception as e:
            logger.error(f"Network loading failed for {blob_name}: {e}")
            return None
    
    def mask_edges(self, edges_df: pd.DataFrame, keep_ratio: float, seed: int) -> pd.DataFrame:
        """
        Apply random edge masking to simulate partial network visibility.
        
        Args:
            edges_df: DataFrame containing network edges
            keep_ratio: Fraction of edges to retain (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame containing the subset of retained edges
        """
        np.random.seed(seed)
        total_edges = len(edges_df)
        edges_to_keep = int(total_edges * keep_ratio)
        
        if edges_to_keep == 0:
            return edges_df.iloc[:0].copy()
        
        keep_indices = np.random.choice(total_edges, size=edges_to_keep, replace=False)
        return edges_df.iloc[keep_indices].copy()
    
    def _prepare_node_features_with_shock_fast(self, shocked_bank_id: int) -> np.ndarray:
        """
        Prepare node features with shock indicator applied to specified bank.
        
        Args:
            shocked_bank_id: ID of the bank receiving the initial shock
            
        Returns:
            Array of node features with shock indicators
        """
        node_features = self.base_node_features.copy()
        
        shocked_bank_idx = self.node_id_map.get(shocked_bank_id)
        if shocked_bank_idx is not None:
            try:
                if 'capital' in self.feature_columns:
                    capital_idx = self.feature_columns.index('capital')
                    node_features[shocked_bank_idx, capital_idx] = 0.0
                if 'total_assets' in self.feature_columns:
                    assets_idx = self.feature_columns.index('total_assets')
                    node_features[shocked_bank_idx, assets_idx] = 0.0
                if 'capital_ratio' in self.feature_columns:
                    ratio_idx = self.feature_columns.index('capital_ratio')
                    node_features[shocked_bank_idx, ratio_idx] = 0.0
            except ValueError as e:
                logger.warning(f"Feature column not found for shock application: {e}")
        
        shock_indicator = np.zeros((node_features.shape[0], 1))
        if shocked_bank_idx is not None:
            shock_indicator[shocked_bank_idx] = 1.0
        
        return np.concatenate([node_features, shock_indicator], axis=1)
    
    def _process_edges_for_gnn(self, edges_df: pd.DataFrame) -> Tuple[List, List]:
        """
        Convert edge DataFrame to GNN-compatible format with features.
        
        Args:
            edges_df: DataFrame containing network edges
            
        Returns:
            Tuple of (edge_list, edge_features)
        """
        layer_mapping = {
            'derivatives': 0, 'securities': 1, 'fx': 2, 
            'deposits_loans': 3, 'firesale': 4, 'default': 5
        }
        
        edge_list, edge_features = [], []
        
        if edges_df.empty:
            return edge_list, edge_features
        
        weights = edges_df['weight'].values
        weight_percentiles = np.percentile(weights, [25, 50, 75, 90, 95]) if len(weights) > 0 else [0]*5
        
        for _, edge in edges_df.iterrows():
            source_idx = self.node_id_map.get(int(edge['source']))
            target_idx = self.node_id_map.get(int(edge['target']))
            
            if source_idx is not None and target_idx is not None:
                edge_list.append([source_idx, target_idx])
                
                one_hot_layer = [0] * len(layer_mapping)
                layer_idx = layer_mapping.get(edge['layer'], 5)
                one_hot_layer[layer_idx] = 1
                
                weight = edge['weight']
                edge_feature_vector = [
                    np.log1p(weight),
                    np.searchsorted(weight_percentiles, weight) / len(weight_percentiles),
                    1 if weight > weight_percentiles[3] else 0,
                    1 if weight > weight_percentiles[4] else 0
                ] + one_hot_layer
                
                edge_features.append(edge_feature_vector)
        
        return edge_list, edge_features
    
    def prepare_gnn_input_fast(self, edges_df: pd.DataFrame, shocked_bank_id: int) -> Optional[Data]:
        """
        Prepare complete GNN input data structure.
        
        Args:
            edges_df: DataFrame containing network edges
            shocked_bank_id: ID of the initially shocked bank
            
        Returns:
            PyTorch Geometric Data object ready for model input
        """
        try:
            edge_list, edge_features = self._process_edges_for_gnn(edges_df)
            
            if not edge_list:
                logger.warning("No valid edges found for GNN input")
                return None
            
            node_features = self._prepare_node_features_with_shock_fast(shocked_bank_id)
            
            return Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_features, dtype=torch.float)
            )
            
        except Exception as e:
            logger.error(f"GNN input preparation failed: {e}")
            return None
    
    def run_gnn_prediction(self, data: Data) -> Optional[Dict[str, Any]]:
        """
        Execute GNN model prediction.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Dictionary containing predictions for all tasks
        """
        try:
            data = data.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(data.x, data.edge_index, data.edge_attr)
                
                default_probs = torch.sigmoid(outputs['default_logits']).cpu().numpy().flatten()
                systemic_risk_pred = outputs['systemic_risk'].cpu().numpy().flatten()[0]
                
                return {
                    'default_probabilities': default_probs,
                    'default_predictions': (default_probs > 0.5).astype(int),
                    'systemic_risk_prediction': systemic_risk_pred
                }
                
        except Exception as e:
            logger.error(f"GNN prediction failed: {e}", exc_info=True)
            return None
    
    def calculate_metrics(self, gnn_results: Dict, ground_truth: Dict) -> Dict[str, float]:
        """
        Calculate performance metrics comparing predictions to ground truth.
        
        Args:
            gnn_results: Dictionary of model predictions
            ground_truth: Dictionary of true values
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            sr_true = ground_truth['systemic_risk']
            sr_pred = gnn_results['systemic_risk_prediction']
            
            default_true = np.array(ground_truth['binary_defaults'])
            default_pred = gnn_results['default_predictions']
            
            min_len = min(len(default_true), len(default_pred))
            default_true = default_true[:min_len]
            default_pred = default_pred[:min_len]
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                default_true, default_pred, average='binary', zero_division=0
            )
            
            return {
                'systemic_risk_rmse': np.sqrt(mean_squared_error([sr_true], [sr_pred])),
                'systemic_risk_mae': mean_absolute_error([sr_true], [sr_pred]),
                'node_precision': precision,
                'node_recall': recall,
                'node_f1': f1,
                'total_defaults_true': int(np.sum(default_true)),
                'total_defaults_pred': int(np.sum(default_pred))
            }
            
        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            return {}
    
    def run_experiment_enhanced(self) -> bool:
        """
        Execute the complete masking experiment with comprehensive statistical sampling.
        
        Returns:
            True if experiment completed successfully, False otherwise
        """
        try:
            logger.info("="*60)
            logger.info("STARTING ENHANCED MASKING EXPERIMENT")
            logger.info("="*60)
            
            if not self.load_best_model():
                logger.error("Model loading failed")
                return False
            
            if not self.load_nodes_data():
                logger.error("Node data loading failed")
                return False
            
            self.load_ground_truth_data()
            
            sampled_networks = self.get_diverse_networks_for_experiment()
            if not sampled_networks:
                logger.error("No networks available for experiment")
                return False
            
            logger.info("Caching network data from cloud storage...")
            network_data_cache = {}
            
            for blob_name, nid, sid in tqdm(sampled_networks, desc="Loading networks"):
                edges_df = self.load_network_from_gcs(blob_name)
                if edges_df is not None:
                    network_data_cache[(nid, sid)] = edges_df
            
            logger.info(f"Successfully cached {len(network_data_cache)} networks")
            
            if not network_data_cache:
                logger.error("No network data could be loaded")
                return False
            
            total_experiments = len(self.masking_levels) * len(self.random_seeds) * len(network_data_cache)
            logger.info(f"Total experiments planned: {total_experiments}")
            logger.info(f"Configuration: {len(self.masking_levels)} masking levels, {len(self.random_seeds)} seeds, {len(network_data_cache)} networks")
            
            experiment_count = 0
            start_time = time.time()
            
            for masking_level in self.masking_levels:
                logger.info(f"\nProcessing masking level: {masking_level:.1%} edges visible")
                level_results = []
                
                for seed in self.random_seeds:
                    logger.info(f"  Running with seed {seed}")
                    
                    for (nid, sid), edges_df in network_data_cache.items():
                        experiment_count += 1
                        
                        if experiment_count % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = experiment_count / elapsed
                            eta = (total_experiments - experiment_count) / rate if rate > 0 else 0
                            logger.info(f"    Progress: {experiment_count}/{total_experiments} ({experiment_count/total_experiments*100:.1f}%) - ETA: {eta/60:.1f}m")
                        
                        try:
                            gt_entry = self.ground_truth_map[(nid, sid)]
                            
                            masked_edges_df = self.mask_edges(edges_df, masking_level, seed)
                            
                            gnn_data = self.prepare_gnn_input_fast(masked_edges_df, gt_entry['initial_shock_bank'])
                            if gnn_data is None:
                                continue
                            
                            gnn_results = self.run_gnn_prediction(gnn_data)
                            if gnn_results is None:
                                continue
                            
                            metrics = self.calculate_metrics(gnn_results, gt_entry)
                            if not metrics:
                                continue
                            
                            result_entry = {
                                'network_id': nid,
                                'scenario_id': sid,
                                'masking_level': masking_level,
                                'seed': seed,
                                'original_edges': len(edges_df),
                                'masked_edges': len(masked_edges_df),
                                'edge_retention_ratio': len(masked_edges_df) / len(edges_df) if len(edges_df) > 0 else 0,
                                **metrics
                            }
                            
                            level_results.append(result_entry)
                            
                        except Exception as e:
                            logger.warning(f"Error processing network {nid}, scenario {sid}, seed {seed}: {e}")
                            continue
                
                self.results.extend(level_results)
                if level_results:
                    level_df = pd.DataFrame(level_results)
                    mean_sr_rmse = level_df['systemic_risk_rmse'].mean()
                    std_sr_rmse = level_df['systemic_risk_rmse'].std()
                    mean_f1 = level_df['node_f1'].mean()
                    std_f1 = level_df['node_f1'].std()
                    
                    logger.info(f"  Level {masking_level:.1%} completed: {len(level_results)} results")
                    logger.info(f"    SR RMSE: {mean_sr_rmse:.4f} ± {std_sr_rmse:.4f}")
                    logger.info(f"    Node F1: {mean_f1:.4f} ± {std_f1:.4f}")
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            total_time = time.time() - start_time
            logger.info(f"Total time: {total_time/60:.1f} minutes")
            logger.info(f"Total results: {len(self.results)}")
            logger.info(f"Average time per experiment: {total_time/len(self.results):.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            return False
    
    def save_results(self) -> bool:
        """Save experiment results to cloud storage."""
        try:
            if not self.results:
                logger.warning("No results to save")
                return False
            
            results_df = pd.DataFrame(self.results)
            csv_buffer = StringIO()
            results_df.to_csv(csv_buffer, index=False)
            
            self.bucket.blob('final_test/results/enhanced_masking_experiment_results.csv').upload_from_string(
                csv_buffer.getvalue(), content_type='text/csv'
            )
            
            agg_results = self._calculate_aggregated_metrics_enhanced(results_df)
            json_buffer = json.dumps(agg_results, indent=2)
            
            self.bucket.blob('final_test/results/enhanced_masking_experiment_summary.json').upload_from_string(
                json_buffer, content_type='application/json'
            )
            
            logger.info("Results saved to cloud storage successfully")
            return True
            
        except Exception as e:
            logger.error(f"Results saving failed: {e}", exc_info=True)
            return False
    
    def _calculate_aggregated_metrics_enhanced(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive aggregated metrics with variance analysis.
        
        Args:
            results_df: DataFrame containing all experiment results
            
        Returns:
            Dictionary of aggregated statistics and analyses
        """
        summary = {}
        
        for level in sorted(results_df['masking_level'].unique(), reverse=True):
            level_data = results_df[results_df['masking_level'] == level]
            
            if len(level_data) > 0:
                summary[f'{level:.0%}_visible'] = {
                    'count': len(level_data),
                    'unique_networks': level_data['network_id'].nunique(),
                    'unique_seeds': level_data['seed'].nunique(),
                    
                    'systemic_risk_rmse_mean': float(level_data['systemic_risk_rmse'].mean()),
                    'systemic_risk_rmse_std': float(level_data['systemic_risk_rmse'].std()),
                    'systemic_risk_rmse_median': float(level_data['systemic_risk_rmse'].median()),
                    'systemic_risk_rmse_q25': float(level_data['systemic_risk_rmse'].quantile(0.25)),
                    'systemic_risk_rmse_q75': float(level_data['systemic_risk_rmse'].quantile(0.75)),
                    
                    'systemic_risk_mae_mean': float(level_data['systemic_risk_mae'].mean()),
                    'systemic_risk_mae_std': float(level_data['systemic_risk_mae'].std()),
                    
                    'node_f1_mean': float(level_data['node_f1'].mean()),
                    'node_f1_std': float(level_data['node_f1'].std()),
                    'node_f1_median': float(level_data['node_f1'].median()),
                    'node_f1_q25': float(level_data['node_f1'].quantile(0.25)),
                    'node_f1_q75': float(level_data['node_f1'].quantile(0.75)),
                    
                    'node_precision_mean': float(level_data['node_precision'].mean()),
                    'node_precision_std': float(level_data['node_precision'].std()),
                    'node_recall_mean': float(level_data['node_recall'].mean()),
                    'node_recall_std': float(level_data['node_recall'].std()),
                    
                    'avg_edge_retention': float(level_data['edge_retention_ratio'].mean()),
                    'std_edge_retention': float(level_data['edge_retention_ratio'].std())
                }
        
        summary['experiment_info'] = {
            'total_experiments': len(results_df),
            'unique_networks': results_df['network_id'].nunique(),
            'masking_levels_tested': sorted(results_df['masking_level'].unique()),
            'seeds_used': sorted(results_df['seed'].unique()),
            'networks_per_level': self.networks_per_level
        }
        
        if len(summary) > 1:
            baseline_level = max(results_df['masking_level'].unique())
            baseline_data = results_df[results_df['masking_level'] == baseline_level]
            
            if len(baseline_data) > 0:
                baseline_sr_rmse = baseline_data['systemic_risk_rmse'].mean()
                baseline_f1 = baseline_data['node_f1'].mean()
                
                summary['degradation_analysis'] = {}
                for level in sorted(results_df['masking_level'].unique(), reverse=True):
                    if level != baseline_level:
                        level_data = results_df[results_df['masking_level'] == level]
                        if len(level_data) > 0:
                            level_sr_rmse = level_data['systemic_risk_rmse'].mean()
                            level_f1 = level_data['node_f1'].mean()
                            
                            summary['degradation_analysis'][f'{level:.0%}_vs_baseline'] = {
                                'sr_rmse_relative_increase': float((level_sr_rmse - baseline_sr_rmse) / baseline_sr_rmse * 100),
                                'f1_relative_decrease': float((baseline_f1 - level_f1) / baseline_f1 * 100)
                            }
        
        return summary
    
    def create_enhanced_plots(self) -> bool:
        """Create comprehensive visualization plots for experiment results."""
        try:
            if not self.results:
                logger.warning("No results available for plotting")
                return False
            
            results_df = pd.DataFrame(self.results)
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Enhanced Masking Experiment Results - Variance Analysis', fontsize=16)
            
            grouped = results_df.groupby('masking_level').agg({
                'systemic_risk_rmse': ['mean', 'std', 'count'],
                'systemic_risk_mae': ['mean', 'std'],
                'node_f1': ['mean', 'std'],
                'node_precision': ['mean', 'std'],
                'node_recall': ['mean', 'std']
            })
            
            masking_levels = grouped.index.values * 100
            
            means = grouped[('systemic_risk_rmse', 'mean')]
            stds = grouped[('systemic_risk_rmse', 'std')]
            counts = grouped[('systemic_risk_rmse', 'count')]
            sem = stds / np.sqrt(counts)
            
            axes[0,0].errorbar(masking_levels, means, yerr=stds, 
                             marker='o', capsize=5, linewidth=2, markersize=6, 
                             label='± 1 STD', alpha=0.7)
            axes[0,0].errorbar(masking_levels, means, yerr=sem, 
                             marker='o', capsize=3, linewidth=1, markersize=4, 
                             label='± 1 SEM', color='red')
            axes[0,0].set_xlabel('% of Edges Visible')
            axes[0,0].set_ylabel('Systemic Risk RMSE')
            axes[0,0].set_title('Systemic Risk Prediction Error vs. Network Visibility')
            axes[0,0].legend()
            axes[0,0].invert_xaxis()
            axes[0,0].grid(True, alpha=0.3)
            
            axes[0,1].plot(masking_levels, stds, 'ro-', linewidth=2, markersize=6)
            axes[0,1].set_xlabel('% of Edges Visible')
            axes[0,1].set_ylabel('Standard Deviation of SR RMSE')
            axes[0,1].set_title('Prediction Uncertainty vs. Network Visibility')
            axes[0,1].invert_xaxis()
            axes[0,1].grid(True, alpha=0.3)
            
            f1_means = grouped[('node_f1', 'mean')]
            f1_stds = grouped[('node_f1', 'std')]
            
            axes[0,2].errorbar(masking_levels, f1_means, yerr=f1_stds, 
                             marker='^', capsize=5, linewidth=2, markersize=6, color='green')
            axes[0,2].set_xlabel('% of Edges Visible')
            axes[0,2].set_ylabel('Node-Level F1 Score')
            axes[0,2].set_title('Node Default Prediction F1 vs. Network Visibility')
            axes[0,2].invert_xaxis()
            axes[0,2].grid(True, alpha=0.3)
            
            axes[1,0].plot(masking_levels, f1_stds, 'go-', linewidth=2, markersize=6)
            axes[1,0].set_xlabel('% of Edges Visible')
            axes[1,0].set_ylabel('Standard Deviation of F1 Score')
            axes[1,0].set_title('F1 Score Uncertainty vs. Network Visibility')
            axes[1,0].invert_xaxis()
            axes[1,0].grid(True, alpha=0.3)
            
            axes[1,1].bar(masking_levels, counts, alpha=0.7, color='purple')
            axes[1,1].set_xlabel('% of Edges Visible')
            axes[1,1].set_ylabel('Number of Experiments')
            axes[1,1].set_title('Sample Size per Masking Level')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3)
            
            cv_sr = stds / means
            cv_f1 = f1_stds / f1_means
            
            axes[1,2].plot(masking_levels, cv_sr, 'bo-', linewidth=2, label='SR RMSE CV', markersize=6)
            axes[1,2].plot(masking_levels, cv_f1, 'go-', linewidth=2, label='F1 Score CV', markersize=6)
            axes[1,2].set_xlabel('% of Edges Visible')
            axes[1,2].set_ylabel('Coefficient of Variation')
            axes[1,2].set_title('Relative Variability vs. Network Visibility')
            axes[1,2].legend()
            axes[1,2].invert_xaxis()
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            self.bucket.blob('final_test/results/enhanced_masking_experiment_plots.png').upload_from_file(
                img_buffer, content_type='image/png'
            )
            plt.close()
            
            self._create_distribution_plots(results_df)
            
            logger.info("Plots created and saved to cloud storage")
            return True
            
        except Exception as e:
            logger.error(f"Plot creation failed: {e}", exc_info=True)
            return False
    
    def _create_distribution_plots(self, results_df: pd.DataFrame):
        """Create box plots showing performance distribution at each masking level."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Performance Distribution Analysis Across Masking Levels', fontsize=16)
            
            masking_levels = sorted(results_df['masking_level'].unique(), reverse=True)
            sr_data = [results_df[results_df['masking_level'] == level]['systemic_risk_rmse'].values 
                      for level in masking_levels]
            
            box1 = axes[0].boxplot(sr_data, labels=[f'{level:.0%}' for level in masking_levels], 
                                  patch_artist=True, showmeans=True)
            axes[0].set_xlabel('% of Edges Visible')
            axes[0].set_ylabel('Systemic Risk RMSE')
            axes[0].set_title('Distribution of SR RMSE Across Masking Levels')
            axes[0].grid(True, alpha=0.3)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(box1['boxes'])))
            for patch, color in zip(box1['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            f1_data = [results_df[results_df['masking_level'] == level]['node_f1'].values 
                      for level in masking_levels]
            
            box2 = axes[1].boxplot(f1_data, labels=[f'{level:.0%}' for level in masking_levels], 
                                  patch_artist=True, showmeans=True)
            axes[1].set_xlabel('% of Edges Visible')
            axes[1].set_ylabel('Node F1 Score')
            axes[1].set_title('Distribution of F1 Scores Across Masking Levels')
            axes[1].grid(True, alpha=0.3)
            
            colors = plt.cm.plasma(np.linspace(0, 1, len(box2['boxes'])))
            for patch, color in zip(box2['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            plt.tight_layout()
            
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            self.bucket.blob('final_test/results/enhanced_masking_distributions.png').upload_from_file(
                img_buffer, content_type='image/png'
            )
            plt.close()
            
        except Exception as e:
            logger.warning(f"Distribution plot creation failed: {e}")
    
    def debug_variance_patterns(self, results_df: pd.DataFrame):
        """Analyze and report detailed variance patterns across masking levels."""
        logger.info("\n" + "="*60)
        logger.info("DETAILED VARIANCE ANALYSIS")
        logger.info("="*60)
        
        for level in sorted(results_df['masking_level'].unique(), reverse=True):
            level_data = results_df[results_df['masking_level'] == level]
            
            if len(level_data) > 0:
                sr_rmse = level_data['systemic_risk_rmse']
                f1_scores = level_data['node_f1']
                
                logger.info(f"\n{level:.0%} edges visible ({len(level_data)} experiments):")
                logger.info(f"  Networks: {level_data['network_id'].nunique()} unique")
                logger.info(f"  Seeds: {level_data['seed'].nunique()} unique")
                logger.info(f"  SR RMSE: mean={sr_rmse.mean():.6f}, std={sr_rmse.std():.6f}, CV={sr_rmse.std()/sr_rmse.mean():.4f}")
                logger.info(f"  SR RMSE range: [{sr_rmse.min():.6f}, {sr_rmse.max():.6f}]")
                logger.info(f"  F1 Score: mean={f1_scores.mean():.6f}, std={f1_scores.std():.6f}, CV={f1_scores.std()/f1_scores.mean():.4f}")
                logger.info(f"  F1 range: [{f1_scores.min():.6f}, {f1_scores.max():.6f}]")


def main():
    """Main execution function for the enhanced masking experiment."""
    parser = argparse.ArgumentParser(description="Run enhanced masking experiment for GNN robustness evaluation")
    parser.add_argument('--config', type=str, default="config/model_config.yaml", 
                       help="Model configuration file path")
    parser.add_argument('--cloud-config', type=str, default="config/cloud_config.yaml", 
                       help="Cloud storage configuration file path")
    parser.add_argument('--networks-per-level', type=int, default=200, 
                       help="Number of networks to test per masking level (default: 200)")
    parser.add_argument('--masking-levels', type=int, default=5, 
                       help="Masking granularity: 3=minimal, 5=standard, 10=comprehensive (default: 5)")
    args = parser.parse_args()
    
    try:
        print(f"Configuration:")
        print(f"  Networks per level: {args.networks_per_level}")
        print(f"  Masking levels: {args.masking_levels}")
        print(f"  Seeds per experiment: 10")
        print(f"  Estimated total experiments: {args.networks_per_level * args.masking_levels * 10}")
        
        experiment = EnhancedMaskingExperiment(
            config_path=args.config,
            cloud_config_path=args.cloud_config,
            networks_per_level=args.networks_per_level,
            masking_levels=args.masking_levels
        )
        
        start_time = time.time()
        
        if experiment.run_experiment_enhanced():
            total_time = time.time() - start_time
            
            print(f"Total time: {total_time/60:.1f} minutes")
            print(f"Total results: {len(experiment.results)}")
            
            if experiment.results:
                results_df = pd.DataFrame(experiment.results)
                experiment.debug_variance_patterns(results_df)
            
            if experiment.save_results():
                print("Results saved to cloud storage successfully")
            else:
                print("Warning: Results saving failed")
            
            if experiment.create_enhanced_plots():
                print("Plots created and saved successfully")
            else:
                print("Warning: Plot creation failed")
            
            print("RESULTS AVAILABLE AT:")
            print(f"  CSV: gs://{experiment.bucket.name}/final_test/results/enhanced_masking_experiment_results.csv")
            print(f"  Summary: gs://{experiment.bucket.name}/final_test/results/enhanced_masking_experiment_summary.json")
            print(f"  Plots: gs://{experiment.bucket.name}/final_test/results/enhanced_masking_experiment_plots.png")
            print(f"  Distributions: gs://{experiment.bucket.name}/final_test/results/enhanced_masking_distributions.png")
            
        else:
            print("Experiment execution failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Main execution failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())