"""
Enhanced Graph Neural Network Trainer for Financial Contagion Prediction

This module implements a  GNN pipeline for creating fast, explainable
surrogate models for DebtRank simulations with proper handling of financial data
characteristics and enhanced interpretability features.

Key capabilities:
- Proper handling of zeros as actual values in financial data
- Multi-task learning for systemic risk prediction
- Attention visualisation for model explainability
- Comparative benchmarking against DebtRank algorithm
- Enhanced feature engineering for financial networks

Project: Multi-Layer Financial Contagion Modelling with Graph Neural Networks
"""

import numpy as np
import pandas as pd
import logging
import time
import json
import argparse
import os
import gc
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import defaultdict

# PyTorch and PyTorch Geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

# Scikit-learn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    f1_score, roc_auc_score, confusion_matrix, 
    mean_squared_error, r2_score
)

# Visualisation and monitoring
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Cloud storage
try:
    import gcsfs
    GCSFS_AVAILABLE = True
except ImportError:
    GCSFS_AVAILABLE = False

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configure comprehensive logging with file and console handlers.
    
    Args:
        output_dir: Directory for log file output
        log_level: Logging verbosity level
        
    Returns:
        Configured logger instance
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gnn_training_{timestamp}.log"
    
    logger = logging.getLogger("GNNTrainer")
    logger.setLevel(getattr(logging, log_level))
    logger.handlers = []
    
    # File handler with detailed formatting
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simplified formatting
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


class MultiTaskFinancialGAT(nn.Module):
    """
    Multi-Task Graph Attention Network for financial contagion prediction.
    
    Architecture supports dual prediction tasks:
    1. Node-level: Individual bank default probability (binary classification)
    2. Graph-level: Systemic risk magnitude (regression)
    
    Incorporates attention mechanisms for model explainability and interpretability.
    
    Args:
        input_dim: Dimension of input node features
        hidden_dim: Hidden layer dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads per layer
        dropout: Dropout probability
        edge_dim: Dimension of edge features
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 4, num_heads: int = 8, 
                 dropout: float = 0.3, edge_dim: int = 7):
        super(MultiTaskFinancialGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.default_pos_weight = None
        
        # Storage for attention weights (explainability)
        self.attention_weights = []
        
        # Input projection with domain-specific preprocessing
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Graph Attention layers with normalization
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First GAT layer
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim // num_heads, 
                   heads=num_heads, dropout=dropout, 
                   edge_dim=edge_dim, concat=True)
        )
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Intermediate GAT layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads,
                       heads=num_heads, dropout=dropout,
                       edge_dim=edge_dim, concat=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Final GAT layer
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim, heads=1,
                   dropout=dropout, edge_dim=edge_dim, concat=False)
        )
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Task 1: Individual bank default prediction head
        self.default_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Task 2: Graph-level systemic risk prediction head
        self.systemic_risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # Multi-scale pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights with appropriate schemes for financial data."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.out_features == 1:
                    # Conservative initialization for output layers
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task GAT model.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge feature matrix [num_edges, edge_dim]
            batch: Batch assignment vector for graph-level predictions
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing default logits, systemic risk predictions,
            and node embeddings. Optionally includes attention weights.
        """
        # Initialize attention storage if requested
        self.attention_weights = [] if return_attention else None
        
        # Project input features
        h = self.input_projection(x)
        
        # Process through GAT layers with residual connections
        for i, (gat_layer, layer_norm, dropout) in enumerate(
            zip(self.gat_layers, self.layer_norms, self.dropout_layers)
        ):
            h_residual = h
            
            # GAT layer with optional attention capture
            if return_attention:
                h, attention = gat_layer(h, edge_index, edge_attr=edge_attr, 
                                        return_attention_weights=True)
                self.attention_weights.append(attention)
            else:
                h = gat_layer(h, edge_index, edge_attr=edge_attr)
            
            h = F.relu(h)
            
            # Residual connection for dimension-matched layers
            if h.size(-1) == h_residual.size(-1):
                h = h + h_residual
            
            h = layer_norm(h)
            h = dropout(h)
        
        # Task 1: Node-level default prediction
        default_logits = self.default_predictor(h)
        
        # Task 2: Graph-level systemic risk prediction
        if batch is not None:
            # Multi-scale graph pooling
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_sum = global_add_pool(h, batch)
            h_graph = torch.cat([h_mean, h_max, h_sum], dim=-1)
            systemic_risk_pred = self.systemic_risk_predictor(h_graph)
        else:
            # Single graph case
            h_mean = h.mean(dim=0, keepdim=True)
            h_max = h.max(dim=0, keepdim=True)[0]
            h_sum = h.sum(dim=0, keepdim=True)
            h_graph = torch.cat([h_mean, h_max, h_sum], dim=-1)
            systemic_risk_pred = self.systemic_risk_predictor(h_graph)
        
        return {
            'default_logits': default_logits,
            'systemic_risk': systemic_risk_pred,
            'node_embeddings': h
        }


class EnhancedFinancialNetworkDataset:
    """
    Enhanced dataset handler with proper financial data characteristics.
    
    Key features:
    - Preserves zeros as actual values (not missing data)
    - Handles only true missing values (NaN)
    - Enhanced feature engineering for financial metrics
    - Stratified sampling by systemic risk levels
    - Advanced edge feature representation
    
    Args:
        networks_dir: Directory containing network edge lists
        ground_truth_path: Path to DebtRank simulation results
        nodes_path: Path to node attribute data
        cache_dir: Directory for caching processed data
        feature_config: Feature engineering configuration
        logger: Logger instance
        use_cloud: Whether to use cloud storage
    """
    
    def __init__(self, networks_dir: str, ground_truth_path: str, 
                 nodes_path: str, cache_dir: Path, feature_config: Dict,
                 logger: logging.Logger, use_cloud: bool = True):
        self.logger = logger
        self.use_cloud = use_cloud
        self.logger.info("Initializing Enhanced Financial Network Dataset...")
        
        # Initialize cloud storage if configured
        self.storage_client = None
        self.bucket = None
        if self.use_cloud:
            try:
                from google.cloud import storage
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket("financial-networks-multilayer-unique123")
                self.logger.info("Connected to Google Cloud Storage")
            except Exception as e:
                self.logger.warning(f"Could not connect to GCS: {e}")
                self.use_cloud = False
        
        # Load core data
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.nodes_df = self._load_nodes(nodes_path)
        
        # Setup paths and caching
        self.networks_dir = Path(networks_dir)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering configuration
        self.feature_config = feature_config
        self.feature_scaler = RobustScaler()
        self._prepare_enhanced_node_features()
        
        # Extract and stratify simulations
        self.unique_simulations = self._get_stratified_simulations()
        
        self.logger.info(f"Dataset initialized successfully:")
        self.logger.info(f"  - Total simulations: {len(self.unique_simulations)}")
        self.logger.info(f"  - Nodes: {len(self.nodes_df)}")
        self.logger.info(f"  - Enhanced features: {self.node_features.shape[1]}")
    
    def _load_ground_truth(self, path: str) -> List[Dict]:
        """
        Load ground truth data with comprehensive error handling.
        
        Supports both local and cloud storage paths (gs://).
        """
        self.logger.info(f"Loading ground truth from: {path}")
        
        try:
            if path.startswith('gs://'):
                try:
                    if GCSFS_AVAILABLE:
                        fs = gcsfs.GCSFileSystem()
                        with fs.open(path, 'r') as f:
                            data = json.load(f)
                    else:
                        raise ImportError("gcsfs not available")
                except Exception:
                    from google.cloud import storage
                    path_parts = path[5:].split('/', 1)
                    bucket_name = path_parts[0]
                    blob_name = path_parts[1] if len(path_parts) > 1 else ''
                    
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    
                    json_text = blob.download_as_text()
                    data = json.loads(json_text)
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
            
            if not data:
                raise ValueError("Ground truth file is empty")
            
            # Validate data structure
            required_keys = {'network_id', 'scenario_id', 'binary_defaults', 
                           'initial_shock_bank', 'systemic_risk'}
            for item in data[:5]:
                missing_keys = required_keys - set(item.keys())
                if missing_keys:
                    self.logger.warning(f"Missing keys in ground truth: {missing_keys}")
            
            self.logger.info(f"Loaded {len(data)} ground truth entries")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load ground truth: {e}")
            raise
    
    def _load_nodes(self, path: str) -> pd.DataFrame:
        """Load and validate node attribute data."""
        self.logger.info(f"Loading nodes from: {path}")
        
        try:
            nodes_df = pd.read_csv(path)
            
            if nodes_df.empty:
                raise ValueError("Nodes file is empty")
            
            self.logger.info(f"Loaded {len(nodes_df)} nodes with {len(nodes_df.columns)} columns")
            return nodes_df
            
        except Exception as e:
            self.logger.error(f"Failed to load nodes: {e}")
            raise
    
    def _prepare_enhanced_node_features(self):
        """
        Enhanced feature preparation with proper financial data handling.
        
        Key principles:
        - Zeros represent actual absence of exposure (preserved)
        - Only true missing values (NaN) are imputed
        - Log transformations for skewed distributions
        - Robust scaling for financial ratios
        """
        self.logger.info("Preparing enhanced node features...")
        
        feature_cols = self.feature_config['node_features']
        log_transform_cols = self.feature_config.get('log_transform_features', [])
        
        # Validate feature columns
        missing_cols = [col for col in feature_cols if col not in self.nodes_df.columns]
        if missing_cols:
            self.logger.warning(f"Missing feature columns: {missing_cols}")
            feature_cols = [col for col in feature_cols if col in self.nodes_df.columns]
        
        if not feature_cols:
            raise ValueError("No valid feature columns found")
        
        # Extract base features
        self.node_features_df = self.nodes_df[feature_cols].copy()
        
        # Handle only true missing values, preserve zeros
        nan_mask = self.node_features_df.isnull()
        if nan_mask.any().any():
            self.logger.info("Found NaN values, filling with column medians (preserving zeros)")
            for col in self.node_features_df.columns:
                if nan_mask[col].any():
                    non_zero_median = self.node_features_df[col][
                        self.node_features_df[col] > 0
                    ].median()
                    fill_value = non_zero_median if not pd.isna(non_zero_median) else 0
                    self.node_features_df[col].fillna(fill_value, inplace=True)
        
        # Apply log transformation to specified columns
        for col in log_transform_cols:
            if col in self.node_features_df.columns:
                self.node_features_df[col] = np.log1p(self.node_features_df[col].clip(0))
                self.logger.debug(f"Applied log1p transform to {col}")
        
        # Engineer additional financial features
        self._engineer_financial_features()
        
        # Apply robust scaling (handles outliers better)
        self.node_features = self.feature_scaler.fit_transform(self.node_features_df)
        
        self.logger.info(f"Prepared {self.node_features.shape[1]} enhanced features")
        self._log_feature_statistics()
    
    def _engineer_financial_features(self):
        """
        Engineer domain-specific financial features.
        
        Creates features based on financial theory:
        - Size metrics (asset percentiles, log-scale)
        - Interconnectedness measures (net positions, intensity)
        - Risk concentration (Herfindahl index)
        - Leverage and solvency ratios
        """
        self.logger.info("Engineering additional financial features...")
        
        # Size-based features
        if 'total_assets' in self.node_features_df.columns:
            asset_values = self.node_features_df['total_assets']
            self.node_features_df['asset_percentile'] = asset_values.rank(pct=True)
            self.node_features_df['log_assets'] = np.log1p(asset_values)
        
        # Interconnectedness features
        if all(col in self.node_features_df.columns for col in 
               ['total_interbank_assets', 'total_interbank_liabilities']):
            ib_assets = self.node_features_df['total_interbank_assets']
            ib_liab = self.node_features_df['total_interbank_liabilities']
            
            self.node_features_df['net_interbank_position'] = ib_assets - ib_liab
            self.node_features_df['interbank_intensity'] = ib_assets + ib_liab
            
            total_ib = ib_assets + ib_liab
            self.node_features_df['interbank_balance'] = np.where(
                total_ib > 0, 
                (ib_assets - ib_liab) / total_ib, 
                0
            )
        
        # Risk concentration features
        exposure_cols = [col for col in self.node_features_df.columns 
                        if col.endswith('_exposure')]
        if len(exposure_cols) > 1:
            exposures = self.node_features_df[exposure_cols]
            
            self.node_features_df['total_exposure'] = exposures.sum(axis=1)
            
            # Herfindahl concentration index
            total_exp = exposures.sum(axis=1)
            exposure_shares = exposures.div(total_exp, axis=0).fillna(0)
            self.node_features_df['exposure_concentration'] = (exposure_shares ** 2).sum(axis=1)
            
            self.node_features_df['dominant_exposure_pct'] = exposures.max(axis=1) / (total_exp + 1e-8)
        
        # Leverage features
        if all(col in self.node_features_df.columns for col in ['capital', 'total_assets']):
            capital = self.node_features_df['capital']
            assets = self.node_features_df['total_assets']
            
            self.node_features_df['leverage_ratio'] = np.where(
                capital > 0,
                assets / capital,
                assets
            )
        
        self.logger.info(f"Engineered features. New shape: {self.node_features_df.shape}")
    
    def _log_feature_statistics(self):
        """Log feature statistics for monitoring and validation."""
        stats = self.node_features_df.describe()
        self.logger.info("Feature statistics (pre-scaling):")
        
        for col in self.node_features_df.columns:
            col_stats = stats[col]
            zero_pct = (self.node_features_df[col] == 0).mean() * 100
            self.logger.debug(
                f"{col}: mean={col_stats['mean']:.3f}, "
                f"std={col_stats['std']:.3f}, zeros={zero_pct:.1f}%"
            )
    
    def _get_stratified_simulations(self) -> List[Tuple[int, str, float]]:
        """Extract simulations with systemic risk stratification."""
        simulations = []
        
        for item in self.ground_truth:
            network_id = item['network_id']
            scenario_id = item['scenario_id']
            systemic_risk = item.get('systemic_risk', 0.0)
            simulations.append((network_id, scenario_id, systemic_risk))
        
        # Log risk distribution
        systemic_risks = [s[2] for s in simulations]
        self.logger.info(
            f"Systemic risk distribution - "
            f"Mean: {np.mean(systemic_risks):.4f}, "
            f"Std: {np.std(systemic_risks):.4f}, "
            f"Range: [{np.min(systemic_risks):.4f}, {np.max(systemic_risks):.4f}]"
        )
        
        return simulations
    
    def get_simulation_data(self, network_id: int, scenario_id: str) -> Optional[Data]:
        """
        Create PyTorch Geometric Data object for a simulation.
        
        Args:
            network_id: Network identifier
            scenario_id: Scenario identifier
            
        Returns:
            PyG Data object with features, edges, and labels
        """
        try:
            # Find ground truth entry
            sim_truth = next(
                (item for item in self.ground_truth 
                 if item['network_id'] == network_id and item['scenario_id'] == scenario_id),
                None
            )
            
            if not sim_truth:
                return None
            
            # Load edge data
            edge_filename = f"edges_sim_{network_id:06d}.csv"
            edges_df = self._load_edge_file(edge_filename)
            
            if edges_df is None or edges_df.empty:
                return None
            
            # Process edges with enhanced features
            edge_list, edge_features = self._process_enhanced_edges(edges_df)
            
            if not edge_list:
                return None
            
            # Prepare node features with shock indicator
            node_features = self._prepare_node_features_with_shock(sim_truth)
            
            # Create PyG Data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                y=torch.tensor(sim_truth['binary_defaults'], dtype=torch.float).unsqueeze(1),
                systemic_risk=torch.tensor([sim_truth.get('systemic_risk', 0.0)], dtype=torch.float),
            )
            
            # Add metadata
            data.network_id = network_id
            data.scenario_id = scenario_id
            data.initial_shock_bank = sim_truth['initial_shock_bank']
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating data for network {network_id}: {e}")
            return None
    
    def _load_edge_file(self, edge_filename: str) -> Optional[pd.DataFrame]:
        """Load edge file from cache, local storage, or cloud."""
        # Check local cache
        local_edge_file = self.cache_dir / "edge_cache" / edge_filename
        if local_edge_file.exists():
            return pd.read_csv(local_edge_file)
        
        # Check original local path
        edge_file = self.networks_dir / edge_filename
        if edge_file.exists():
            return pd.read_csv(edge_file)
        
        # Try cloud storage
        if self.use_cloud and self.bucket:
            try:
                blob = self.bucket.blob(f"networks/edge_lists/{edge_filename}")
                if blob.exists():
                    csv_content = blob.download_as_text()
                    from io import StringIO
                    edges_df = pd.read_csv(StringIO(csv_content))
                    
                    # Cache locally
                    cache_path = self.cache_dir / "edge_cache"
                    cache_path.mkdir(parents=True, exist_ok=True)
                    cache_file = cache_path / edge_filename
                    edges_df.to_csv(cache_file, index=False)
                    
                    return edges_df
            except Exception as e:
                self.logger.warning(f"Could not load {edge_filename} from cloud: {e}")
        
        return None
    
    def _process_enhanced_edges(self, edges_df: pd.DataFrame) -> Tuple[List, List]:
        """
        Process edges with enhanced feature representation.
        
        Creates comprehensive edge features including:
        - Log-transformed weights
        - Percentile rankings
        - Size category indicators
        - Layer one-hot encoding
        """
        layer_mapping = {
            'derivatives': 0, 'securities': 1, 'fx': 2,
            'deposits_loans': 3, 'firesale': 4, 'default': 5
        }
        
        edge_list = []
        edge_features = []
        node_id_map = {node_id: i for i, node_id in enumerate(self.nodes_df['node_id'])}
        
        # Calculate edge statistics for normalization
        weights = edges_df['weight'].values
        log_weights = np.log1p(weights)
        weight_percentiles = np.percentile(weights, [25, 50, 75, 90, 95])
        
        for _, edge in edges_df.iterrows():
            source_idx = node_id_map.get(edge['source'])
            target_idx = node_id_map.get(edge['target'])
            
            if source_idx is None or target_idx is None:
                continue
            
            edge_list.append([source_idx, target_idx])
            
            weight = edge['weight']
            log_weight = np.log1p(weight)
            
            # Layer one-hot encoding
            one_hot_layer = [0] * len(layer_mapping)
            layer_idx = layer_mapping.get(edge['layer'], 5)
            one_hot_layer[layer_idx] = 1
            
            # Weight percentile
            weight_percentile = np.searchsorted(weight_percentiles, weight) / len(weight_percentiles)
            
            # Size category indicators
            is_large_exposure = 1 if weight > weight_percentiles[3] else 0
            is_systemically_important = 1 if weight > weight_percentiles[4] else 0
            
            # Combine features
            edge_feature_vector = (
                [log_weight, weight_percentile, is_large_exposure, is_systemically_important] + 
                one_hot_layer
            )
            
            edge_features.append(edge_feature_vector)
        
        return edge_list, edge_features
    
    def _prepare_node_features_with_shock(self, sim_truth: Dict) -> np.ndarray:
        """
        Prepare node features with initial shock applied.
        
        Modifies financial health metrics for the shocked institution
        to simulate initial distress state.
        
        Args:
            sim_truth: Ground truth dictionary with shock information
            
        Returns:
            Node feature matrix with shock indicator
        """
        # Create copy for this specific graph
        node_features_copy = self.node_features.copy()

        # Find shocked bank index
        node_id_map = {node_id: i for i, node_id in enumerate(self.nodes_df['node_id'])}
        shock_bank_idx = node_id_map.get(sim_truth['initial_shock_bank'])

        if shock_bank_idx is not None:
            # Find feature indices for financial health metrics
            feature_columns = list(self.node_features_df.columns)
            try:
                # Zero out financial health metrics to simulate failure
                capital_idx = feature_columns.index('capital')
                assets_idx = feature_columns.index('total_assets')
                capital_ratio_idx = feature_columns.index('capital_ratio')

                node_features_copy[shock_bank_idx, capital_idx] = 0.0
                node_features_copy[shock_bank_idx, assets_idx] = 0.0
                node_features_copy[shock_bank_idx, capital_ratio_idx] = 0.0

            except ValueError as e:
                self.logger.warning(f"Feature column not found for applying shock: {e}")

        # Add shock indicator as separate feature
        shock_indicator_feature = np.zeros((self.node_features.shape[0], 1))
        if shock_bank_idx is not None:
            shock_indicator_feature[shock_bank_idx] = 1.0

        return np.concatenate([node_features_copy, shock_indicator_feature], axis=1)
    
    def create_stratified_dataset_split(self, train_ratio: float = 0.8, 
                                      val_ratio: float = 0.1,
                                      random_seed: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Create stratified dataset split based on systemic risk levels.
        
        Uses stratified sampling to ensure balanced representation of
        different risk levels across train/validation/test splits.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data) lists
        """
        self.logger.info(f"Creating stratified dataset split: {train_ratio:.0%} train, {val_ratio:.0%} val")
        
        # Create risk level bins for stratification
        systemic_risks = [s[2] for s in self.unique_simulations]
        risk_bins = np.percentile(systemic_risks, [0, 33, 66, 100])
        
        # Assign risk levels
        risk_levels = []
        for _, _, risk in self.unique_simulations:
            if risk <= risk_bins[1]:
                risk_levels.append(0)  # Low risk
            elif risk <= risk_bins[2]:
                risk_levels.append(1)  # Medium risk
            else:
                risk_levels.append(2)  # High risk
        
        # Stratified split
        sim_array = np.array(self.unique_simulations, dtype=object)
        
        # First split: train+val vs test
        splitter1 = StratifiedShuffleSplit(n_splits=1, 
                                         train_size=train_ratio + val_ratio,
                                         random_state=random_seed)
        train_val_idx, test_idx = next(splitter1.split(sim_array, risk_levels))
        
        # Second split: train vs val
        train_val_risks = [risk_levels[i] for i in train_val_idx]
        splitter2 = StratifiedShuffleSplit(n_splits=1,
                                         train_size=train_ratio / (train_ratio + val_ratio),
                                         random_state=random_seed)
        train_idx_rel, val_idx_rel = next(splitter2.split(train_val_idx, train_val_risks))
        
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        
        # Get simulation tuples for each split
        train_sims = [self.unique_simulations[i] for i in train_idx]
        val_sims = [self.unique_simulations[i] for i in val_idx]
        test_sims = [self.unique_simulations[i] for i in test_idx]
        
        self.logger.info(f"Stratified split sizes - Train: {len(train_sims)}, Val: {len(val_sims)}, Test: {len(test_sims)}")
        
        # Process each split
        def process_split(simulations, split_name):
            data_list = []
            failed = 0
            
            for network_id, scenario_id, _ in tqdm(simulations, desc=f"Processing {split_name}"):
                data = self.get_simulation_data(network_id, scenario_id)
                if data is not None:
                    data_list.append(data)
                else:
                    failed += 1
            
            if failed > 0:
                self.logger.warning(f"{split_name}: {failed}/{len(simulations)} simulations failed to load")
            
            return data_list
        
        train_data = process_split(train_sims, "Train")
        val_data = process_split(val_sims, "Val")
        test_data = process_split(test_sims, "Test")
        
        # Validate splits
        if not train_data:
            raise ValueError("No training data loaded successfully")
        
        self.logger.info(f"Final split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Log split statistics
        self._log_split_statistics(train_data, val_data, test_data)
        
        return train_data, val_data, test_data
    
    def _log_split_statistics(self, train_data: List[Data], val_data: List[Data], test_data: List[Data]):
        """Log detailed statistics for each data split."""
        for data_list, name in [(train_data, "Train"), (val_data, "Val"), (test_data, "Test")]:
            if not data_list:
                continue
                
            all_labels = []
            systemic_risks = []
            
            for data in data_list:
                all_labels.extend(data.y.numpy().flatten())
                systemic_risks.append(data.systemic_risk.item())
            
            default_rate = np.mean(all_labels)
            avg_systemic_risk = np.mean(systemic_risks)
            std_systemic_risk = np.std(systemic_risks)
            
            self.logger.info(f"{name} split statistics:")
            self.logger.info(f"  Default rate: {default_rate:.2%}")
            self.logger.info(f"  Systemic risk - Mean: {avg_systemic_risk:.4f}, Std: {std_systemic_risk:.4f}")


class EnhancedGNNTrainer:
    """
    Enhanced GNN trainer with multi-task learning and explainability.
    
    Capabilities:
    - Multi-task learning (default prediction + systemic risk)
    - Attention visualization for model explainability
    - Comparative benchmarking against DebtRank
    - Comprehensive evaluation metrics
    - TensorBoard integration for training monitoring
    
    Args:
        config: Training configuration dictionary
        output_dir: Directory for outputs and checkpoints
        logger: Logger instance
    """
    
    def __init__(self, config: Dict, output_dir: Path, logger: logging.Logger):
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        
        # Device setup
        self.device = self._setup_device()
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.default_pos_weight = None
        
        # Loss functions
        self.systemic_risk_criterion = nn.MSELoss()
        self.default_criterion = nn.BCEWithLogitsLoss(pos_weight=self.default_pos_weight)
        
        # Loss weights for multi-task learning
        self.default_loss_weight = config.get('training', {}).get('default_loss_weight', 1.0)
        self.systemic_risk_loss_weight = config.get('training', {}).get('systemic_risk_loss_weight', 0.5)
        
        # Training history
        self.history = defaultdict(list)
        self.attention_history = []
        
        self.logger.info(f"Enhanced GNN Trainer initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Configure computing device with diagnostics."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            self.logger.warning("GPU not available, using CPU")
        return device
    
    def setup_model(self, input_dim: int, edge_dim: int):
        """
        Initialize enhanced multi-task model with optimized hyperparameters.
        
        Args:
            input_dim: Input feature dimension
            edge_dim: Edge feature dimension
        """
        self.logger.info(f"Setting up enhanced multi-task model - Input: {input_dim}, Edge: {edge_dim}")
        
        # Create model
        self.model = MultiTaskFinancialGAT(
            input_dim=input_dim,
            edge_dim=edge_dim,
            **self.config['model_architecture']
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Setup optimizer with component-specific learning rates
        param_groups = [
            {'params': self.model.gat_layers.parameters(), 'lr': self.config['optimizer']['lr']},
            {'params': self.model.default_predictor.parameters(), 'lr': self.config['optimizer']['lr'] * 0.5},
            {'params': self.model.systemic_risk_predictor.parameters(), 'lr': self.config['optimizer']['lr'] * 0.5}
        ]
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            **self.config['scheduler']
        )
        
        self.logger.info("Enhanced model setup complete")
    
    def calculate_class_weights(self, train_loader: DataLoader):
        """
        Calculate positive class weight to handle severe class imbalance.
        
        Adjusts loss function to account for imbalanced default rates
        in financial data.
        
        Args:
            train_loader: Training data loader
        """
        self.logger.info("Calculating class weights for imbalanced dataset...")
        num_positives = 0
        num_negatives = 0
        
        for batch in train_loader:
            num_positives += torch.sum(batch.y == 1)
            num_negatives += torch.sum(batch.y == 0)

        if num_positives == 0:
            self.logger.warning("No positive samples found in training data. Cannot set pos_weight.")
            return

        # Weight = (Number of negatives) / (Number of positives)
        pos_weight_value = num_negatives / num_positives
        self.default_pos_weight = torch.tensor([pos_weight_value], device=self.device)

        # Re-initialize loss function with calculated weight
        self.default_criterion = nn.BCEWithLogitsLoss(pos_weight=self.default_pos_weight)
        self.logger.info(f"Class imbalance detected. Setting pos_weight to: {self.default_pos_weight.item():.2f}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Execute enhanced training loop with multi-task learning.
        
        Implements comprehensive training with:
        - Multi-task loss optimization
        - Learning rate scheduling
        - Early stopping
        - TensorBoard monitoring
        - Periodic checkpointing
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.logger.info("Starting enhanced multi-task training...")
        
        # Setup TensorBoard
        log_dir = self.output_dir / f"tensorboard_{int(time.time())}"
        writer = SummaryWriter(log_dir)
        self.logger.info(f"TensorBoard logs: {log_dir}")
        
        # Training state
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        patience_counter = 0
        
        # Training configuration
        epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping_patience']
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch, epochs)
            
            # Validation phase
            val_metrics = self._validate(val_loader)
            
            # Update scheduler
            total_val_loss = val_metrics['total_loss']
            self.scheduler.step(total_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} "
                f"(Default: {train_metrics['default_loss']:.4f}, SR: {train_metrics['systemic_risk_loss']:.4f}) | "
                f"Val Loss: {total_val_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val SR-R²: {val_metrics['systemic_risk_r2']:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )
            
            # TensorBoard logging
            writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
            writer.add_scalar('Loss/Train_Default', train_metrics['default_loss'], epoch)
            writer.add_scalar('Loss/Train_SystemicRisk', train_metrics['systemic_risk_loss'], epoch)
            writer.add_scalar('Loss/Val_Total', total_val_loss, epoch)
            writer.add_scalar('Metrics/F1', val_metrics['f1'], epoch)
            writer.add_scalar('Metrics/ROC_AUC', val_metrics['roc_auc'], epoch)
            writer.add_scalar('Metrics/SystemicRisk_R2', val_metrics['systemic_risk_r2'], epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            
            # Check for improvement (weighted combination of F1 and R²)
            combined_metric = 0.7 * val_metrics['f1'] + 0.3 * max(0, val_metrics['systemic_risk_r2'])
            
            if combined_metric > best_val_f1:
                best_val_f1 = combined_metric
                best_val_loss = total_val_loss
                self._save_checkpoint('best_model.pth', epoch, val_metrics)
                patience_counter = 0
                self.logger.info(f"New best model saved (Combined metric: {combined_metric:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Periodic checkpoint and attention analysis
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_metrics)
                self._analyze_attention_patterns(val_loader, epoch)
        
        writer.close()
        self._save_history()
        
        self.logger.info(f"Training completed. Best combined metric: {best_val_f1:.4f}")
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Execute one training epoch with multi-task loss."""
        self.model.train()
        
        total_default_loss = 0
        total_systemic_risk_loss = 0
        total_loss = 0
        num_batches = len(train_loader)
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Multi-task loss computation
                default_loss = self.default_criterion(outputs['default_logits'], batch.y)
                systemic_risk_loss = self.systemic_risk_criterion(
                    outputs['systemic_risk'], 
                    batch.systemic_risk.unsqueeze(1) if batch.systemic_risk.dim() == 1 else batch.systemic_risk
                )
                
                # Combined weighted loss
                total_batch_loss = (
                    self.default_loss_weight * default_loss + 
                    self.systemic_risk_loss_weight * systemic_risk_loss
                )
                
                # Backward pass
                total_batch_loss.backward()
                
                # Gradient clipping
                if self.config.get('training', {}).get('gradient_clipping', False):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training'].get('max_grad_norm', 1.0)
                    )
                
                self.optimizer.step()
                
                # Update metrics
                total_default_loss += default_loss.item()
                total_systemic_risk_loss += systemic_risk_loss.item()
                total_loss += total_batch_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.4f}',
                    'def': f'{total_default_loss/(batch_idx+1):.4f}',
                    'sr': f'{total_systemic_risk_loss/(batch_idx+1):.4f}'
                })
        
        return {
            'total_loss': total_loss / num_batches,
            'default_loss': total_default_loss / num_batches,
            'systemic_risk_loss': total_systemic_risk_loss / num_batches
        }
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model with comprehensive multi-task metrics."""
        self.model.eval()
        
        total_default_loss = 0
        total_systemic_risk_loss = 0
        
        all_default_preds = []
        all_default_labels = []
        all_systemic_risk_preds = []
        all_systemic_risk_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                # Compute losses
                default_loss = self.default_criterion(outputs['default_logits'], batch.y)
                systemic_risk_loss = self.systemic_risk_criterion(
                    outputs['systemic_risk'], 
                    batch.systemic_risk.unsqueeze(1) if batch.systemic_risk.dim() == 1 else batch.systemic_risk
                )
                
                total_default_loss += default_loss.item()
                total_systemic_risk_loss += systemic_risk_loss.item()
                
                # Collect predictions
                all_default_preds.extend(torch.sigmoid(outputs['default_logits']).cpu().numpy())
                all_default_labels.extend(batch.y.cpu().numpy())
                all_systemic_risk_preds.extend(outputs['systemic_risk'].cpu().numpy())
                all_systemic_risk_labels.extend(batch.systemic_risk.cpu().numpy())
        
        # Calculate comprehensive metrics
        num_batches = len(val_loader)
        avg_default_loss = total_default_loss / num_batches
        avg_systemic_risk_loss = total_systemic_risk_loss / num_batches
        total_loss = (
            self.default_loss_weight * avg_default_loss + 
            self.systemic_risk_loss_weight * avg_systemic_risk_loss
        )
        
        # Default prediction metrics
        all_default_preds = np.array(all_default_preds).flatten()
        all_default_labels = np.array(all_default_labels).flatten()
        binary_preds = (all_default_preds > 0.5).astype(int)
        
        f1 = f1_score(all_default_labels, binary_preds, zero_division=0)
        roc_auc = roc_auc_score(all_default_labels, all_default_preds) if len(np.unique(all_default_labels)) > 1 else 0.0
        accuracy = np.mean(binary_preds == all_default_labels)
        
        # Systemic risk prediction metrics
        all_systemic_risk_preds = np.array(all_systemic_risk_preds).flatten()
        all_systemic_risk_labels = np.array(all_systemic_risk_labels).flatten()
        
        systemic_risk_r2 = r2_score(all_systemic_risk_labels, all_systemic_risk_preds)
        systemic_risk_rmse = np.sqrt(mean_squared_error(all_systemic_risk_labels, all_systemic_risk_preds))
        
        return {
            'total_loss': total_loss,
            'default_loss': avg_default_loss,
            'systemic_risk_loss': avg_systemic_risk_loss,
            'f1': f1,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'systemic_risk_r2': systemic_risk_r2,
            'systemic_risk_rmse': systemic_risk_rmse
        }
    
    def _analyze_attention_patterns(self, val_loader: DataLoader, epoch: int):
        """
        Analyse attention patterns for model explainability.
        
        Extracts and analyzes attention weights to understand which
        network connections the model focuses on during prediction.
        """
        self.model.eval()
        
        attention_analysis = {
            'epoch': epoch,
            'high_attention_edges': [],
            'attention_by_layer_type': defaultdict(list),
            'shocked_bank_attention': []
        }
        
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            sample_batch = sample_batch.to(self.device)
            
            # Get model output with attention weights
            outputs = self.model(
                sample_batch.x, 
                sample_batch.edge_index, 
                sample_batch.edge_attr, 
                sample_batch.batch,
                return_attention=True
            )
            
            if hasattr(self.model, 'attention_weights') and self.model.attention_weights:
                # Analyze attention patterns
                for layer_idx, (edge_index, attention_weights) in enumerate(self.model.attention_weights):
                    attention_scores = attention_weights.cpu().numpy()

                    if attention_scores.ndim == 2:
                        attention_scores = attention_scores.mean(axis=1)

                    high_attention_threshold = np.percentile(attention_scores, 95)
                    high_attention_edges = edge_index[:, attention_scores > high_attention_threshold]
                    
                    attention_analysis['high_attention_edges'].append({
                        'layer': layer_idx,
                        'edges': high_attention_edges.cpu().tolist(),
                        'scores': attention_scores[attention_scores > high_attention_threshold].tolist()
                    })
        
        # Store attention analysis
        self.attention_history.append(attention_analysis)
        
        # Save attention analysis periodically
        if epoch % 20 == 0:
            attention_file = self.output_dir / f'attention_analysis_epoch_{epoch}.json'
            with open(attention_file, 'w') as f:
                json.dump(attention_analysis, f, indent=2)
    
    def evaluate_comprehensive(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Execute comprehensive evaluation with detailed metrics.
        
        Provides extensive performance analysis including:
        - Default prediction accuracy metrics
        - Systemic risk prediction quality
        - Risk-stratified performance
        - Inference timing analysis
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        self.logger.info("Running comprehensive evaluation...")
        
        self.model.eval()
        
        # Collect all predictions and ground truth
        all_results = {
            'network_ids': [],
            'scenario_ids': [],
            'default_predictions': [],
            'default_probabilities': [],
            'default_ground_truth': [],
            'systemic_risk_predictions': [],
            'systemic_risk_ground_truth': [],
            'inference_times': [],
            'node_embeddings': []
        }
        
        total_inference_time = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                
                # Time inference
                start_time = time.time()
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_attention=True)
                inference_time = time.time() - start_time
                
                total_inference_time += inference_time
                
                # Collect results
                default_probs = torch.sigmoid(outputs['default_logits']).cpu().numpy().flatten()
                default_preds = (default_probs > 0.5).astype(int)
                
                all_results['default_predictions'].extend(default_preds)
                all_results['default_probabilities'].extend(default_probs)
                all_results['default_ground_truth'].extend(batch.y.cpu().numpy().flatten())
                all_results['systemic_risk_predictions'].extend(outputs['systemic_risk'].cpu().numpy().flatten())
                all_results['systemic_risk_ground_truth'].extend(batch.systemic_risk.cpu().numpy().flatten())
                all_results['inference_times'].append(inference_time)
                all_results['node_embeddings'].extend(outputs['node_embeddings'].cpu().numpy())
                
                # Collect metadata
                if hasattr(batch, 'network_id'):
                    all_results['network_ids'].extend([getattr(data, 'network_id', -1) for data in batch.to_data_list()])
                    all_results['scenario_ids'].extend([getattr(data, 'scenario_id', 'unknown') for data in batch.to_data_list()])
        
        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(all_results)
        
        # Add timing analysis
        results['timing'] = {
            'total_inference_time': total_inference_time,
            'average_inference_time': total_inference_time / len(test_loader),
            'predictions_per_second': len(all_results['default_predictions']) / total_inference_time
        }
        
        # Log and save results
        self._log_comprehensive_results(results)
        self._save_comprehensive_results(results, all_results)
        
        return results
    
    def _calculate_comprehensive_metrics(self, all_results: Dict) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics across all tasks."""
        # Default prediction metrics
        default_preds = np.array(all_results['default_predictions'])
        default_probs = np.array(all_results['default_probabilities'])
        default_true = np.array(all_results['default_ground_truth'])
        
        # Systemic risk prediction metrics
        sr_preds = np.array(all_results['systemic_risk_predictions'])
        sr_true = np.array(all_results['systemic_risk_ground_truth'])
        
        # Basic classification metrics
        f1 = f1_score(default_true, default_preds, zero_division=0)
        roc_auc = roc_auc_score(default_true, default_probs) if len(np.unique(default_true)) > 1 else 0.0
        accuracy = np.mean(default_preds == default_true)
        
        # Confusion matrix
        cm = confusion_matrix(default_true, default_preds)
        
        # Precision/Recall
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1] if cm.shape[1]>1 else 0, cm[1,0] if cm.shape[0]>1 else 0, cm[1,1] if cm.size==4 else 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Systemic risk regression metrics
        sr_r2 = r2_score(sr_true, sr_preds)
        sr_rmse = np.sqrt(mean_squared_error(sr_true, sr_preds))
        sr_mae = np.mean(np.abs(sr_true - sr_preds))
        
        # Risk-stratified performance
        risk_terciles = np.percentile(sr_true, [33, 67])

        # Calculate nodes per graph
        if len(sr_true) > 0:
            num_nodes_per_graph = len(default_true) // len(sr_true)
        else:
            num_nodes_per_graph = 0

        # Create graph-level masks
        graph_level_low_mask = sr_true <= risk_terciles[0]
        graph_level_med_mask = (sr_true > risk_terciles[0]) & (sr_true <= risk_terciles[1])
        graph_level_high_mask = sr_true > risk_terciles[1]

        # Expand to node level
        low_risk_mask = np.repeat(graph_level_low_mask, num_nodes_per_graph)
        med_risk_mask = np.repeat(graph_level_med_mask, num_nodes_per_graph)
        high_risk_mask = np.repeat(graph_level_high_mask, num_nodes_per_graph)
        
        risk_stratified_f1 = {
            'low_risk': f1_score(default_true[low_risk_mask], default_preds[low_risk_mask], zero_division=0),
            'medium_risk': f1_score(default_true[med_risk_mask], default_preds[med_risk_mask], zero_division=0),
            'high_risk': f1_score(default_true[high_risk_mask], default_preds[high_risk_mask], zero_division=0)
        }
        
        return {
            'default_prediction': {
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'specificity': float(specificity),
                'confusion_matrix': cm.tolist(),
                'risk_stratified_f1': risk_stratified_f1
            },
            'systemic_risk_prediction': {
                'r2_score': float(sr_r2),
                'rmse': float(sr_rmse),
                'mae': float(sr_mae),
                'correlation': float(np.corrcoef(sr_true, sr_preds)[0, 1]) if len(sr_true) > 1 else 0.0
            },
            'combined_performance': {
                'weighted_score': 0.7 * f1 + 0.3 * max(0, sr_r2)
            }
        }
    
    def _log_comprehensive_results(self, results: Dict):
        """Log comprehensive evaluation results to console and log file."""
        self.logger.info("=" * 80)
        self.logger.info("COMPREHENSIVE EVALUATION RESULTS")
        self.logger.info("=" * 80)
        
        # Default prediction results
        dp = results['default_prediction']
        self.logger.info("DEFAULT PREDICTION PERFORMANCE:")
        self.logger.info(f"  F1 Score: {dp['f1_score']:.4f}")
        self.logger.info(f"  ROC-AUC: {dp['roc_auc']:.4f}")
        self.logger.info(f"  Accuracy: {dp['accuracy']:.4f}")
        self.logger.info(f"  Precision: {dp['precision']:.4f}")
        self.logger.info(f"  Recall: {dp['recall']:.4f}")
        self.logger.info(f"  Specificity: {dp['specificity']:.4f}")
        
        # Risk-stratified performance
        rsf1 = dp['risk_stratified_f1']
        self.logger.info(f"  Risk-Stratified F1 - Low: {rsf1['low_risk']:.4f}, "
                        f"Med: {rsf1['medium_risk']:.4f}, High: {rsf1['high_risk']:.4f}")
        
        # Systemic risk results
        sr = results['systemic_risk_prediction']
        self.logger.info("SYSTEMIC RISK PREDICTION PERFORMANCE:")
        self.logger.info(f"  R² Score: {sr['r2_score']:.4f}")
        self.logger.info(f"  RMSE: {sr['rmse']:.4f}")
        self.logger.info(f"  MAE: {sr['mae']:.4f}")
        self.logger.info(f"  Correlation: {sr['correlation']:.4f}")
        
        # Combined performance
        cp = results['combined_performance']
        self.logger.info(f"COMBINED WEIGHTED SCORE: {cp['weighted_score']:.4f}")
        
        # Timing
        if 'timing' in results:
            timing = results['timing']
            self.logger.info(f"INFERENCE PERFORMANCE:")
            self.logger.info(f"  Predictions/second: {timing['predictions_per_second']:.1f}")
            self.logger.info(f"  Avg inference time: {timing['average_inference_time']*1000:.2f}ms")
        
        self.logger.info("=" * 80)
    
    def _save_comprehensive_results(self, results: Dict, all_results: Dict):
        """Save comprehensive results and detailed predictions."""
        # Save metrics
        results_file = self.output_dir / 'comprehensive_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Calculate nodes per graph for data alignment
        num_nodes = 0
        if len(all_results['systemic_risk_ground_truth']) > 0:
            num_nodes = len(all_results['default_ground_truth']) // len(all_results['systemic_risk_ground_truth'])

        # Expand graph-level data to node-level
        sr_pred_expanded = np.repeat(all_results['systemic_risk_predictions'], num_nodes)
        sr_true_expanded = np.repeat(all_results['systemic_risk_ground_truth'], num_nodes)
        network_ids_expanded = np.repeat(all_results.get('network_ids', []), num_nodes)
        scenario_ids_expanded = np.repeat(all_results.get('scenario_ids', []), num_nodes)

        predictions_df = pd.DataFrame({
            'network_id': network_ids_expanded,
            'scenario_id': scenario_ids_expanded,
            'default_prediction': all_results['default_predictions'],
            'default_probability': all_results['default_probabilities'],
            'default_ground_truth': all_results['default_ground_truth'],
            'systemic_risk_prediction': sr_pred_expanded,
            'systemic_risk_ground_truth': sr_true_expanded
        })
        
        predictions_file = self.output_dir / 'detailed_predictions.csv'
        predictions_df.to_csv(predictions_file, index=False)
        
        # Create visualizations
        self._create_evaluation_plots(predictions_df)
        
        self.logger.info(f"Comprehensive results saved to {self.output_dir}")
    
    def _create_evaluation_plots(self, predictions_df: pd.DataFrame):
        """Create comprehensive evaluation visualization plots."""
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # Default prediction performance plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(predictions_df['default_ground_truth'], predictions_df['default_probability'])
        axes[0,0].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc_score(predictions_df["default_ground_truth"], predictions_df["default_probability"]):.3f}')
        axes[0,0].plot([0, 1], [0, 1], 'k--')
        axes[0,0].set_xlabel('False Positive Rate')
        axes[0,0].set_ylabel('True Positive Rate')
        axes[0,0].set_title('ROC Curve - Default Prediction')
        axes[0,0].legend()
        
        # Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(predictions_df['default_ground_truth'], predictions_df['default_probability'])
        ap_score = average_precision_score(predictions_df['default_ground_truth'], predictions_df['default_probability'])
        axes[0,1].plot(recall, precision, label=f'AP = {ap_score:.3f}')
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].set_title('Precision-Recall Curve')
        axes[0,1].legend()
        
        # Prediction distribution
        axes[1,0].hist(predictions_df[predictions_df['default_ground_truth']==0]['default_probability'], 
                      bins=50, alpha=0.7, label='Non-Default', density=True)
        axes[1,0].hist(predictions_df[predictions_df['default_ground_truth']==1]['default_probability'], 
                      bins=50, alpha=0.7, label='Default', density=True)
        axes[1,0].set_xlabel('Predicted Probability')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Prediction Probability Distribution')
        axes[1,0].legend()
        
        # Confusion matrix
        cm = confusion_matrix(predictions_df['default_ground_truth'], predictions_df['default_prediction'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,1], cmap='Blues')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        axes[1,1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'default_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Systemic risk prediction plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot: Predicted vs Actual
        axes[0,0].scatter(predictions_df['systemic_risk_ground_truth'], 
                         predictions_df['systemic_risk_prediction'], alpha=0.6)
        min_val = min(predictions_df['systemic_risk_ground_truth'].min(), 
                     predictions_df['systemic_risk_prediction'].min())
        max_val = max(predictions_df['systemic_risk_ground_truth'].max(), 
                     predictions_df['systemic_risk_prediction'].max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0,0].set_xlabel('Actual Systemic Risk')
        axes[0,0].set_ylabel('Predicted Systemic Risk')
        axes[0,0].set_title('Systemic Risk: Predicted vs Actual')
        
        # Residuals plot
        residuals = predictions_df['systemic_risk_prediction'] - predictions_df['systemic_risk_ground_truth']
        axes[0,1].scatter(predictions_df['systemic_risk_ground_truth'], residuals, alpha=0.6)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Actual Systemic Risk')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals Plot')
        
        # Distribution comparison
        axes[1,0].hist(predictions_df['systemic_risk_ground_truth'], bins=50, alpha=0.7, 
                      label='Actual', density=True)
        axes[1,0].hist(predictions_df['systemic_risk_prediction'], bins=50, alpha=0.7, 
                      label='Predicted', density=True)
        axes[1,0].set_xlabel('Systemic Risk')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Systemic Risk Distribution')
        axes[1,0].legend()
        
        # Error by risk level
        risk_bins = pd.qcut(predictions_df['systemic_risk_ground_truth'], q=10, duplicates='drop')
        error_by_bin = predictions_df.groupby(risk_bins).apply(
            lambda x: np.sqrt(mean_squared_error(x['systemic_risk_ground_truth'], 
                                               x['systemic_risk_prediction']))
        )
        axes[1,1].bar(range(len(error_by_bin)), error_by_bin.values)
        axes[1,1].set_xlabel('Risk Level Bins')
        axes[1,1].set_ylabel('RMSE')
        axes[1,1].set_title('RMSE by Risk Level')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'systemic_risk_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Evaluation plots saved to {plots_dir}")
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint with comprehensive state information."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'feature_scaler_params': {
                'mean_': getattr(self, '_feature_scaler_mean', None),
                'scale_': getattr(self, '_feature_scaler_scale', None)
            }
        }
        
        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_history(self):
        """Save complete training history and attention analysis."""
        history_file = self.output_dir / 'training_history.json'
        
        # Convert to serializable format
        history_dict = {}
        for key, values in self.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_file, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Save attention history
        attention_file = self.output_dir / 'attention_history.json'
        with open(attention_file, 'w') as f:
            json.dump(self.attention_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_file}")
    
    def create_explainability_report(self, test_loader: DataLoader, sample_networks: int = 5):
        """
        Generate comprehensive explainability report with attention analysis.
        
        Analyzes model attention patterns to provide insights into
        decision-making process and network importance.
        
        Args:
            test_loader: Test data loader
            sample_networks: Number of networks to analyze in detail
        """
        self.logger.info("Generating explainability report...")
        
        self.model.eval()
        explainability_data = {
            'attention_patterns': [],
            'feature_importance': {},
            'network_analyses': []
        }
        
        # Sample networks for detailed analysis
        sampled_batches = []
        for i, batch in enumerate(test_loader):
            if i >= sample_networks:
                break
            sampled_batches.append(batch)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(sampled_batches):
                batch = batch.to(self.device)
                
                # Get predictions with attention
                outputs = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, 
                    batch.batch, return_attention=True
                )
                
                # Analyze this network
                network_analysis = {
                    'batch_idx': batch_idx,
                    'network_id': getattr(batch, 'network_id', [f'unknown_{i}' for i in range(len(batch.to_data_list()))])[0] if hasattr(batch, 'network_id') else f'batch_{batch_idx}',
                    'num_nodes': batch.x.shape[0],
                    'num_edges': batch.edge_index.shape[1],
                    'default_rate_actual': float(batch.y.mean()),
                    'default_rate_predicted': float(torch.sigmoid(outputs['default_logits']).mean()),
                    'systemic_risk_actual': float(batch.systemic_risk.mean()),
                    'systemic_risk_predicted': float(outputs['systemic_risk'].mean())
                }
                
                # Attention analysis
                if hasattr(self.model, 'attention_weights') and self.model.attention_weights:
                    attention_summary = self._analyze_attention_for_network(
                        self.model.attention_weights, batch, outputs
                    )
                    network_analysis['attention_summary'] = attention_summary
                
                explainability_data['network_analyses'].append(network_analysis)
        
        # Save explainability report
        report_file = self.output_dir / 'explainability_report.json'
        with open(report_file, 'w') as f:
            json.dump(explainability_data, f, indent=2)
        
        self.logger.info(f"Explainability report saved to {report_file}")
        
        return explainability_data
    
    def _analyze_attention_for_network(self, attention_weights: List, batch: Data, outputs: Dict) -> Dict:
        """Analyze attention patterns for specific network instance."""
        attention_summary = {
            'high_attention_edges': [],
            'attention_to_shocked_banks': [],
            'layer_attention_stats': []
        }
        
        # Find shocked bank indices (shock indicator is last feature)
        shocked_banks = torch.where(batch.x[:, -1] == 1.0)[0]
        
        for layer_idx, (edge_index, attention) in enumerate(attention_weights):
            attention_scores = attention.cpu().numpy()
            
            # Calculate high attention edges
            high_attention_threshold = np.percentile(attention_scores, 95)
            high_attention_mask = attention_scores > high_attention_threshold
            high_attention_edges = edge_index[:, high_attention_mask].cpu().tolist()
            
            # Analyze attention involving shocked banks
            shocked_attention = []
            for shocked_bank in shocked_banks:
                from_shocked = edge_index[0, :] == shocked_bank
                to_shocked = edge_index[1, :] == shocked_bank
                
                if from_shocked.any():
                    avg_attention_from = attention_scores[from_shocked].mean()
                    shocked_attention.append({
                        'direction': 'from_shocked',
                        'avg_attention': float(avg_attention_from),
                        'max_attention': float(attention_scores[from_shocked].max())
                    })
                
                if to_shocked.any():
                    avg_attention_to = attention_scores[to_shocked].mean()
                    shocked_attention.append({
                        'direction': 'to_shocked',
                        'avg_attention': float(avg_attention_to),
                        'max_attention': float(attention_scores[to_shocked].max())
                    })
            
            layer_stats = {
                'layer': layer_idx,
                'mean_attention': float(attention_scores.mean()),
                'std_attention': float(attention_scores.std()),
                'max_attention': float(attention_scores.max()),
                'high_attention_edges_count': int(high_attention_mask.sum())
            }
            
            attention_summary['high_attention_edges'].append(high_attention_edges)
            attention_summary['attention_to_shocked_banks'].append(shocked_attention)
            attention_summary['layer_attention_stats'].append(layer_stats)
        
        return attention_summary


def validate_enhanced_config(config: Dict, logger: logging.Logger):
    """
    Validate configuration completeness and consistency.
    
    Ensures all required configuration sections are present
    and applies sensible defaults where needed.
    """
    required_sections = ['model_architecture', 'training', 'optimizer', 'scheduler', 'feature_config']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate multi-task training parameters
    training_config = config['training']
    if 'default_loss_weight' not in training_config:
        config['training']['default_loss_weight'] = 1.0
        logger.warning("Setting default_loss_weight to 1.0")
    
    if 'systemic_risk_loss_weight' not in training_config:
        config['training']['systemic_risk_loss_weight'] = 0.5
        logger.warning("Setting systemic_risk_loss_weight to 0.5")
    
    # Validate feature configuration
    if 'node_features' not in config['feature_config']:
        raise ValueError("Missing 'node_features' in feature_config")
    
    logger.info("Configuration validated successfully")


def memory_cleanup():
    """Execute comprehensive memory cleanup for resource management."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    """Main execution function for enhanced GNN training pipeline."""
    parser = argparse.ArgumentParser(
        description="Enhanced GNN Trainer for Financial Contagion Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument('--results-json', type=str, 
                       default="gs://financial-networks-multilayer-unique123/data/results/debtrank_results.json",
                       help="Path to DebtRank results JSON (local or gs://)")
    parser.add_argument('--nodes-path', type=str, default="data/processed/nodes.csv",
                       help="Path to nodes CSV file")
    parser.add_argument('--edges-dir', type=str, default="data/networks/edge_lists",
                       help="Directory containing network edge files")
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default="models/enhanced_gnn",
                       help="Directory for model outputs")
    parser.add_argument('--experiment-name', type=str, default=None,
                       help="Experiment name for organization")
    
    # Model configuration
    parser.add_argument('--config', type=str, default="config/model_config.yaml",
                       help="Model configuration file")
    parser.add_argument('--checkpoint', type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Training overrides
    parser.add_argument('--epochs', type=int, default=None,
                       help="Override config epochs")
    parser.add_argument('--batch-size', type=int, default=None,
                       help="Override config batch size")
    parser.add_argument('--lr', type=float, default=None,
                       help="Override learning rate")
    
    # Enhanced features
    parser.add_argument('--generate-explainability', action='store_true',
                       help="Generate explainability report")
    parser.add_argument('--create-visualizations', action='store_true',
                       help="Create evaluation visualizations")
    
    # System configuration
    parser.add_argument('--log-level', type=str, default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument('--num-workers', type=int, default=2,
                       help="DataLoader worker processes")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup output directory
    if args.experiment_name:
        output_dir = Path(args.output_dir) / args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"enhanced_run_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir, args.log_level)
    
    try:
        logger.info("=" * 80)
        logger.info("ENHANCED GNN TRAINING FOR FINANCIAL CONTAGION PREDICTION")
        logger.info("=" * 80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Random seed: {args.seed}")
        
        # Load and validate configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply command line overrides
        if args.epochs:
            config['training']['epochs'] = args.epochs
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
        if args.lr:
            config['optimizer']['lr'] = args.lr
        
        validate_enhanced_config(config, logger)
        
        # Save configuration
        config_save_path = output_dir / 'config_used.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Initialize dataset
        dataset = EnhancedFinancialNetworkDataset(
            networks_dir=args.edges_dir,
            ground_truth_path=args.results_json,
            nodes_path=args.nodes_path,
            cache_dir=output_dir / "cache",
            feature_config=config['feature_config'],
            logger=logger,
            use_cloud=True
        )
        
        # Create stratified splits
        train_data, val_data, test_data = dataset.create_stratified_dataset_split(
            train_ratio=0.8,
            val_ratio=0.1,
            random_seed=args.seed
        )
        
        if not train_data:
            raise ValueError("No training data available")
        
        # Create data loaders
        batch_size = config['training'].get('batch_size', 16)
        
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
        )
        
        # Initialize trainer
        trainer = EnhancedGNNTrainer(config, output_dir, logger)
        trainer.calculate_class_weights(train_loader)
        
        # Setup model
        input_dim = train_data[0].x.shape[1]
        edge_dim = train_data[0].edge_attr.shape[1]
        trainer.setup_model(input_dim, edge_dim)
        
        # Load checkpoint if provided
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=trainer.device, weights_only=False)
                trainer.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Train model
        trainer.train(train_loader, val_loader)
        
        # Load best model for evaluation
        best_model_path = output_dir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model for evaluation")
        
        # Comprehensive evaluation
        test_results = trainer.evaluate_comprehensive(test_loader)
        
        # Generate explainability report if requested
        if args.generate_explainability:
            trainer.create_explainability_report(test_loader)
        
        # Memory cleanup
        memory_cleanup()
        
        print("\nEnhanced GNN training completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"Combined performance score: {test_results['combined_performance']['weighted_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        memory_cleanup()
        logging.shutdown()


if __name__ == '__main__':
    main()