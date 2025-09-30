"""
Graph Neural Network Training Pipeline for Financial Contagion Analysis

This module implements a training pipeline for Graph Attention Networks
designed to predict systemic risk in financial networks. The pipeline is specifically
configured to support robustness testing through edge masking experiments.

########## THIS SCRIPT SHOULD BE RAN ONLY TO SET UP THE GNN FOR THE MASKING EXPIERMENT #########

Architecture Design:
- Multi-task learning framework for node-level defaults and network-level systemic risk
- Pure intrinsic features to prevent data leakage during masking experiments
- Training dataset: Networks 10000-16999 (7,000 networks)
- Holdout dataset: Networks 17000-17999 (1,000 networks for masking evaluation)

Key Features:
- Stratified sampling ensuring representative risk distribution
- Class-weighted loss functions for imbalanced datasets
- Comprehensive evaluation metrics and visualization
- Cloud-based data pipeline with local caching
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
from io import StringIO

from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, r2_score)

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configure comprehensive logging infrastructure with file and console outputs.
    
    Args:
        output_dir: Directory for log file storage
        log_level: Logging verbosity level
        
    Returns:
        Configured logger instance
    """
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"masking_gnn_training_{timestamp}.log"
    
    logger = logging.getLogger("MaskingGNNTrainer")
    logger.setLevel(getattr(logging, log_level))
    logger.handlers = []
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
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
    Multi-Task Graph Attention Network for Financial Contagion Prediction.
    
    This architecture combines node-level default prediction with graph-level
    systemic risk regression through a shared representation learning framework.
    The model uses multi-head attention mechanisms to capture complex inter-bank
    dependencies across multiple network layers.
    
    Architecture Components:
    - Input projection layer with batch normalization
    - Stacked GAT layers with residual connections
    - Task-specific prediction heads for defaults and systemic risk
    - Edge attribute integration for relationship characterization
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 4, num_heads: int = 8, 
                 dropout: float = 0.3, edge_dim: int = 10):
        """
        Initialize the multi-task GAT architecture.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden representation dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads per layer
            dropout: Dropout probability for regularization
            edge_dim: Dimension of edge features
        """
        super(MultiTaskFinancialGAT, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention_weights = []
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim // num_heads, 
                    heads=num_heads, dropout=dropout, 
                    edge_dim=edge_dim, concat=True)
        )
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads,
                        heads=num_heads, dropout=dropout,
                        edge_dim=edge_dim, concat=True)
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim, heads=1,
                    dropout=dropout, edge_dim=edge_dim, concat=False)
        )
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        self.default_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.systemic_risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier initialization to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task GAT.
        
        Args:
            x: Node feature matrix [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment vector for graph pooling
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and embeddings
        """
        self.attention_weights = [] if return_attention else None
        h = self.input_projection(x)
        
        for i, (gat_layer, layer_norm, dropout) in enumerate(
            zip(self.gat_layers, self.layer_norms, self.dropout_layers)
        ):
            h_residual = h
            if return_attention:
                h, attention = gat_layer(h, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                self.attention_weights.append(attention)
            else:
                h = gat_layer(h, edge_index, edge_attr=edge_attr)
            
            h = F.relu(h)
            if h.size(-1) == h_residual.size(-1):
                h = h + h_residual
            h = layer_norm(h)
            h = dropout(h)
            
        default_logits = self.default_predictor(h)
        
        if batch is not None:
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_sum = global_add_pool(h, batch)
        else:
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


class MaskingFinancialNetworkDataset:
    """
    Dataset manager for financial network data with masking experiment support.
    
    This class handles data loading, preprocessing, and partitioning for the GNN
    training pipeline. It enforces strict separation between training data 
    (networks 10000-16999) and holdout data for masking experiments (17000-17999).
    
    Features:
    - Cloud storage integration with local caching
    - Pure feature extraction to prevent data leakage
    - Stratified sampling based on systemic risk distribution
    - Shock scenario simulation through feature manipulation
    """
    
    def __init__(self, networks_dir: str, ground_truth_path: str, 
                 nodes_path: str, cache_dir: Path,
                 logger: logging.Logger):
        """
        Initialize the dataset manager.
        
        Args:
            networks_dir: Cloud storage path to network edge lists
            ground_truth_path: Path to ground truth results
            nodes_path: Path to node feature data
            cache_dir: Local directory for data caching
            logger: Logger instance for progress tracking
        """
        self.logger = logger
        self.logger.info("Initializing Financial Network Dataset for masking experiments...")
        
        if not GCS_AVAILABLE:
            raise ImportError("Google Cloud Storage SDK is required for this dataset.")
            
        self.storage_client = storage.Client()
        self.bucket_name = "financial-networks-multilayer-unique123"
        self.bucket = self.storage_client.bucket(self.bucket_name)

        all_ground_truth = self._load_gcs_json(ground_truth_path)
        self.logger.info(f"Loaded {len(all_ground_truth)} total ground truth entries.")

        self.ground_truth = [
            item for item in all_ground_truth
            if 10000 <= item.get('network_id', -1) <= 16999
        ]
        self.logger.info(f"Filtered to {len(self.ground_truth)} entries for training (Networks 10000-16999).")
        
        if not self.ground_truth:
            raise ValueError("No ground truth data found in the specified range for training (10000-16999).")

        self.nodes_df = self._load_gcs_csv(nodes_path)
        
        self.networks_dir_prefix = networks_dir.split(self.bucket_name + '/')[-1]
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_scaler = RobustScaler()
        self._prepare_pure_node_features()
        
        self.unique_simulations = self._get_stratified_simulations()
        
        self.logger.info(f"Dataset initialized: {len(self.unique_simulations)} simulations available.")

    def _load_gcs_json(self, gcs_path: str) -> List[Dict]:
        """Load JSON data from cloud storage."""
        self.logger.info(f"Loading JSON from: {gcs_path}")
        try:
            blob_name = gcs_path.split(self.bucket_name + '/')[-1]
            blob = self.bucket.blob(blob_name)
            return json.loads(blob.download_as_text())
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {gcs_path}: {e}")
            raise

    def _load_gcs_csv(self, gcs_path: str) -> pd.DataFrame:
        """Load CSV data from cloud storage."""
        self.logger.info(f"Loading CSV from: {gcs_path}")
        try:
            blob_name = gcs_path.split(self.bucket_name + '/')[-1]
            blob = self.bucket.blob(blob_name)
            csv_content = blob.download_as_text()
            return pd.read_csv(StringIO(csv_content))
        except Exception as e:
            self.logger.error(f"Failed to load CSV from {gcs_path}: {e}")
            raise

    def _prepare_node_features_with_shock(self, sim_truth: Dict) -> np.ndarray:
        """
        Prepare node features with shock scenario applied to specified institution.
        
        Args:
            sim_truth: Ground truth dictionary containing shock information
            
        Returns:
            Feature matrix with shock indicators and adjustments applied
        """
        node_features_copy = self.node_features.copy()
        node_id_map = {node_id: i for i, node_id in enumerate(self.nodes_df['node_id'])}
        shock_bank_idx = node_id_map.get(sim_truth['initial_shock_bank'])

        if shock_bank_idx is not None:
            feature_columns = list(self.node_features_df.columns)
            try:
                if 'capital' in feature_columns: 
                    node_features_copy[shock_bank_idx, feature_columns.index('capital')] = 0.0
                if 'total_assets' in feature_columns: 
                    node_features_copy[shock_bank_idx, feature_columns.index('total_assets')] = 0.0
                if 'capital_ratio' in feature_columns: 
                    node_features_copy[shock_bank_idx, feature_columns.index('capital_ratio')] = 0.0
            except ValueError as e:
                self.logger.warning(f"Feature column not found during shock application: {e}")

        shock_indicator_feature = np.zeros((self.node_features.shape[0], 1))
        if shock_bank_idx is not None:
            shock_indicator_feature[shock_bank_idx] = 1.0

        return np.concatenate([node_features_copy, shock_indicator_feature], axis=1)

    def _get_stratified_simulations(self) -> List[Tuple[int, str, float]]:
        """Extract simulation identifiers with systemic risk labels for stratification."""
        simulations = [(item['network_id'], item['scenario_id'], item.get('systemic_risk', 0.0))
                       for item in self.ground_truth]
        return simulations

    def get_simulation_data(self, network_id: int, scenario_id: str) -> Optional[Data]:
        """
        Construct PyTorch Geometric Data object for a specific simulation.
        
        Args:
            network_id: Network identifier
            scenario_id: Scenario identifier
            
        Returns:
            PyTorch Geometric Data object or None if construction fails
        """
        try:
            sim_truth = next((item for item in self.ground_truth 
                            if item['network_id'] == network_id and item['scenario_id'] == scenario_id), None)
            if not sim_truth: 
                return None
            
            edge_filename = f"edges_sim_{network_id:06d}.csv"
            edges_df = self._load_edge_file(edge_filename)
            if edges_df is None or edges_df.empty: 
                return None

            edge_list, edge_features = self._process_enhanced_edges(edges_df)
            if not edge_list: 
                return None
            
            node_features = self._prepare_node_features_with_shock(sim_truth)
            
            return Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                y=torch.tensor(sim_truth['binary_defaults'], dtype=torch.float).unsqueeze(1),
                systemic_risk=torch.tensor([sim_truth.get('systemic_risk', 0.0)], dtype=torch.float),
                network_id=network_id,
                scenario_id=scenario_id,
                initial_shock_bank=sim_truth['initial_shock_bank']
            )
        except Exception as e:
            self.logger.error(f"Error creating data for network {network_id}: {e}")
            return None

    def _load_edge_file(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load edge list from cache or cloud storage.
        
        Args:
            filename: Name of the edge list file
            
        Returns:
            DataFrame containing edge data or None if unavailable
        """
        local_edge_file = self.cache_dir / filename
        if local_edge_file.exists():
            return pd.read_csv(local_edge_file)
        
        try:
            gcs_blob_path = f"{self.networks_dir_prefix}/{filename}"
            blob = self.bucket.blob(gcs_blob_path)
            if blob.exists():
                csv_content = blob.download_as_text()
                edges_df = pd.read_csv(StringIO(csv_content))
                edges_df.to_csv(local_edge_file, index=False)
                return edges_df
        except Exception as e:
            self.logger.warning(f"Could not load {filename} from cloud storage: {e}")
        
        self.logger.warning(f"Edge file not found: {filename}")
        return None

    def _process_enhanced_edges(self, edges_df: pd.DataFrame) -> Tuple[List, List]:
        """
        Convert edge DataFrame to GNN-compatible format with enhanced features.
        
        Args:
            edges_df: DataFrame containing network edges
            
        Returns:
            Tuple of (edge_list, edge_feature_list)
        """
        layer_mapping = {'derivatives': 0, 'securities': 1, 'fx': 2, 'deposits_loans': 3, 'firesale': 4, 'default': 5}
        edge_list, edge_features = [], []
        node_id_map = {node_id: i for i, node_id in enumerate(self.nodes_df['node_id'])}
        
        weights = edges_df['weight'].values
        weight_percentiles = np.percentile(weights, [25, 50, 75, 90, 95]) if len(weights) > 0 else [0]*5
        
        for _, edge in edges_df.iterrows():
            source_idx, target_idx = node_id_map.get(edge['source']), node_id_map.get(edge['target'])
            if source_idx is None or target_idx is None: 
                continue
            
            edge_list.append([source_idx, target_idx])
            
            one_hot_layer = [0] * len(layer_mapping)
            one_hot_layer[layer_mapping.get(edge['layer'], 5)] = 1
            weight = edge['weight']
            
            edge_feature_vector = [np.log1p(weight), 
                                   np.searchsorted(weight_percentiles, weight) / len(weight_percentiles),
                                   1 if weight > weight_percentiles[3] else 0,
                                   1 if weight > weight_percentiles[4] else 0] + one_hot_layer
            edge_features.append(edge_feature_vector)
        return edge_list, edge_features

    def _prepare_pure_node_features(self):
        """
        Extract pure intrinsic node features to prevent data leakage.
        
        This method ensures that only fundamental financial characteristics
        are used, avoiding any graph-derived or topological features that
        could compromise masking experiments.
        """
        self.logger.info("Preparing pure node features for masking experiment model...")

        pure_feature_cols = [
            'total_assets', 'capital', 'capital_ratio',
            'bank_tier', 'lgd', 'leverage_ratio'
        ]

        available_features = [col for col in pure_feature_cols if col in self.nodes_df.columns]
        self.logger.info(f"Using pure features: {available_features}")

        if not available_features:
            raise ValueError("No pure feature columns found in the nodes file.")

        self.node_features_df = self.nodes_df[available_features].copy()

        for col in self.node_features_df.columns:
            if self.node_features_df[col].isnull().any():
                median_val = self.node_features_df[col].median()
                self.node_features_df[col].fillna(median_val, inplace=True)

        self.node_features = self.feature_scaler.fit_transform(self.node_features_df)
        self.logger.info(f"Prepared {self.node_features.shape[1]} features from pure inputs.")
        
    def create_stratified_dataset_split(self, train_ratio: float = 0.8, 
                                       val_ratio: float = 0.1, 
                                       random_seed: int = 42) -> Tuple[List[Data], List[Data], List[Data]]:
        """
        Create stratified train/validation/test splits ensuring representative risk distribution.
        
        Args:
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, val_data, test_data) lists
        """
        self.logger.info(f"Creating stratified dataset split: {train_ratio:.0%} train, {val_ratio:.0%} val")
        
        systemic_risks = [s[2] for s in self.unique_simulations]
        risk_bins = np.percentile(systemic_risks, [0, 33, 66, 100])
        risk_levels = np.digitize(systemic_risks, bins=risk_bins[1:-1])

        sim_array = np.array(self.unique_simulations, dtype=object)
        
        splitter1 = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio + val_ratio, random_state=random_seed)
        train_val_idx, test_idx = next(splitter1.split(sim_array, risk_levels))
        
        train_val_risks = risk_levels[train_val_idx]
        splitter2 = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio / (train_ratio + val_ratio), random_state=random_seed)
        train_idx_rel, val_idx_rel = next(splitter2.split(train_val_idx, train_val_risks))
        
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
        
        def process_split(indices, split_name):
            data_list = []
            for i in tqdm(indices, desc=f"Processing {split_name}"):
                network_id, scenario_id, _ = self.unique_simulations[i]
                data = self.get_simulation_data(network_id, scenario_id)
                if data: 
                    data_list.append(data)
            return data_list

        train_data = process_split(train_idx, "Train")
        val_data = process_split(val_idx, "Val")
        test_data = process_split(test_idx, "Test")
        
        return train_data, val_data, test_data


class EnhancedGNNTrainer:
    """
    Production-grade training orchestrator for multi-task Graph Neural Networks.
    
    This class manages the complete training lifecycle including:
    - Model initialization and optimization
    - Training loop with gradient clipping and learning rate scheduling
    - Comprehensive evaluation and metrics tracking
    - Checkpoint management and experiment logging
    """
    
    def __init__(self, config: Dict, output_dir: Path, logger: logging.Logger):
        """
        Initialize the training orchestrator.
        
        Args:
            config: Training configuration dictionary
            output_dir: Directory for outputs and checkpoints
            logger: Logger instance for progress tracking
        """
        self.config = config
        self.output_dir = output_dir
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.systemic_risk_criterion = nn.MSELoss()
        self.default_criterion = nn.BCEWithLogitsLoss()
        
        training_config = self.config.get('training', {})
        self.default_loss_weight = training_config.get('default_loss_weight', 1.0)
        self.systemic_risk_loss_weight = training_config.get('systemic_risk_loss_weight', 0.5)
        self.history = defaultdict(list)
        self.logger.info(f"Loss weights configured: Default={self.default_loss_weight}, Systemic Risk={self.systemic_risk_loss_weight}")

    def setup_model(self, input_dim: int, edge_dim: int):
        """
        Initialize model, optimizer, and learning rate scheduler.
        
        Args:
            input_dim: Dimension of input node features
            edge_dim: Dimension of edge features
        """
        self.model = MultiTaskFinancialGAT(
            input_dim=input_dim,
            edge_dim=edge_dim,
            **self.config['model_architecture']
        ).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), **self.config['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, **self.config['scheduler'])
        self.logger.info(f"Model setup complete on {self.device}.")

    def calculate_class_weights(self, train_loader: DataLoader):
        """
        Calculate and apply class weights for imbalanced datasets.
        
        Args:
            train_loader: Training data loader
        """
        num_positives, num_negatives = 0, 0
        for batch in train_loader:
            num_positives += torch.sum(batch.y == 1)
            num_negatives += torch.sum(batch.y == 0)
        
        if num_positives > 0:
            pos_weight = torch.tensor([num_negatives / num_positives], device=self.device)
            self.default_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.logger.info(f"Setting BCE positive class weight to: {pos_weight.item():.2f}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Execute complete training pipeline with validation and early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.logger.info("Starting multi-task training with systemic risk optimization...")
        writer = SummaryWriter(self.output_dir / "tensorboard")
        best_val_metric = -float('inf')
        patience_counter = 0
        training_config = self.config['training']
        epochs = training_config['epochs']
        early_stopping_patience = training_config['early_stopping_patience']

        for epoch in range(epochs):
            train_metrics = self._train_epoch(train_loader, epoch, epochs)
            val_metrics = self._validate(val_loader)
            
            self.scheduler.step(val_metrics['primary_metric'])
            
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val SR-R2: {val_metrics['systemic_risk_r2']:.4f}"
            )
            
            for key, value in train_metrics.items(): 
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items(): 
                self.history[f'val_{key}'].append(value)
            
            writer.add_scalars('Loss', {'train': train_metrics['total_loss'], 'val': val_metrics['total_loss']}, epoch)
            writer.add_scalar('Metrics/Val_F1', val_metrics['f1'], epoch)
            writer.add_scalar('Metrics/Val_SystemicRisk_R2', val_metrics['systemic_risk_r2'], epoch)

            if val_metrics['primary_metric'] > best_val_metric:
                best_val_metric = val_metrics['primary_metric']
                self._save_checkpoint('best_model.pth', epoch, val_metrics)
                patience_counter = 0
                self.logger.info(f"New best systemic risk R2: {best_val_metric:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        writer.close()
        self._save_history()

    def _train_epoch(self, train_loader: DataLoader, epoch: int, total_epochs: int) -> Dict[str, float]:
        """
        Execute single training epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            total_epochs: Total number of epochs
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        training_config = self.config['training']
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            default_loss = self.default_criterion(outputs['default_logits'], batch.y)
            sr_loss = self.systemic_risk_criterion(outputs['systemic_risk'], batch.systemic_risk.unsqueeze(1))
            
            epoch_progress = epoch / total_epochs
            adaptive_sr_weight = self.systemic_risk_loss_weight * (1 + epoch_progress)
            
            batch_loss = (self.default_loss_weight * default_loss) + (adaptive_sr_weight * sr_loss)
            batch_loss.backward()

            if training_config.get('gradient_clipping', False):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=training_config.get('max_grad_norm', 1.0))
            
            self.optimizer.step()
            total_loss += batch_loss.item()
            
        return {'total_loss': total_loss / len(train_loader)}

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Execute validation loop and compute metrics.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        all_default_preds, all_default_labels = [], []
        all_sr_preds, all_sr_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                
                default_loss = self.default_criterion(outputs['default_logits'], batch.y)
                sr_loss = self.systemic_risk_criterion(outputs['systemic_risk'], batch.systemic_risk.unsqueeze(1))
                total_loss += ((self.default_loss_weight * default_loss) + (self.systemic_risk_loss_weight * sr_loss)).item()
                
                all_default_preds.extend(torch.sigmoid(outputs['default_logits']).cpu().numpy())
                all_default_labels.extend(batch.y.cpu().numpy())
                all_sr_preds.extend(outputs['systemic_risk'].cpu().numpy())
                all_sr_labels.extend(batch.systemic_risk.cpu().numpy())
        
        binary_preds = (np.array(all_default_preds) > 0.5).astype(int)
        f1 = f1_score(all_default_labels, binary_preds, zero_division=0)
        sr_r2 = r2_score(all_sr_labels, all_sr_preds)
        
        return {
            'total_loss': total_loss / len(val_loader), 
            'f1': f1, 
            'systemic_risk_r2': sr_r2,
            'primary_metric': sr_r2
        }
        
    def evaluate_comprehensive(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        self.logger.info("Running comprehensive evaluation...")
        self.model.eval()
        all_default_preds, all_default_probs, all_default_labels = [], [], []
        all_sr_preds, all_sr_labels = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs = torch.sigmoid(outputs['default_logits']).cpu().numpy().flatten()
                all_default_probs.extend(probs)
                all_default_preds.extend((probs > 0.5).astype(int))
                all_default_labels.extend(batch.y.cpu().numpy().flatten())
                all_sr_preds.extend(outputs['systemic_risk'].cpu().numpy().flatten())
                all_sr_labels.extend(batch.systemic_risk.cpu().numpy().flatten())

        f1 = f1_score(all_default_labels, all_default_preds, zero_division=0)
        roc_auc = roc_auc_score(all_default_labels, all_default_probs) if len(np.unique(all_default_labels)) > 1 else 0.0
        sr_r2 = r2_score(all_sr_labels, all_sr_preds)
        cm = confusion_matrix(all_default_labels, all_default_preds)
        
        results = {
            'default_prediction': {'f1_score': f1, 'roc_auc': roc_auc, 'confusion_matrix': cm.tolist()},
            'systemic_risk_prediction': {'r2_score': sr_r2, 'rmse': np.sqrt(mean_squared_error(all_sr_labels, all_sr_preds))}
        }
        with open(self.output_dir / 'comprehensive_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """
        Save model checkpoint with metadata.
        
        Args:
            filename: Name of checkpoint file
            epoch: Current epoch number
            metrics: Dictionary of current metrics
        """
        torch.save({
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics, 'config': self.config
        }, self.output_dir / filename)
        
    def _save_history(self):
        """Save training history to JSON file."""
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main execution function for the GNN training pipeline."""
    parser = argparse.ArgumentParser(description="GNN Training Pipeline for Financial Contagion Analysis")
    parser.add_argument('--results-json', type=str, 
                        default="gs://financial-networks-multilayer-unique123/final_test/results/debtrank_results.json",
                        help="Cloud storage path to ground truth results")
    parser.add_argument('--nodes-path', type=str, 
                        default="gs://financial-networks-multilayer-unique123/data/nodes.csv",
                        help="Cloud storage path to node features")
    parser.add_argument('--edges-dir', type=str, 
                        default="gs://financial-networks-multilayer-unique123/final_test/networks/edge_lists",
                        help="Cloud storage path to network edge lists")
    parser.add_argument('--output-dir', type=str, default="models/masking_gnn",
                        help="Local directory for outputs and checkpoints")
    parser.add_argument('--config', type=str, default="config/model_config.yaml",
                        help="Path to model configuration file")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"masking_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, "INFO")

    try:
        logger.info("=" * 80)
        logger.info("GNN TRAINING PIPELINE FOR MASKING EXPERIMENTS")
        logger.info("=" * 80)
        
        with open(args.config, 'r') as f: 
            config = yaml.safe_load(f)
        with open(output_dir / 'config_used.yaml', 'w') as f: 
            yaml.dump(config, f)

        dataset = MaskingFinancialNetworkDataset(
            networks_dir=args.edges_dir, 
            ground_truth_path=args.results_json,
            nodes_path=args.nodes_path, 
            cache_dir=output_dir / "cache",
            logger=logger
        )
        
        train_data, val_data, test_data = dataset.create_stratified_dataset_split()
        if not train_data: 
            raise ValueError("No training data loaded.")

        batch_size = config['training'].get('batch_size', 16)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
        
        trainer = EnhancedGNNTrainer(config, output_dir, logger)
        trainer.calculate_class_weights(train_loader)
        
        input_dim = dataset.node_features.shape[1] + 1
        edge_dim = train_data[0].edge_attr.shape[1]
        trainer.setup_model(input_dim, edge_dim)
        
        trainer.train(train_loader, val_loader)
        
        logger.info("Loading best model for final evaluation...")
        best_model_path = output_dir / 'best_model.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=trainer.device, weights_only=False)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_results = trainer.evaluate_comprehensive(test_loader)
        logger.info(f"Final Test Results - F1: {test_results['default_prediction']['f1_score']:.4f}, Systemic Risk R2: {test_results['systemic_risk_prediction']['r2_score']:.4f}")
        
        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Model saved to: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        return 1
    finally:
        if 'trainer' in locals() and 'val_loader' in locals():
            gc.collect()
            if torch.cuda.is_available(): 
                torch.cuda.empty_cache()
        logging.shutdown()
    
    return 0


if __name__ == '__main__':
    exit(main())