"""
Node Data Processor for Financial Contagion Analysis

This module processes raw banking financial statement data and converts it into
standardized node features for multi-layer financial network generation.

Handles balance sheet data for North American banking institutions and extracts
features for multi-layer financial network construction including core financial
metrics, layer-specific exposures, and institutional size classifications.

Project: Multi-Layer Financial Contagion Modeling with Graph Neural Networks
"""

import numpy as np
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Configure module-level logging
logger = logging.getLogger(__name__)


class NodeDataProcessor:
    """
    Processor for banking financial data to create standardized network nodes.
    
    This processor transforms raw financial statement data into standardized
    node features suitable for multi-layer financial network analysis. It handles:
    - Core financial metrics (assets, capital, ratios)
    - Layer-specific exposures (derivatives, securities, FX, deposits/loans)
    - Institutional size classification based on asset percentiles
    - Risk measures and leverage ratios
    
    The processor ensures proper calculation of exposures to avoid double-counting
    across network layers and maintains data integrity through validation.
    
    Args:
        data_path: File path for input banking data (CSV or Excel)
        config_path: Optional path to configuration file for parameters
    """

    def __init__(self, data_path: str, config_path: Optional[str] = None):
        """
        Initialize processor and execute data loading and processing pipeline.
        
        Args:
            data_path: Path to raw banking data file
            config_path: Optional path to YAML configuration file
        """
        self.data_path = Path(data_path)
        self.config = self._load_config(config_path)
        self.df = None
        self.n = 0

        # Core financial attributes
        self.total_assets = np.array([])
        self.capital = np.array([])
        self.capital_ratios = np.array([])
        self.interbank_assets = np.array([])
        self.interbank_liabilities = np.array([])
        self.lgd = np.array([])
        self.leverage = np.array([])
        self.cds_spreads = np.array([])
        self.bank_tiers = np.array([])

        # Layer-specific exposure data
        self.layer_data = {}

        # Execute processing pipeline
        self._load_and_clean_data()
        if self.df is not None:
            self._extract_node_variables()
            self._classify_banks()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration parameters from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary (empty if file not provided/found)
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _load_and_clean_data(self):
        """
        Load banking data and standardize column names.
        
        Supports both CSV and Excel formats. Standardizes column names to
        lowercase with underscores for consistent access. Handles legacy
        Excel formats (.xls) and modern formats (.xlsx).
        """
        logger.info(f"Loading banking data from '{self.data_path}'...")
        
        if not self.data_path.exists():
            logger.error(f"Error: The file was not found at '{self.data_path}'.")
            logger.error("Please verify the data file location.")
            return

        try:
            ext = self.data_path.suffix.lower()
            if ext == '.csv':
                df = pd.read_csv(self.data_path)
            else:
                # Handle Excel formats
                try:
                    df = pd.read_excel(self.data_path, engine='openpyxl')
                except ValueError:
                    # Fallback for legacy .xls files
                    df = pd.read_excel(self.data_path, engine='xlrd')

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return

        # Standardize column names: lowercase with underscores
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[^0-9a-z]+", "_", regex=True)
        )

        self.df = df
        self.n = len(df)
        logger.info(f"Successfully loaded and cleaned data for {self.n} institutions.")
        logger.info(f"Column sample: {list(df.columns)[:10]}")

    def _to_numeric(self, col: str, default: float = 0.0) -> np.ndarray:
        """
        Safely convert DataFrame column to numeric numpy array.
        
        Handles string formatting (commas, spaces), missing columns, and
        non-numeric entries with appropriate defaults.
        
        Args:
            col: Column name to convert
            default: Value to use for missing data or failed conversions
            
        Returns:
            Numpy array of numeric values
        """
        if col in self.df.columns:
            series = self.df[col]
            # Clean string formatting if present
            if series.dtype == 'object':
                series = series.astype(str).str.replace(',', '').str.replace(' ', '')
            # Convert to numeric, handling errors gracefully
            return pd.to_numeric(series, errors='coerce').fillna(default).values
        else:
            logger.warning(f"Column '{col}' not found. Using default value {default} for all entries.")
            return np.full(self.n, default)

    def _extract_node_variables(self):
        """
        Extract and calculate financial characteristics for each institution.
        
        Implements the core feature engineering logic:
        
        1. Core Balance Sheet Items:
           - Total assets, capital, interbank positions
           - Capital ratios calculated with division safety
        
        2. Layer-Specific Exposures (avoiding double-counting):
           - FX: Notional value × exposure rate
           - Securities: Trading and fair value assets
           - Derivatives: Net exposure (total minus FX to avoid overlap)
           - Deposits/Loans: Residual interbank assets after instruments
        
        3. Risk Measures:
           - Loss Given Default (LGD) from loan loss reserves
           - Basel III leverage ratios
           - CDS spreads as market-implied risk
        
        The exposure calculation ensures each dollar of interbank assets
        is counted exactly once across network layers.
        """
        if self.df is None:
            return

        logger.info("Extracting and calculating node variables...")

        # Core balance sheet items
        self.total_assets = self._to_numeric('total_assets')
        self.capital = self._to_numeric('total_equity')
        self.interbank_assets = self._to_numeric('interbank_assets') 
        self.interbank_liabilities = self._to_numeric('interbank_liabilities')
        
        # Capital ratio calculation with division safety
        self.capital_ratios = np.divide(
            self.capital, self.total_assets,
            out=np.zeros_like(self.capital, dtype=float),
            where=self.total_assets > 0
        )

        # Layer-specific exposure calculation (avoiding double-counting)
        
        # 1. FX exposure: Notional value scaled by exposure rate
        fx_notional = self._to_numeric('foreign_exchange_derivatives_notional_value_')
        fx_rate = float(self.config.get('fx_notional_exposure_rate', 0.01))
        fx_exp = fx_notional * fx_rate
        
        # 2. Securities exposure: Trading and fair value positions
        sec_exp = self._to_numeric('financial_assets_trading_and_at_fair_value_through_p_l')
        
        # 3. Net derivatives exposure: Total minus FX to avoid double-counting
        total_deriv_exp = self._to_numeric('derivative_financial_instruments_assets_')
        net_deriv_exp = np.maximum(0, total_deriv_exp - fx_exp)
        
        # 4. Deposits/loans: Residual interbank assets after specific instruments
        #    Represents general interbank lending not captured by other layers
        instrument_based_assets = net_deriv_exp + sec_exp + fx_exp
        dep_loans_exp = np.maximum(0, self.interbank_assets - instrument_based_assets)

        # Store layer exposures for export
        self.layer_data = {
            'derivatives_exposure': net_deriv_exp,
            'securities_exposure': sec_exp,
            'fx_exposure': fx_exp,
            'deposits_loans_exposure': dep_loans_exp
        }

        # Risk measures
        
        # Loss Given Default (LGD) from loan loss reserves
        reserves = self._to_numeric('loan_loss_reserves_average_risk_weighted_assets_rwas_')
        rwas = self._to_numeric('total_risk_weighted_assets_rwas_fully_loaded_highest_')
        self.lgd = np.divide(
            reserves, rwas, 
            out=np.full_like(reserves, 0.4, dtype=float), 
            where=rwas > 0
        )
        
        # Regulatory ratios and market indicators
        self.leverage = self._to_numeric('basel_iii_leverage_ratio_as_reported_')
        self.cds_spreads = self._to_numeric('credit_default_swaps_most_recent_rating_5_years')

        logger.info("Variable extraction complete.")

    def _classify_banks(self):
        """
        Classify institutions into size tiers based on total assets.
        
        - Tier 1 (Medium): 75th to 95th percentile
        - Tier 2 (Large): Above 95th percentile (systemically important)
        
        This classification aligns with regulatory frameworks for
        systemically important financial institutions.
        """
        if self.df is None:
            return

        logger.info("Classifying institutions into size tiers...")

        # Calculate percentile thresholds
        asset_95th = np.percentile(self.total_assets, 95)
        asset_75th = np.percentile(self.total_assets, 75)

        # Initialize all as Tier 0 (Small)
        self.bank_tiers = np.zeros(self.n, dtype=int)

        # Tier 2: Large institutions (top 5%)
        large_mask = (self.total_assets >= asset_95th)
        self.bank_tiers[large_mask] = 2

        # Tier 1: Medium institutions (75th-95th percentile)
        medium_mask = (self.total_assets >= asset_75th) & (self.total_assets < asset_95th)
        self.bank_tiers[medium_mask] = 1

        # Log tier distribution
        tier_counts = np.bincount(self.bank_tiers)
        tier_names = {0: 'Small', 1: 'Medium', 2: 'Large'}
        logger.info(f"Classification complete: "
                    f"{tier_counts[0]} {tier_names[0]}, "
                    f"{tier_counts[1]} {tier_names[1]}, "
                    f"{tier_counts[2]} {tier_names[2]}")

    def export_nodes_to_csv(self, output_dir: str = "data/processed", 
                           filename: str = "nodes.csv") -> Path:
        """
        Export processed node data to CSV file.
        
        Creates a comprehensive node attribute file containing:
        - Unique node identifiers
        - Core financial metrics
        - Layer-specific exposures
        - Risk measures and classifications
        
        Args:
            output_dir: Directory for output file
            filename: Name of output CSV file
            
        Returns:
            Path object pointing to exported file, or None if export fails
        """
        if self.df is None:
            logger.error("Cannot export nodes: Data has not been loaded.")
            return None

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        full_path = output_path / filename
        
        logger.info(f"Preparing to export node data to '{full_path}'...")

        # Compile all node attributes
        nodes_data_dict = {
            'node_id': range(self.n),
            'total_assets': self.total_assets,
            'capital': self.capital,
            'capital_ratio': self.capital_ratios,
            'bank_tier': self.bank_tiers,
            'lgd': self.lgd,
            'leverage_ratio': self.leverage,
            'total_interbank_assets': self.interbank_assets,
            'total_interbank_liabilities': self.interbank_liabilities,
        }

        # Add layer-specific exposures
        nodes_data_dict.update(self.layer_data)

        # Create DataFrame and export
        nodes_df = pd.DataFrame(nodes_data_dict)
        nodes_df.to_csv(full_path, index=False)

        logger.info(f"Successfully exported {len(nodes_df)} nodes to '{full_path}'.")
        
        # Log summary statistics for validation
        logger.info(f"Asset range: ${self.total_assets.min():.0f} - ${self.total_assets.max():.0f}")
        logger.info(f"Capital ratio range: {self.capital_ratios.min():.3f} - {self.capital_ratios.max():.3f}")
        
        return full_path

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Generate summary statistics for processed nodes.
        
        Provides comprehensive statistics for validation and reporting:
        - Total node count
        - Tier distribution
        - Asset statistics (mean, median, std)
        - Capital ratio statistics
        
        Returns:
            Dictionary containing summary statistics
        """
        if self.df is None:
            return {}
            
        return {
            'total_nodes': self.n,
            'tier_distribution': dict(zip(*np.unique(self.bank_tiers, return_counts=True))),
            'asset_statistics': {
                'mean': float(np.mean(self.total_assets)),
                'median': float(np.median(self.total_assets)),
                'std': float(np.std(self.total_assets))
            },
            'capital_ratio_statistics': {
                'mean': float(np.mean(self.capital_ratios)),
                'median': float(np.median(self.capital_ratios)),
                'std': float(np.std(self.capital_ratios))
            }
        }


def main():
    """
    Main execution function for standalone processing.
    
    Configures logging, processes banking data, and exports results.
    Designed for command-line execution and integration into pipelines.
    """
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/node_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    # Define paths relative to project root
    data_dir = Path("data/raw")
    input_file = data_dir / "new_NA_model_input_cleaned.csv"
    
    print("=" * 60)
    print("      Financial Network Node Data Processor")
    print("=" * 60)

    # Initialize and execute processor
    try:
        node_processor = NodeDataProcessor(data_path=input_file)
        
        if node_processor.df is not None:
            # Export processed data
            output_path = node_processor.export_nodes_to_csv()
            
            # Display summary statistics
            stats = node_processor.get_summary_stats()
            print(f"\nProcessing completed successfully!")
            print(f"Processed {stats['total_nodes']} financial institutions")
            print(f"Output saved to: {output_path}")
            
            # Display tier distribution
            print(f"\nInstitution size distribution:")
            tier_names = {0: 'Small', 1: 'Medium', 2: 'Large'}
            for tier, count in stats['tier_distribution'].items():
                print(f"  {tier_names[tier]}: {count} institutions")
            
        else:
            print("\nProcessing failed. Please check the logs for detailed error information.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


if __name__ == '__main__':
    main()