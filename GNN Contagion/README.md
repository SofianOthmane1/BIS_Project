# Multi-Layer Financial Contagion Modelling with Graph Neural Networks ##

## Introduction
This project implements a complete pipeline for analysisng systemic risk inside interbank exposure networks using Graph Neural Networks (GNNs). We shock random banks within empirically backed networks to test the network determinants of contagion - then train a GNN model to predict complete contagion given an initial failed bank.

## What This Project Does
- Processes real banking data from 248 North American banks
- Generates synthetic multi-layer networks representing different types of interbank exposures (derivatives, securities, FX, deposits/loans)
- Simulates financial contagion using the DebtRank algorithm with various shock scenarios
- Trains Graph Neural Networks to predict systemic risk faster than traditional simulation methods
- Tests the GNNs performance under edge masking, in which we remove network data sequentially to test model robustness

## System Requirements
- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Google Cloud SDK (for distributed generation)

# Quick Start (local network generation) 

### Install dependencies
pip install -r requirements.txt

### Process banking data
python -m src.node_processor

### Generate networks locally
python -m src.network_generator

### Run DebtRank simulations
python -m src.debtrank_simulator

### Train GNN model
python -m src.gnn_training

### Masking experiment
python -m src.masking_experiment

# Configuration
## Edit YAML files in config/ directory:

network_params.yaml - Network generation parameters (ultimately determine the base density for each contagion layer)
model_config.yaml - GNN architecture and training settings
cloud_config.yaml - Cloud infrastructure configuration

# Research Applications
- Systemic risk assessment for financial regulators
- Stress testing under various shock scenarios
- Network topology analysis
- Early warning system development

# Contact 
For questions, please contact me via email: sofianothmane0@gmail.com