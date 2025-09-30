"""
Cloud-Based Distributed Network Generator

Orchestrates the generation of thousands of financial networks using
Google Cloud infrastructure for scalable distributed processing.

This module manages VM instance creation, job distribution, monitoring,
and result aggregation for large-scale network generation tasks.
"""

import numpy as np
import pandas as pd
import logging
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

from google.cloud import compute_v1
from google.cloud import storage
from google.cloud import tasks_v2
from google.oauth2 import service_account
import googleapiclient.discovery
from google.api_core import exceptions

from .network_generator import HalajKokStochasticGenerator, get_distributed_targets
from .node_processor import NodeDataProcessor

logger = logging.getLogger(__name__)


@dataclass
class CloudConfig:
    """Configuration for Google Cloud infrastructure."""
    project_id: str
    project_number: str
    api_key: str
    region: str
    zone: str
    instance_type: str
    storage_bucket: str
    service_account_email: str
    max_instances: int
    networks_per_batch: int
    total_networks: int


@dataclass
class BatchJob:
    """Represents a batch job for network generation."""
    batch_id: int
    start_network_id: int
    end_network_id: int
    networks_count: int
    status: str = "pending"
    instance_name: Optional[str] = None
    created_at: Optional[float] = None
    completed_at: Optional[float] = None


class CloudNetworkGenerator:
    """
    Manages distributed generation of financial networks using Google Cloud.
    
    This class orchestrates the entire distributed generation pipeline:
    - Configuration loading and validation
    - VM instance creation and management
    - Job distribution and monitoring
    - Result aggregation and reporting
    """

    def __init__(self, config_path: str = "config/cloud_config.yaml"):
        """Initialize the cloud network generator."""
        self.config = self._load_config(config_path)
        self.gcp_config = self._parse_gcp_config()
        self._init_gcp_clients()
        self.batch_jobs: List[BatchJob] = []
        self.active_instances: Dict[str, Dict] = {}
        self.completed_networks = 0
        self.nodes_df: Optional[pd.DataFrame] = None
        logger.info(f"CloudNetworkGenerator initialized for project: {self.gcp_config.project_id}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _parse_gcp_config(self) -> CloudConfig:
        """Parse GCP configuration from loaded config."""
        gcp_config = self.config['gcp']
        network_config = self.config.get('network_generation', {})
        return CloudConfig(
            project_id=gcp_config['project_id'],
            project_number=gcp_config['project_number'],
            api_key=gcp_config['api_key'],
            region=gcp_config['region'],
            zone=gcp_config['zone'],
            instance_type=gcp_config['instance_type'],
            storage_bucket=gcp_config['storage_bucket'],
            service_account_email=gcp_config['service_account_email'],
            max_instances=network_config.get('max_instances', 20),
            networks_per_batch=network_config.get('networks_per_batch', 800),
            total_networks=network_config.get('total_networks', 10000)
        )

    def _init_gcp_clients(self):
        """Initialize Google Cloud service clients."""
        try:
            self.compute_client = compute_v1.InstancesClient()
            self.storage_client = storage.Client(project=self.gcp_config.project_id)
            self.bucket = self.storage_client.bucket(self.gcp_config.storage_bucket)
            self.tasks_client = tasks_v2.CloudTasksClient()
            logger.info("Google Cloud clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            raise

    def load_node_data(self, nodes_path: str = "data/processed/nodes.csv") -> bool:
        """Load and validate node data for network generation."""
        try:
            nodes_file = Path(nodes_path)
            if not nodes_file.exists():
                logger.error(f"Nodes file not found: {nodes_path}")
                return False
            
            self.nodes_df = pd.read_csv(nodes_file)
            logger.info(f"Loaded {len(self.nodes_df)} nodes from {nodes_path}")

            required_cols = [
                'node_id', 'total_assets', 'capital', 'bank_tier',
                'derivatives_exposure', 'securities_exposure', 
                'fx_exposure', 'deposits_loans_exposure',
                'total_interbank_assets', 'total_interbank_liabilities'
            ]
            
            missing_cols = [col for col in required_cols if col not in self.nodes_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in {nodes_path}: {missing_cols}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error loading node data: {e}")
            return False

    def create_batch_jobs(self) -> List[BatchJob]:
        """Create batch jobs for distributed network generation."""
        total_networks = self.gcp_config.total_networks
        networks_per_batch = self.gcp_config.networks_per_batch
        num_batches = (total_networks + networks_per_batch - 1) // networks_per_batch
        self.batch_jobs = []
        current_network_id = 0
        
        for batch_id in range(num_batches):
            start_id = current_network_id
            end_id = min(start_id + networks_per_batch - 1, total_networks - 1)
            networks_count = end_id - start_id + 1
            job = BatchJob(
                batch_id=batch_id,
                start_network_id=start_id,
                end_network_id=end_id,
                networks_count=networks_count,
                created_at=time.time()
            )
            self.batch_jobs.append(job)
            current_network_id = end_id + 1
            
        logger.info(f"Created {len(self.batch_jobs)} batch jobs for {total_networks} networks")
        return self.batch_jobs

    def create_vm_instance(self, instance_name: str, batch_job: BatchJob) -> bool:
        """Create a Google Cloud VM instance for network generation."""
        try:
            startup_script = self._generate_startup_script(batch_job)
            machine_type = f"zones/{self.gcp_config.zone}/machineTypes/{self.gcp_config.instance_type}"
            
            instance_config = {
                "name": instance_name,
                "machine_type": machine_type,
                "disks": [{
                    "boot": True,
                    "auto_delete": True,
                    "initialize_params": {
                        "source_image": "projects/debian-cloud/global/images/family/debian-11",
                        "disk_size_gb": "10",
                        "disk_type": f"projects/{self.gcp_config.project_id}/zones/{self.gcp_config.zone}/diskTypes/pd-ssd"
                    }
                }],
                "network_interfaces": [{
                    "network": "global/networks/default"
                }],
                "metadata": {"items": [
                    {"key": "startup-script", "value": startup_script},
                    {"key": "batch-id", "value": str(batch_job.batch_id)},
                    {"key": "networks-count", "value": str(batch_job.networks_count)}
                ]},
                "service_accounts": [{
                    "email": self.gcp_config.service_account_email,
                    "scopes": [
                        "https://www.googleapis.com/auth/cloud-platform",
                        "https://www.googleapis.com/auth/devstorage.read_write"
                    ]
                }],
                "scheduling": {"preemptible": True}
            }
            
            operation = self.compute_client.insert(
                project=self.gcp_config.project_id, 
                zone=self.gcp_config.zone, 
                instance_resource=instance_config
            )
            
            logger.info(f"Creating VM instance: {instance_name} for batch {batch_job.batch_id}")
            self.active_instances[instance_name] = {
                "batch_job": batch_job, 
                "operation": operation, 
                "created_at": time.time()
            }
            batch_job.instance_name = instance_name
            batch_job.status = "creating"
            return True
            
        except Exception as e:
            logger.error(f"Failed to create VM instance {instance_name}: {e}")
            batch_job.status = "failed"
            return False

    def _generate_startup_script(self, batch_job: BatchJob) -> str:
        """Generate startup script for VM instance."""
        timeout_seconds = 2700
        timeout_minutes = int(timeout_seconds / 60)

        script = f"""#!/bin/bash
set -e

exec > >(tee -a /var/log/startup-script.log) 2>&1
echo "VM Startup for Batch {batch_job.batch_id} at $(date)"

# Setup environment
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-pip git
pip3 install numpy pandas scipy tqdm pyyaml google-cloud-storage

mkdir -p /opt/network_generation
cd /opt/network_generation

echo "Downloading dependencies..."
gsutil cp gs://{self.gcp_config.storage_bucket}/data/nodes.csv ./nodes.csv
gsutil cp gs://{self.gcp_config.storage_bucket}/scripts/network_generator.py ./network_generator.py
gsutil cp gs://{self.gcp_config.storage_bucket}/config/network_params.yaml ./network_params.yaml

# Generate Python worker script
cat > generate_batch.py << 'EOF'
import sys, os, traceback, yaml
from pathlib import Path
import pandas as pd
from google.cloud import storage
from network_generator import run_single_simulation_and_save

def create_and_upload_summary(edge_file_path, sim_id, bucket, total_node_count):
    """Generate and upload network summary statistics."""
    try:
        print(f"Creating summary for network {{sim_id}}...")
        df = pd.read_csv(edge_file_path)

        total_edges = len(df)
        if total_edges == 0:
            print(f"Skipping summary for network {{sim_id}} - no edges.")
            return

        # Calculate union graph statistics
        pairs = df[['source','target']].drop_duplicates()
        union_edges = len(pairs)
        unique_nodes = pd.concat([pairs['source'], pairs['target']]).nunique()
        max_possible_edges = unique_nodes * (unique_nodes - 1)
        union_density = union_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        out_deg = pairs['source'].value_counts()
        in_deg = pairs['target'].value_counts()
        avg_out_degree = union_edges / unique_nodes if unique_nodes > 0 else 0.0
        avg_in_degree = union_edges / unique_nodes if unique_nodes > 0 else 0.0
        max_out_degree = int(out_deg.max()) if len(out_deg) else 0
        max_in_degree = int(in_deg.max()) if len(in_deg) else 0

        # Calculate reciprocity
        edge_set = set(zip(pairs['source'], pairs['target']))
        mutual = sum(1 for (i,j) in edge_set if (j,i) in edge_set and i < j)
        dyads = sum(1 for (i,j) in edge_set if i < j and ((i,j) in edge_set or (j,i) in edge_set))
        reciprocity = float(mutual / dyads) if dyads > 0 else 0.0

        # Weight statistics
        total_weight = df['weight'].sum()
        avg_weight = float(df['weight'].mean())
        std_weight = float(df['weight'].std())
        median_weight = float(df['weight'].median())

        # Layer-specific statistics
        edges_per_layer = df['layer'].value_counts().to_dict()
        layer_densities = {{}}
        layer_avg_weights = {{}}
        layer_total_weights = {{}}
        
        for layer in sorted(df['layer'].unique()):
            if layer == 'firesale':
                continue
            layer_df = df[df['layer'] == layer]
            layer_edge_count = len(layer_df)
            layer_density = layer_edge_count / max_possible_edges if max_possible_edges > 0 else 0.0
            layer_densities[f'density_{{layer}}'] = float(layer_density)
            layer_avg_weights[f'avg_weight_{{layer}}'] = float(layer_df['weight'].mean())
            layer_total_weights[f'total_weight_{{layer}}'] = float(layer_df['weight'].sum())

        # Calculate HHI concentration
        pair_weight = df.groupby(['source','target'])['weight'].sum().reset_index()
        hhi = 0.0
        if total_weight > 0:
            weight_shares = pair_weight['weight'] / total_weight
            hhi = float((weight_shares ** 2).sum())

        # Validation flags
        flags = []
        N = unique_nodes
        if abs(avg_out_degree - union_edges / N) > 1e-9:
            flags.append("avg_out_degree_identity_fail")
        if abs(avg_in_degree - union_edges / N) > 1e-9:
            flags.append("avg_in_degree_identity_fail")
        if max_out_degree > N - 1:
            flags.append("max_out_degree_exceeds_N_minus_1")

        # Assemble summary
        summary_data = {{
            'simulation_id': sim_id,
            'unique_nodes': unique_nodes,
            'total_edges_multilayer': total_edges,
            'union_edges': union_edges,
            'union_density': float(union_density),
            'total_exposure_weight': float(total_weight),
            'avg_edge_weight': float(avg_weight),
            'std_edge_weight': float(std_weight),
            'median_edge_weight': float(median_weight),
            'weight_concentration_hhi': float(hhi),
            'avg_out_degree_union': float(avg_out_degree),
            'avg_in_degree_union': float(avg_in_degree),
            'max_out_degree_union': int(max_out_degree),
            'max_in_degree_union': int(max_in_degree),
            'reciprocity_union': float(reciprocity),
            'flags': ";".join(flags)
        }}

        # Add layer-specific metrics
        for layer, count in edges_per_layer.items():
            if layer != 'firesale':
                summary_data[f'edges_{{layer}}'] = int(count)

        summary_data.update(layer_densities)
        summary_data.update(layer_avg_weights)
        summary_data.update(layer_total_weights)
        
        summary_df = pd.DataFrame([summary_data])
        
        # Save and upload
        summary_filename = f"summary_sim_{{sim_id:06d}}.csv"
        local_summary_path = edge_file_path.parent / summary_filename
        summary_df.to_csv(local_summary_path, index=False)

        summary_blob = bucket.blob(f'networks/summaries/{{summary_filename}}')
        summary_blob.upload_from_filename(str(local_summary_path), timeout=120)
        print(f"Uploaded summary for network {{sim_id}}.")

    except Exception as e:
        print(f"Could not create summary for network {{sim_id}}: {{e}}")


def main():
    batch_id = {batch_job.batch_id}
    start_id = {batch_job.start_network_id}
    end_id = {batch_job.end_network_id}
    storage_client = None
    
    try:
        print(f"Python worker starting for batch {{batch_id}}.")
        with open('network_params.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        density_params = config['network_generation']['density_params']
        ras_params = config.get('ras_algorithm', {{}})
        nodes_df = pd.read_csv('nodes.csv')
        output_dir = Path(f'batch_{{batch_id:04d}}')
        output_dir.mkdir(exist_ok=True)
        
        storage_client = storage.Client()
        bucket = storage_client.bucket('{self.gcp_config.storage_bucket}')
        
        successful_networks = 0
        for net_id in range(start_id, end_id + 1):
            result_path = run_single_simulation_and_save(
                simulation_id=net_id,
                nodes_df=nodes_df,
                density_params=density_params,
                ras_params=ras_params,
                output_dir=output_dir
            )
            
            if result_path:
                successful_networks += 1
                blob = bucket.blob(f'networks/edge_lists/{{result_path.name}}')
                blob.upload_from_filename(str(result_path), timeout=300)
                print(f"Uploaded network {{net_id}}.")
                
                create_and_upload_summary(
                    edge_file_path=result_path,
                    sim_id=net_id,
                    bucket=bucket,
                    total_node_count=len(nodes_df)
                )
        
        print(f"Generation completed. {{successful_networks}} networks created.")

        if successful_networks > 0:
            completion_blob = bucket.blob(f'status/batch_{{batch_id:04d}}_complete.txt')
            completion_blob.upload_from_string(f"successful:{{successful_networks}}", timeout=60)
            print(f"Uploaded completion signal for batch {{batch_id}}.")
        else:
            print(f"ERROR: Batch {{batch_id}} produced 0 networks.")
            error_blob = bucket.blob(f'status/batch_{{batch_id:04d}}_error.txt')
            error_blob.upload_from_string("Batch completed but generated 0 networks.", timeout=60)

    except Exception as e:
        print(f"FATAL ERROR in batch {{batch_id}}: {{e}}")
        traceback.print_exc()
        if storage_client:
            try:
                bucket = storage_client.bucket('{self.gcp_config.storage_bucket}')
                error_blob = bucket.blob(f'status/batch_{{batch_id:04d}}_error.txt')
                error_blob.upload_from_string(traceback.format_exc(), timeout=60)
            except Exception as upload_err:
                print(f"Failed to upload error signal: {{upload_err}}")
    finally:
        print("Python script execution finished.")

if __name__ == '__main__':
    main()
EOF

# Execute with monitoring
echo "Starting Python script..."
python3 generate_batch.py &
PYTHON_PID=$!
echo "Python script running with PID $PYTHON_PID"

TIMEOUT={timeout_seconds}
ELAPSED=0
while ps -p $PYTHON_PID > /dev/null; do
    sleep 30
    ELAPSED=$((ELAPSED + 30))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "TIMEOUT: Script exceeded {timeout_minutes} minutes. Terminating."
        kill -9 $PYTHON_PID
        echo "Timeout after {timeout_minutes} minutes" > timeout_signal.txt
        gsutil cp timeout_signal.txt gs://{self.gcp_config.storage_bucket}/status/batch_{batch_job.batch_id:04d}_timeout.txt || true
        break 
    fi
done

echo "Process completed. Shutting down."
sleep 15
sudo shutdown -h now
"""
        return script

    def upload_dependencies(self):
        """Upload necessary files to Cloud Storage."""
        try:
            self.bucket.blob("data/nodes.csv").upload_from_filename("data/processed/nodes.csv")
            self.bucket.blob("scripts/network_generator.py").upload_from_filename("src/network_generator.py")
            self.bucket.blob("config/network_params.yaml").upload_from_filename("config/network_params.yaml")
            logger.info("Dependencies uploaded to Cloud Storage")
        except Exception as e:
            logger.error(f"Failed to upload dependencies: {e}")
            raise

    def monitor_instances(self) -> Dict[str, Any]:
        """Monitor running instances and update job status."""
        completed_jobs_count = sum(1 for job in self.batch_jobs if job.status == "completed")
        failed_jobs_count = sum(1 for job in self.batch_jobs if job.status in ["failed", "error", "timeout"])

        stats = {
            "active_instances": len(self.active_instances),
            "completed_jobs": completed_jobs_count,
            "failed_jobs": failed_jobs_count,
        }

        for instance_name, instance_info in list(self.active_instances.items()):
            batch_job = instance_info["batch_job"]

            def process_terminated_job(job, name):
                """Process a terminated job and update status."""
                completion_blob = self.bucket.blob(f"status/batch_{job.batch_id:04d}_complete.txt")
                error_blob = self.bucket.blob(f"status/batch_{job.batch_id:04d}_error.txt")
                
                if completion_blob.exists():
                    job.status = "completed"
                    job.completed_at = time.time()
                    logger.info(f"Batch {job.batch_id} completed successfully.")
                elif error_blob.exists():
                    job.status = "failed"
                    try:
                        error_content = error_blob.download_as_text()
                        logger.error(f"Batch {job.batch_id} failed: {error_content}")
                    except Exception:
                        logger.error(f"Batch {job.batch_id} failed with unknown error.")
                else:
                    job.status = "failed"
                    logger.warning(f"Batch {job.batch_id} terminated without completion signal.")
                
                if name in self.active_instances:
                    del self.active_instances[name]

            try:
                instance = self.compute_client.get(
                    project=self.gcp_config.project_id,
                    zone=self.gcp_config.zone,
                    instance=instance_name
                )
                
                if instance.status == "TERMINATED":
                    process_terminated_job(batch_job, instance_name)
                    continue

                elif instance.status == "RUNNING":
                    if batch_job.status == "creating":
                        batch_job.status = "running"
                        logger.info(f"Instance {instance_name} is now running.")
                    
                    runtime = time.time() - instance_info.get("created_at", 0)
                    max_runtime = 45 * 60
                    if runtime > max_runtime:
                        logger.warning(f"Instance {instance_name} exceeded max runtime. Terminating.")
                        self.compute_client.delete(
                            project=self.gcp_config.project_id,
                            zone=self.gcp_config.zone,
                            instance=instance_name
                        )
                        batch_job.status = "timeout"
                        if instance_name in self.active_instances:
                            del self.active_instances[instance_name]
                        
            except exceptions.NotFound:
                logger.warning(f"Instance {instance_name} not found. Processing as terminated.")
                process_terminated_job(batch_job, instance_name)
                
            except Exception as e:
                error_str = str(e).lower()
                if 'remotedisconnected' in error_str or 'connection aborted' in error_str:
                    logger.warning(f"Temporary network error for {instance_name}. Will retry.")
                    continue
                
                logger.error(f"Error monitoring instance {instance_name}: {e}", exc_info=True)
                batch_job.status = "error"
                if instance_name in self.active_instances:
                    del self.active_instances[instance_name]
        
        stats["total_networks_generated"] = sum(
            job.networks_count for job in self.batch_jobs if job.status == "completed"
        )
        self.completed_networks = stats["total_networks_generated"]
        return stats

    def cleanup_completed_instances(self):
        """Clean up terminated instances."""
        for instance_name, instance_info in list(self.active_instances.items()):
            batch_job = instance_info["batch_job"]
            
            if batch_job.status in ["completed", "failed", "error", "timeout"]:
                try:
                    self.compute_client.delete(
                        project=self.gcp_config.project_id,
                        zone=self.gcp_config.zone,
                        instance=instance_name
                    )
                    logger.info(f"Deleted instance: {instance_name}")
                except exceptions.NotFound:
                    logger.debug(f"Instance {instance_name} already deleted")
                except Exception as e:
                    logger.warning(f"Could not delete instance {instance_name}: {e}")

    async def generate_networks_distributed(self, max_concurrent_instances: Optional[int] = None) -> bool:
        """Main method to generate networks using distributed cloud infrastructure."""
        if max_concurrent_instances is None:
            max_concurrent_instances = self.gcp_config.max_instances
        
        try:
            self.upload_dependencies()
            batch_jobs = self.create_batch_jobs()
            
            logger.info(f"Starting distributed generation of {self.gcp_config.total_networks} networks")
            logger.info(f"Using up to {max_concurrent_instances} concurrent instances")
            
            pending_jobs = batch_jobs.copy()
            
            while pending_jobs or self.active_instances:
                while len(self.active_instances) < max_concurrent_instances and pending_jobs:
                    job = pending_jobs.pop(0)
                    instance_name = f"network-gen-batch-{job.batch_id:04d}"
                    
                    if self.create_vm_instance(instance_name, job):
                        logger.info(f"Launched instance for batch {job.batch_id}")
                        await asyncio.sleep(5)
                    else:
                        pending_jobs.insert(0, job)
                        logger.warning(f"Failed to launch instance for batch {job.batch_id}, will retry")
                        await asyncio.sleep(30)
                
                stats = self.monitor_instances()
                
                logger.info(
                    f"Progress: {stats['total_networks_generated']}/{self.gcp_config.total_networks} networks, "
                    f"{stats['active_instances']} active instances, "
                    f"{stats['completed_jobs']} completed batches, "
                    f"{stats['failed_jobs']} failed batches"
                )
                
                self.cleanup_completed_instances()
                await asyncio.sleep(30)
            
            final_stats = self.monitor_instances()
            success_rate = final_stats['completed_jobs'] / len(batch_jobs) if batch_jobs else 0
            
            logger.info("Network generation completed")
            logger.info(f"Total networks generated: {final_stats['total_networks_generated']}")
            logger.info(f"Success rate: {success_rate:.2%}")
            
            return success_rate > 0.9
            
        except Exception as e:
            logger.error(f"Error in distributed network generation: {e}", exc_info=True)
            return False

    def get_generation_summary(self) -> Dict[str, Any]:
        """Get a summary of the network generation process."""
        completed_jobs = [job for job in self.batch_jobs if job.status == "completed"]
        failed_jobs = [job for job in self.batch_jobs if job.status in ["failed", "error", "timeout"]]
        
        return {
            "total_batches": len(self.batch_jobs),
            "completed_batches": len(completed_jobs),
            "failed_batches": len(failed_jobs),
            "total_networks_generated": sum(job.networks_count for job in completed_jobs),
            "success_rate": len(completed_jobs) / len(self.batch_jobs) if self.batch_jobs else 0,
            "storage_location": f"gs://{self.gcp_config.storage_bucket}/networks/",
            "generation_time": (
                time.time() - min(job.created_at for job in self.batch_jobs) 
                if self.batch_jobs else 0
            )
        }


async def main():
    """Main execution function for cloud network generation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/cloud_generation.log'),
            logging.StreamHandler()
        ]
    )
    
    try:
        generator = CloudNetworkGenerator()
        
        if not generator.load_node_data():
            logger.error("Failed to load node data. Exiting.")
            return
        
        print(f"Target networks: {generator.gcp_config.total_networks}")
        print(f"Networks per batch: {generator.gcp_config.networks_per_batch}")
        print(f"Max concurrent instances: {generator.gcp_config.max_instances}")
        print(f"Storage bucket: {generator.gcp_config.storage_bucket}")
        
        success = await generator.generate_networks_distributed()
        
        summary = generator.get_generation_summary()
        print(f"Total networks generated: {summary['total_networks_generated']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Generation time: {summary['generation_time']:.1f} seconds")
        print(f"Networks stored at: {summary['storage_location']}")
        
        if success:
            logger.info("Distributed network generation completed successfully")
        else:
            logger.warning("Network generation completed with some failures")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)


if __name__ == '__main__':
    asyncio.run(main())