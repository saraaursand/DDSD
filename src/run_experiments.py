"""
Load and run experiments from YAML config
"""
import yaml
import subprocess
import sys

def load_experiments(config_path="experiments.yaml"):
    """Load experiments from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['experiments']

def build_command(experiment):
    """Build command line arguments from experiment dict"""
    cmd = [sys.executable, "src/retrain.py"]
    
    for key, value in experiment.items():
        if key != "name":
            cmd.append(f"--{key}")
            
            # Handle lists (e.g., unfreeze_layer_indices)
            if isinstance(value, list):
                cmd.extend(str(v) for v in value)
            else:
                cmd.append(str(value))
    
    return cmd

def run_experiments(config_path="experiments.yaml"):
    """Run all experiments sequentially"""
    experiments = load_experiments(config_path)
    
    print(f"\n{'='*60}")
    print(f"Running {len(experiments)} experiments from {config_path}")
    print(f"{'='*60}\n")
    
    for idx, exp in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {idx}/{len(experiments)}: {exp['name']}")
        print(f"{'='*60}")
        
        cmd = build_command(exp)
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"✅ Experiment '{exp['name']}' completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Experiment '{exp['name']}' failed")
            print("Continuing with next experiment...\n")
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ All experiments completed!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else "experiments.yaml"
    run_experiments(config_file)