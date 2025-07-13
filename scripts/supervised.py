
import subprocess

# List of datasets and their corresponding configurations
datasets = {
    "xsum": ["roberta-base", "roberta-large"],
    "squad": ["roberta-base", "roberta-large"],
    "writing": ["roberta-base", "roberta-large"]
}

# Iterate over each dataset and its models
for dataset, models in datasets.items():
    for model in models:
        # Construct the command
        command = [
            "python", "scripts/supervised.py",
            f"--model_name", f"{model}",
            f"--dataset", f"{dataset}",
            "--n_samples", "200"
        ]
        
        # Run the command
        subprocess.run(command, check=True)
