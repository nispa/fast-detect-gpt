
import subprocess

# List of datasets and their corresponding reference models
datasets = {
    "xsum": "gpt-neo-2.7B",
    "squad": "gpt-neo-2.7B",
    "writing": "gpt-neo-2.7B",
    "pubmed": "gpt-neo-2.7B"
}

# List of models to be tested
models = ["gpt-j-6B", "gpt-neox-20b", "gpt-3.5-turbo", "gpt-4"]

# Iterate over each dataset and its reference model
for dataset, ref_model in datasets.items():
    for model in models:
        # Construct the command
        command = [
            "python", "scripts/fast_detect_gpt.py",
            f"--ref_model_name", f"{ref_model}",
            f"--model_name", f"{model}",
            f"--dataset", f"{dataset}",
            "--n_samples", "200"
        ]
        
        # Run the command
        subprocess.run(command, check=True)
