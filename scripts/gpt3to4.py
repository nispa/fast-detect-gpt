
import subprocess

# List of datasets and models to process
datasets = ["xsum", "writing", "pubmed"]
models = ["davinci", "gpt-3.5-turbo", "gpt-4"]

# Iterate over each dataset and model
for dataset in datasets:
    for model in models:
        # Construct the command to run the data_builder script
        command = [
            "python",
            "scripts/data_builder.py",
            f"--dataset", f"{dataset}",
            f"--model", f"{model}",
            "--n_samples", "200",
            "--overwrite_output_dir"
        ]
        
        # Run the command
        subprocess.run(command, check=True)
