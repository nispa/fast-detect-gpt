
import subprocess

# Common settings
DATASET = "xsum"
DATA_PATH = f"data/{DATASET}_gpt-4.jsonl"

# List of methods and their specific configurations
methods = [
    # Likelihood
    ("likelihood", ["gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b"]),
    # Log-Likelihood
    ("log_rank", ["gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b"]),
    # Rank
    ("rank", ["gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b"]),
    # Entropy
    ("entropy", ["gpt-neo-2.7B", "gpt-j-6B", "gpt-neox-20b"]),
    # LRR
    ("lrr", ["t5-11b", "gpt-neo-2.7B"]),
    # NPR
    ("npr", ["t5-11b", "gpt-neo-2.7B"]),
    # DetectGPT
    ("detect_gpt", ["t5-11b", "gpt-neo-2.7B"]),
    # DNA-GPT
    ("dna_gpt", ["gpt-neo-2.7B", "gpt-j-6B"]),
    # Fast-DetectGPT
    ("fast_detect_gpt", ["t5-11b", "gpt-neo-2.7B"])
]

# Iterate over each method and its models
for method, models_list in methods:
    for model in models_list:
        # Construct the command
        command = [
            "python", "scripts/detect_llm.py",
            f"--method", f"{method}",
            f"--model_name", f"{model}",
            f"--data_path", f"{DATA_PATH}",
            "--n_samples", "200"
        ]
        
        # Add perturbation_model_name for specific methods
        if method in ["lrr", "npr", "detect_gpt", "fast_detect_gpt"]:
            command.extend(["--perturbation_model_name", "t5-3b"])
        
        # Run the command
        subprocess.run(command, check=True)
