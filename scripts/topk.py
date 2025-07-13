
import subprocess

# List of top_k values to test
top_k_values = [1, 5, 10, 20, 30, 40, 50]

# Iterate over each top_k value
for top_k in top_k_values:
    # Construct the command
    command = [
        "python", "scripts/fast_detect_gpt.py",
        "--ref_model_name", "gpt-neo-2.7B",
        "--model_name", "gpt-j-6B",
        "--dataset", "xsum",
        "--n_samples", "200",
        f"--top_k", f"{top_k}"
    ]
    
    # Run the command
    subprocess.run(command, check=True)
