
import subprocess

# List of top_p values to test
top_p_values = [0.8, 0.85, 0.9, 0.95, 1.0]

# Iterate over each top_p value
for top_p in top_p_values:
    # Construct the command
    command = [
        "python", "scripts/fast_detect_gpt.py",
        "--ref_model_name", "gpt-neo-2.7B",
        "--model_name", "gpt-j-6B",
        "--dataset", "xsum",
        "--n_samples", "200",
        f"--top_p", f"{top_p}"
    ]
    
    # Run the command
    subprocess.run(command, check=True)
