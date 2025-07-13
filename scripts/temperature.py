
import subprocess

# List of temperatures to test
temperatures = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Iterate over each temperature
for temp in temperatures:
    # Construct the command
    command = [
        "python", "scripts/fast_detect_gpt.py",
        "--ref_model_name", "gpt-neo-2.7B",
        "--model_name", "gpt-j-6B",
        "--dataset", "xsum",
        "--n_samples", "200",
        f"--temperature", f"{temp}"
    ]
    
    # Run the command
    subprocess.run(command, check=True)
