
import subprocess
import os

# Set CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Construct the command to run the paraphrasing script
command = [
    "python",
    "scripts/paraphrasing.py",
    "--model_name_or_path", "t5-3b",
    "--batch_size", "1",
    "--input_file", "data/writing_gpt-4.jsonl",
    "--output_file", "data/writing_gpt-4_paraphrased.jsonl"
]

# Run the command
subprocess.run(command, check=True)
