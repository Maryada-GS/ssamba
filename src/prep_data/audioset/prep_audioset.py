"""
Author: Siavash Shams
Date: 03/29/2024
"""

import os
import json
import re


def create_json(directory, output_filename):
    data_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".flac"):
            file_path = os.path.join(directory, filename)
            data_list.append({"wav": file_path, "labels": "/m/01b_21"})

    with open(output_filename, "w") as outfile:
        json.dump({"data": data_list}, outfile, indent=4)


def remove_error_files(log_file_path):
    # Regular expression to match the error lines and capture the file paths
    error_line_regex = r"Error processing file (\S+):"

    # Counters
    removed_files_count = 0
    error_files_not_found_count = 0

    # Open and read the log file
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            # Search for error lines
            match = re.search(error_line_regex, line)
            if match:
                error_file_path = match.group(1)
                try:
                    # Remove the file if it exists
                    os.remove(error_file_path)
                    print(f"Removed file: {error_file_path}")
                    removed_files_count += 1
                except FileNotFoundError:
                    print(f"File not found, cannot remove: {error_file_path}")
                    error_files_not_found_count += 1

    print(f"Total removed files: {removed_files_count}")
    print(f"Total error files not found: {error_files_not_found_count}")


DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "audioset")
os.makedirs(os.path.join(DATASET_ROOT, "audio", "unbal_train"), exist_ok=True)
os.makedirs(os.path.join(DATASET_ROOT, "audio", "eval"), exist_ok=True)
os.makedirs(os.path.join(DATASET_ROOT, "datafiles"), exist_ok=True)

# Directory containing the audio files
train_directory = os.path.join(DATASET_ROOT, "audio", "unbal_train")
eval_directory = os.path.join(DATASET_ROOT, "audio", "eval")

# Output JSON file names
train_output_filename = os.path.join(DATASET_ROOT, "datafiles", "unbal_train_data.json")
eval_output_filename = os.path.join(DATASET_ROOT, "datafiles", "eval_data.json")

log_file_path = os.path.join(DATASET_ROOT, "preprocess_errors.log")

# Execute the function with your log file (only if log exists)
if os.path.exists(log_file_path):
    remove_error_files(log_file_path)


# Create JSON for training data
create_json(train_directory, train_output_filename)

# Create JSON for evaluation data
create_json(eval_directory, eval_output_filename)

print(f"JSON files created: {train_output_filename}, {eval_output_filename}")
