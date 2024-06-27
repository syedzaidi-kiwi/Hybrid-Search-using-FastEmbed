import os
from pathlib import Path
from langchain_community.document_loaders import JSONLoader
from datasets import Dataset
import json

# Path to the folder containing JSONL files
folder_path = "/Users/kiwitech/Documents/Final_Wasde/data/labelled_wasde_jsonl"

# Function to load data using Langchain JSONLoader
def load_data_with_jsonloader(folder_path):
    all_data = []
    files_read = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            print(f"Reading file: {filename}")
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.',
                json_lines=True,
                text_content=False  # Set this to False to handle JSON objects
            )
            data = loader.load()
            for document in data:
                all_data.append(json.loads(document.page_content))  # Parse the JSON content to dictionary
            files_read.append(filename)
    
    return all_data, files_read

# Read all JSONL files
data, files_read = load_data_with_jsonloader(folder_path)

# Ensure all JSONL files are read
expected_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
if set(expected_files) != set(files_read):
    print("Not all JSONL files were read. Please check the folder.")
    missing_files = set(expected_files) - set(files_read)
    print(f"Missing files: {missing_files}")
else:
    print("All JSONL files have been read successfully.")

# Check if there is data to push
if data:
    # Convert the list of dictionaries to a proper format for Hugging Face Dataset
    dataset = Dataset.from_list(data)

    # Push the dataset to Hugging Face
    dataset.push_to_hub("US_Wasde_Data", token="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")

    print("Data successfully pushed to Hugging Face!")
else:
    print("No data found to push to Hugging Face.")




