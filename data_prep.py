import os
import json
from PyPDF2 import PdfReader

# Define the input and output directories
input_dir = "/Users/kiwitech/Documents/USWASDE"
output_dir = "/Users/kiwitech/Documents/labelled_wasde_jsonl"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define a function to extract text and metadata from PDF
def extract_pdf_metadata(pdf_path):
    try:
        pdf_reader = PdfReader(pdf_path)
        full_text = ""
        
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        
        # Sample metadata
        metadata = {
            "year": 2024,
            "month": "January",
            "report_number": "WASDE-644",
            "issn": "1554-9089",
            "commodity_type": ["Wheat", "Corn", "Rice", "Oilseeds", "Cotton", "Sugar", "Livestock", "Poultry", "Dairy"],
            "summary_type": "Global and U.S.",
            "key_statistics": {
                "wheat_production": "U.S. and global estimates",
                "corn_supply_and_use": "U.S. and global details",
                "rice_imports_and_exports": "Global outlook",
                "oilseeds_price_forecast": "U.S. and global price projections"
            },
            "authors_contributors": [
                {"name": "Mark Simone", "role": "ICEC Chair, WAOB"},
                {"name": "Michael Jewison", "role": "ICEC Chair, WAOB"},
                {"name": "Joanna Hitchner", "role": "ICEC Chair, WAOB"}
            ],
            "approval_details": {
                "approved_by": "Robert Bonnie, Secretary of Agriculture Designate",
                "chairman": "Mark Jekanowski, Chairman of the World Agricultural Outlook Board"
            },
            "full_text": full_text
        }
        
        return metadata

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

# Process each PDF in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(input_dir, filename)
        metadata = extract_pdf_metadata(pdf_path)
        
        if metadata:
            # Define the output JSONL file for the current PDF
            jsonl_file_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jsonl")
            
            # Write the metadata to the JSONL file
            try:
                with open(jsonl_file_path, 'w') as jsonl_file:
                    jsonl_file.write(json.dumps(metadata) + '\n')
                print(f"Created JSONL file for: {filename}")
            except Exception as e:
                print(f"Error writing JSONL file for {filename}: {e}")
        else:
            print(f"Skipping {filename} due to extraction failure.")

print("Metadata extraction and JSONL file creation complete.")
 