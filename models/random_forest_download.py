import gdown
import os
import zipfile

# Ensure the models directory exists
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Google Drive File ID
file_id = "1TAMVaC8oxNKsUB-3KiclMIEroVYhHCIK"
model_url = f"https://drive.google.com/uc?id={file_id}"

# Define local paths
compressed_path = os.path.join(models_dir, "random_forest.zip")
extracted_model_path = os.path.join(models_dir, "random_forest.pkl")

# Download the compressed model if it doesn't exist
if not os.path.exists(extracted_model_path):
    print("Downloading compressed model...")
    gdown.download(model_url, compressed_path, quiet=False)

    # Extract the ZIP file correctly
    print("Extracting model...")
    with zipfile.ZipFile(compressed_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith("random_forest.pkl"): 
                zip_ref.extract(file, models_dir)
                extracted_file_path = os.path.join(models_dir, os.path.basename(file))
                os.rename(os.path.join(models_dir, file), extracted_file_path)

    print("Model extracted successfully!")

    # Remove the compressed file after extraction
    os.remove(compressed_path)
    print("Cleaned up compressed file.")
