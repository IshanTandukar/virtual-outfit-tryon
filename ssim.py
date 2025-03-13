import os
from PIL import Image
from simple_extractor import parsing

# Define paths
input_folder = "image"  # Replace with your actual folder path
output_folder = "parseoutput"  # Replace with your actual folder path

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get the first 100 images from the folder
image_files = [f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))][:1000]

# Process and save images
for filename in image_files:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")  # Ensure PNG format

    # Parse and save image
    input_image = Image.open(input_path)
    processed_image = parsing(input_image)

    # Ensure the output image is in 'P' mode before saving
    processed_image = processed_image.convert("P")  
    processed_image.save(output_path)

print(f"Processing complete! {len(image_files)} images saved in: {output_folder}")
