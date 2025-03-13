# import os
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# from PIL import Image

# # Define paths
# parsed_folder = "parseoutput"  # Folder containing parsed images
# reference_folder = "image-parse-v3"  # Folder with original/reference images

# # Get the list of images (assuming names match)
# image_files = [f for f in os.listdir(parsed_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

# # Function to calculate SSIM
# def calculate_ssim(img1_path, img2_path):
#     img1 = Image.open(img1_path).convert("L")  # Convert to grayscale
#     img2 = Image.open(img2_path).convert("L")  # Convert to grayscale

#     img1 = np.array(img1)
#     img2 = np.array(img2)

#     # Ensure both images have the same size
#     if img1.shape != img2.shape:
#         img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

#     # Compute SSIM
#     score, _ = ssim(img1, img2, full=True)
#     return score

# # Compute SSIM for each pair of images
# ssim_scores = {}
# for filename in image_files:
#     parsed_image_path = os.path.join(parsed_folder, filename)
#     reference_image_path = os.path.join(reference_folder, filename)

#     if os.path.exists(reference_image_path):
#         score = calculate_ssim(parsed_image_path, reference_image_path)
#         ssim_scores[filename] = score
#     else:
#         print(f"Reference image not found for: {filename}")

# # Print SSIM scores
# for filename, score in ssim_scores.items():
#     print(f"SSIM for {filename}: {score:.4f}")

import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# Define paths
parsed_folder = "parseoutput"  # Folder containing parsed images
reference_folder = "image-parse-v3"  # Folder with original/reference images

# Get the list of images (assuming names match)
image_files = [f for f in os.listdir(parsed_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

# Function to calculate SSIM for two images
def calculate_ssim(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("L")  # Convert to grayscale
    img2 = Image.open(img2_path).convert("L")  # Convert to grayscale

    img1 = np.array(img1)
    img2 = np.array(img2)

    # Ensure both images have the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute SSIM
    score, _ = ssim(img1, img2, full=True)
    return score

# Compute SSIM for each pair of images
ssim_scores = []
for filename in image_files:
    parsed_image_path = os.path.join(parsed_folder, filename)
    reference_image_path = os.path.join(reference_folder, filename)

    if os.path.exists(reference_image_path):
        score = calculate_ssim(parsed_image_path, reference_image_path)
        ssim_scores.append(score)
    else:
        print(f"Reference image not found for: {filename}")

# Compute overall SSIM
if ssim_scores:
    overall_ssim = sum(ssim_scores) / len(ssim_scores)
    print(f"\n✅ Overall SSIM: {overall_ssim:.4f}")
else:
    print("\n⚠️ No valid image pairs found for SSIM calculation.")
