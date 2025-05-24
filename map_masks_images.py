import os
import cv2
import numpy as np
from tqdm import tqdm

# Set your paths
images_folder = '/media/iml1/Disk2/UMAIR/Drone-Detection/frames_output_folder_updated'
masks_folder = '/media/iml1/Disk2/UMAIR/Drone-Detection/masks_output_folder_updated'
output_folder = '/media/iml1/Disk2/UMAIR/Drone-Detection/masked_outputs_updated'  # For visualization only
bbox_images_folder = '/media/iml1/Disk2/UMAIR/Drone-Detection/all_images'      # Original images with objects
labels_folder = '/media/iml1/Disk2/UMAIR/Drone-Detection/all_labels'                # YOLO format labels

# Create output directories
os.makedirs(output_folder, exist_ok=True)
os.makedirs(bbox_images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

# List all image files (assuming .png; change if needed)
image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.png')])

# Loop through each image
for img_file in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(images_folder, img_file)
    mask_path = os.path.join(masks_folder, img_file)  # assumes same filename
    
    # Read image and mask
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)  # grayscale

    if image is None or mask is None:
        print(f"Skipping {img_file}: image or mask not found.")
        continue

    # Ensure binary mask (0 and 1)
    binary_mask = (mask > 0).astype(np.uint8)

    # Find contours in the binary mask
    contours_info = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    
    # Skip if no contours found (no objects)
    if not contours:
        continue
    
    # Save the original image (without any boxes) to bbox_images_folder
    cv2.imwrite(os.path.join(bbox_images_folder, img_file), image)
    
    # Create YOLO format label file
    label_path = os.path.join(labels_folder, os.path.splitext(img_file)[0] + '.txt')
    height, width = image.shape[:2]
    
    with open(label_path, 'w') as f:
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to YOLO format (normalized center x, center y, width, height)
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height
            
            # Write to label file (class 0 for drone, adjust if needed)
            f.write(f'0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n')

    # (Optional) Visualization with transparent red mask only (no bounding boxes)
    color_mask = np.zeros_like(image)
    color_mask[binary_mask == 1] = [0, 0, 255]  # Red
    alpha = 0.5
    overlayed = cv2.addWeighted(image, 1, color_mask, alpha, 0)
    cv2.imwrite(os.path.join(output_folder, img_file), overlayed)