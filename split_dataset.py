import os
import random
import shutil

# Paths (relative to your 'children_safety' folder)
image_source = "dataset"          # Original images
label_source = "labels_yolo"      # YOLO labels from conversion
dest_root = "yolo_dataset"        # Organized dataset

# Split ratios (70% train, 15% val, 15% test)
train_ratio = 0.7
val_ratio = 0.15

# Get all image files and shuffle randomly
all_images = [f for f in os.listdir(image_source) if f.endswith(".jpg")]
random.shuffle(all_images)

# Calculate split sizes
num_train = int(len(all_images) * train_ratio)
num_val = int(len(all_images) * val_ratio)

# Move files to train/val/test folders
for i, img_file in enumerate(all_images):
    txt_file = img_file.replace(".jpg", ".txt")
    src_img = os.path.join(image_source, img_file)
    src_label = os.path.join(label_source, txt_file)

    # Decide destination folder
    if i < num_train:
        dest = "train"
    elif i < num_train + num_val:
        dest = "val"
    else:
        dest = "test"

    # Copy image
    shutil.copy(src_img, os.path.join(dest_root, "images", dest, img_file))
    
    # Copy label (if exists)
    if os.path.exists(src_label):
        shutil.copy(src_label, os.path.join(dest_root, "labels", dest, txt_file))

print(f"Dataset split complete! Files copied to {dest_root}/")