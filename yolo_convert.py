import os
import cv2

# Define class name to ID mapping (adjust based on your dataset)
CLASS_MAPPING = {
    "battery": 0,
    "dice": 1,
    "toycar": 2,
    "candle": 3,
    "highlighter": 4,
    "spoon": 5
}

def kitti_to_yolo(kitti_line, img_width, img_height):
    parts = kitti_line.split()
    class_name = parts[0]  # Now reading class name (e.g., "dice")
    
    # Map class name to ID
    class_id = CLASS_MAPPING.get(class_name.lower())
    if class_id is None:
        raise ValueError(f"Unknown class: {class_name}")

    x1, y1, x2, y2 = map(float, parts[4:8])  # Bounding box coordinates
    
    # Convert to YOLO format (normalized)
    center_x = (x1 + x2) / 2 / img_width
    center_y = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"

# Rest of the script remains the same...
dataset_path = "dataset"
output_label_dir = "labels_yolo"
os.makedirs(output_label_dir, exist_ok=True)

for img_file in os.listdir(dataset_path):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(dataset_path, img_file)
        txt_file = img_file.replace(".jpg", ".txt")
        txt_path = os.path.join(dataset_path, txt_file)
        
        if not os.path.exists(txt_path):
            print(f"Warning: No annotation for {img_file}. Skipping.")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_file}. Skipping.")
            continue
        img_height, img_width = img.shape[:2]
        
        with open(txt_path, 'r') as f:
            kitti_lines = f.readlines()
        
        yolo_lines = []
        for line in kitti_lines:
            try:
                yolo_line = kitti_to_yolo(line.strip(), img_width, img_height)
                yolo_lines.append(yolo_line)
            except ValueError as e:
                print(f"Error in {txt_file}: {e}. Skipping line: '{line.strip()}'")
        
        if yolo_lines:  # Only save if valid annotations exist
            output_txt_path = os.path.join(output_label_dir, txt_file)
            with open(output_txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))

print("Conversion complete! YOLO labels saved in:", output_label_dir)