import os
import cv2
from tqdm import tqdm

def preprocess_jaad_dataset(base_dir):
    """
    Loops through the raw full-frame images, reads the YOLO bounding boxes,
    crops out the pedestrian, resizes the crop to 224x224, and saves it
    to a new 'Preprocessed_Dataset' directory.
    """
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_src_dir = os.path.join(base_dir, 'Processed Dataset', 'images', split)
        label_src_dir = os.path.join(base_dir, 'Processed Dataset', 'labels', split)
        
        # Target directories for clean output
        img_dst_dir = os.path.join(base_dir, 'Preprocessed_Dataset', 'images', split)
        label_dst_dir = os.path.join(base_dir, 'Preprocessed_Dataset', 'labels', split)
        
        os.makedirs(img_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)
        
        if not os.path.exists(img_src_dir):
            print(f"Directory not found, skipping split: {img_src_dir}")
            continue
            
        print(f"Processing {split} split...")
        img_files = sorted([f for f in os.listdir(img_src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        for img_name in tqdm(img_files):
            img_path = os.path.join(img_src_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(label_src_dir, label_name)
            
            # Read full image via OpenCV
            img = cv2.imread(img_path)
            if img is None:
                continue
            h_img, w_img, _ = img.shape
            
            crossing_label = "0"
            crop_success = False
            
            # If label file exists and is not empty, extract coordinates
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    line = f.readline().split()
                    if line:
                        crossing_label = line[0]
                        x_center, y_center, w_box, h_box = map(float, line[1:5])
                        
                        # Convert normalized coordinates to absolute pixel values
                        xmin = int((x_center - w_box / 2) * w_img)
                        xmax = int((x_center + w_box / 2) * w_img)
                        ymin = int((y_center - h_box / 2) * h_img)
                        ymax = int((y_center + h_box / 2) * h_img)
                        
                        # Clamp bounds to image dimensions
                        xmin, xmax = max(0, xmin), min(w_img, xmax)
                        ymin, ymax = max(0, ymin), min(h_img, ymax)
                        
                        # Crop the image around the box
                        if (xmax > xmin) and (ymax > ymin):
                            cropped_img = img[ymin:ymax, xmin:xmax]
                            # Resize to target uniform dimension
                            final_img = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_LINEAR)
                            crop_success = True
            
            # Fallback if no label coordinates or bad crop dimensions found
            if not crop_success:
                final_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # Save the preprocessed 224x224 crop image
            cv2.imwrite(os.path.join(img_dst_dir, img_name), final_img)
            
            # Save a simplified corresponding text label containing just the binary target class
            with open(os.path.join(label_dst_dir, label_name), 'w') as f_out:
                f_out.write(f"{crossing_label}\n")

if __name__ == "__main__":
    # Assumes execution from the root project folder containing your dataset
    BASE_DIRECTORY = "." 
    preprocess_jaad_dataset(BASE_DIRECTORY)
    print("\nData preprocessing complete! Cleaned crops are located in './Preprocessed_Dataset'")