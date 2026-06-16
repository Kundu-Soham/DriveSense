import os

# --- PATH CONFIGURATION ---
# Since this script sits inside 'Scripts/', we go up one level to 'DRIVESENSE'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed Dataset")
IMAGES_ROOT = os.path.join(PROCESSED_DIR, "images")
LABELS_ROOT = os.path.join(PROCESSED_DIR, "labels")


def get_split_stats(split_name):
    """Parses YOLO label files within a specific split to count total frames and bounding box annotations."""
    split_images_dir = os.path.join(IMAGES_ROOT, split_name)
    split_labels_dir = os.path.join(LABELS_ROOT, split_name)

    total_frames = 0
    class_0_count = 0  # Not Crossing [cite: 34]
    class_1_count = 0  # Crossing [cite: 34]

    # Verify both directories exist for the split
    if not os.path.exists(split_images_dir) or not os.path.exists(split_labels_dir):
        return 0, 0, 0

    # Count actual image frames in the images split directory
    for img_name in os.listdir(split_images_dir):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            total_frames += 1

            # Determine the corresponding txt label filename
            base_name = os.path.splitext(img_name)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(split_labels_dir, label_file)

            # If an annotation file exists for this frame, parse its contents
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        # First element in a YOLO line format is the class ID [cite: 34]
                        class_id = parts[0]
                        if class_id == "0":
                            class_0_count += 1
                        elif class_id == "1":
                            class_1_count += 1

    return total_frames, class_0_count, class_1_count


def generate_report():
    if not os.path.exists(PROCESSED_DIR):
        print(f"Error: Processed Dataset folder not found at {PROCESSED_DIR}")
        return

    splits = ["train", "val", "test"]

    print("=" * 65)
    print(f"{'JAAD YOLO DATASET STATISTICS':^65}")
    print("=" * 65)
    print(
        f"{'Split':<10} | {'Total Frames':<13} | {'Not Crossing (Class 0)':<22} | {'Crossing (Class 1)':<15}"
    )
    print("-" * 65)

    grand_total_frames = 0
    grand_total_c0 = 0
    grand_total_c1 = 0

    for split in splits:
        tot_frames, c0, c1 = get_split_stats(split)

        grand_total_frames += tot_frames
        grand_total_c0 += c0
        grand_total_c1 += c1

        print(f"{split.capitalize():<10} | {tot_frames:<13} | {c0:<22} | {c1:<15}")

    print("-" * 65)
    print(
        f"{'Total':<10} | {grand_total_frames:<13} | {grand_total_c0:<22} | {grand_total_c1:<15}"
    )
    print("=" * 65)


if __name__ == "__main__":
    generate_report()