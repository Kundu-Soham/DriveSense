import os
import matplotlib.pyplot as plt

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed Dataset")
IMAGES_ROOT = os.path.join(PROCESSED_DIR, "images")
LABELS_ROOT = os.path.join(PROCESSED_DIR, "labels")


def get_split_stats(split_name):
    """Parses YOLO label files within a specific split to count total frames and bounding box annotations."""
    split_images_dir = os.path.join(IMAGES_ROOT, split_name)
    split_labels_dir = os.path.join(LABELS_ROOT, split_name)

    total_frames = 0
    class_0_count = 0  # Not Crossing
    class_1_count = 0  # Crossing

    if not os.path.exists(split_images_dir) or not os.path.exists(split_labels_dir):
        return 0, 0, 0

    for img_name in os.listdir(split_images_dir):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            total_frames += 1

            base_name = os.path.splitext(img_name)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(split_labels_dir, label_file)

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        class_id = parts[0]
                        if class_id == "0":
                            class_0_count += 1
                        elif class_id == "1":
                            class_1_count += 1

    return total_frames, class_0_count, class_1_count


def generate_visual_report():
    if not os.path.exists(PROCESSED_DIR):
        print(f"Error: Processed Dataset folder not found at {PROCESSED_DIR}")
        return

    splits = ["train", "val", "test"]
    
    # Lists to hold plotting data
    split_labels = []
    not_crossing_counts = []
    crossing_counts = []

    # Text report metrics
    grand_total_frames = 0
    grand_total_c0 = 0
    grand_total_c1 = 0

    print("=" * 65)
    print(f"{'JAAD DATASET STATISTICS':^65}")
    print("=" * 65)
    print(f"{'Split':<10} | {'Total Frames':<13} | {'Not Crossing (Class 0)':<22} | {'Crossing (Class 1)':<15}")
    print("-" * 65)

    for split in splits:
        tot_frames, c0, c1 = get_split_stats(split)

        grand_total_frames += tot_frames
        grand_total_c0 += c0
        grand_total_c1 += c1

        print(f"{split.capitalize():<10} | {tot_frames:<13} | {c0:<22} | {c1:<15}")
        
        # Append to arrays for charting
        split_labels.append(split.capitalize())
        not_crossing_counts.append(c0)
        crossing_counts.append(c1)

    print("-" * 65)
    print(f"{'Total':<10} | {grand_total_frames:<13} | {grand_total_c0:<22} | {grand_total_c1:<15}")
    print("=" * 65)

    # --- VISUAL DIAGRAM GENERATION ---
    print("\nGenerating dataset statistics diagram...")
    
    # Setup aesthetic dark/clean theme style parameters
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    bar_width = 0.5
    
    # Render Stacked Bars (Class 0 at base, Class 1 stacked directly on top)
    bars_c0 = ax.bar(split_labels, not_crossing_counts, width=bar_width, label='Not Crossing (Class 0)', color='#1f77b4')
    bars_c1 = ax.bar(split_labels, crossing_counts, bottom=not_crossing_counts, width=bar_width, label='Crossing (Class 1)', color='#ff7f0e')
    
    # Add exact numerical value labels inside/above the bars
    for b_c0, b_c1 in zip(bars_c0, bars_c1):
        h_c0 = b_c0.get_height()
        h_c1 = b_c1.get_height()
        total_h = h_c0 + h_c1
        
        # Display class counts centered inside their respective bar segments if large enough
        if h_c0 > 0:
            ax.text(b_c0.get_x() + b_c0.get_width()/2., h_c0 / 2, f'{h_c0}', ha='center', va='center', color='white', fontweight='bold')
        if h_c1 > 0:
            ax.text(b_c1.get_x() + b_c1.get_width()/2., h_c0 + (h_c1 / 2), f'{h_c1}', ha='center', va='center', color='white', fontweight='bold')
            
        # Display overall combined bounding box total on top of the bar
        ax.text(b_c0.get_x() + b_c0.get_width()/2., total_h + (total_h * 0.01), f'Total: {total_h}', ha='center', va='bottom', fontweight='bold', color='#333333')

    # Figure Formatting
    ax.set_title('JAAD Dataset: Annotation Class Distribution Across Splits', fontsize=14, pad=20, fontweight='bold', color='#222222')
    ax.set_xlabel('Dataset Split', fontsize=12, labelpad=10)
    ax.set_ylabel('Number of Bounding Box Instances', fontsize=12, labelpad=10)
    ax.legend(loc='upper right', frameon=True, edgecolor='#E0E0E0')
    
    # Clean up grid borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')
    ax.set_axisbelow(True)

    plt.tight_layout()
    
    # Save the output figure image
    output_path = os.path.join(BASE_DIR, "dataset_statistics.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Successfully saved diagram figure to: {output_path}")
    plt.close()


if __name__ == "__main__":
    generate_visual_report()