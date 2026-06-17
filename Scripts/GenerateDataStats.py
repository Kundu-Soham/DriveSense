import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed JAAD Dataset")
IMAGES_ROOT = os.path.join(PROCESSED_DIR, "images")
LABELS_ROOT = os.path.join(PROCESSED_DIR, "labels")

def get_split_stats(split_name):
    split_images_dir = os.path.join(IMAGES_ROOT, split_name)
    split_labels_dir = os.path.join(LABELS_ROOT, split_name)
    
    if not os.path.exists(split_images_dir) or not os.path.exists(split_labels_dir):
        return 0, 0, 0

    tot_frames = c0 = c1 = 0
    for img_name in os.listdir(split_images_dir):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            tot_frames += 1
            label_path = os.path.join(split_labels_dir, f"{os.path.splitext(img_name)[0]}.txt")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            if parts[0] == "0": c0 += 1
                            elif parts[0] == "1": c1 += 1
    return tot_frames, c0, c1

def generate_visual_report():
    if not os.path.exists(PROCESSED_DIR): return

    splits = ["train", "val", "test"]
    plot_data = []
    g_frames = g_c0 = g_c1 = 0

    print("=" * 65 + f"\n{'JAAD DATASET STATISTICS':^65}\n" + "=" * 65)
    print(f"{'Split':<10} | {'Total Frames':<13} | {'Not Crossing (Class 0)':<22} | {'Crossing (Class 1)':<15}\n" + "-" * 65)

    for s in splits:
        tot, c0, c1 = get_split_stats(s)
        g_frames, g_c0, g_c1 = g_frames + tot, g_c0 + c0, g_c1 + c1
        print(f"{s.capitalize():<10} | {tot:<13} | {c0:<22} | {c1:<15}")
        plot_data.append((s.capitalize(), tot, c0, c1))

    print("-" * 65 + f"\n{'Total':<10} | {g_frames:<13} | {g_c0:<22} | {g_c1:<15}\n" + "=" * 65)

    plt.rcParams.update({'font.sans-serif': 'Helvetica', 'axes.edgecolor': '#CCCCCC'})
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    labels, total_frames, c0_vals, c1_vals = zip(*plot_data)
    bars_c0 = ax.bar(labels, c0_vals, width=0.5, label='Not Crossing (Class 0)', color='#1f77b4')
    bars_c1 = ax.bar(labels, c1_vals, bottom=c0_vals, width=0.5, label='Crossing (Class 1)', color='#ff7f0e')
    
    for b_c0, b_c1, label_total in zip(bars_c0, bars_c1, total_frames):
        h_c0, h_c1 = b_c0.get_height(), b_c1.get_height()
        if h_c0 > 0:
            ax.text(b_c0.get_x() + 0.25, h_c0 / 2, f'{h_c0}', ha='center', va='center', color='white', fontweight='bold')
        if h_c1 > 0:
            ax.text(b_c1.get_x() + 0.25, h_c0 + (h_c1 / 2), f'{h_c1}', ha='center', va='center', color='white', fontweight='bold')
        
        ax.text(b_c0.get_x() + 0.25, (h_c0 + h_c1) * 1.01, f'Total: {label_total}', ha='center', va='bottom', fontweight='bold', color='#333333')

    ax.set_title('JAAD Dataset: Annotation Class Distribution Across Splits', fontsize=14, pad=20, fontweight='bold')
    ax.set_xlabel('Dataset Split', fontsize=12, labelpad=10)
    ax.set_ylabel('Number of Bounding Box Instances', fontsize=12, labelpad=10)
    ax.legend(loc='upper right', frameon=True, edgecolor='#E0E0E0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "dataset_statistics.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_visual_report()