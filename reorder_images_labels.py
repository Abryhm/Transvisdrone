import os
import shutil
from pathlib import Path
from collections import defaultdict

# Input folders
images_dir = Path("/media/iml1/Disk2/UMAIR/Drone-Detection/all_images")
labels_dir = Path("/media/iml1/Disk2/UMAIR/Drone-Detection/all_labels")

# Output folders
output_images_dir = Path("/media/iml1/Disk2/UMAIR/Drone-Detection/renamed_images")
output_labels_dir = Path("/media/iml1/Disk2/UMAIR/Drone-Detection/renamed_labels")

output_images_dir.mkdir(parents=True, exist_ok=True)
output_labels_dir.mkdir(parents=True, exist_ok=True)

# Group files by clip
clips = defaultdict(list)

for img_file in images_dir.glob("Clip_*.png"):
    name = img_file.stem  # e.g., Clip_1_000056
    clip_prefix, frame_num = name.rsplit("_", 1)
    clips[clip_prefix].append((int(frame_num), img_file))

# Process each clip
for clip_prefix, frames in clips.items():
    # Sort frames by original number
    frames.sort()
    min_frame = frames[0][0]

    for new_idx, (orig_frame_num, img_path) in enumerate(frames):
        new_frame_str = f"{new_idx:06d}"
        new_name = f"{clip_prefix}_{new_frame_str}"

        # Copy and rename image
        new_img_path = output_images_dir / f"{new_name}.png"
        shutil.copy(img_path, new_img_path)

        # Copy and rename label if it exists
        label_path = labels_dir / f"{clip_prefix}_{orig_frame_num:06d}.txt"
        if label_path.exists():
            new_label_path = output_labels_dir / f"{new_name}.txt"
            shutil.copy(label_path, new_label_path)

print("Renaming and copying complete.")
