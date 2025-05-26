import os
from pathlib import Path

def rename_files(folder, suffix):
    for file in Path(folder).glob(f"*.{suffix}"):
        filename = file.stem  # e.g., Clip_10_000123
        parts = filename.split("_")
        if len(parts) >= 3 and parts[2].startswith("000"):
            # Remove one zero from the last part
            parts[2] = parts[2][1:]
            new_name = "_".join(parts) + f".{suffix}"
            new_path = file.with_name(new_name)
            print(f"Renaming: {file.name} -> {new_path.name}")
            file.rename(new_path)

# Replace with your actual paths
image_folder = "Transdrone_data/Images/val"
label_folder = "Transdrone_data/labels/val"

rename_files(image_folder, "png")
rename_files(label_folder, "txt")
