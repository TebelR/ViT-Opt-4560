import os
import shutil
import glob

def find_and_move_file(search_dir, target_dir, file_pattern, new_filename):
    
    file_paths = glob.glob(os.path.join(search_dir, "**", file_pattern), recursive=True)

    if not file_paths:
        return "File not found."

    old_path = file_paths[0]  # Take the first match
    new_path = os.path.join(target_dir, new_filename)
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Move and rename the file
    shutil.move(old_path, new_path)

    return f"File moved and renamed: {old_path} -> {new_path}"
