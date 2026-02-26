import os
import sys
import platform
import time

def get_box_root():
    """Attempts to automatically find the Box Drive root folder."""
    home = os.path.expanduser("~")
    system = platform.system()
    potential_paths = []

    if system == "Darwin":  # macOS
        potential_paths.append(os.path.join(home, "Library", "CloudStorage", "Box-Box"))
        potential_paths.append(os.path.join(home, "Box"))
    elif system == "Windows":
        potential_paths.append(os.path.join(home, "Box"))
        
    for path in potential_paths:
        if os.path.exists(path):
            return path
    return None

def ensure_files_local(directory, extension=".pkl"):
    """
    Iterates through files to force macOS to download them (hydrate) 
    before the main analysis tries to read them.
    """
    print(f"\nVerifying files in {directory}...")
    print("This may trigger Box to download files. Please wait.")
    
    count = 0
    errors = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                filepath = os.path.join(root, file)
                try:
                    # We open the file in read mode to trigger the cloud download
                    with open(filepath, 'rb') as f:
                        # Read just 1 byte to force the OS to fetch it
                        f.read(1) 
                    count += 1
                    if count % 10 == 0:
                        sys.stdout.write(f"\r  ✓ Checked {count} files...")
                        sys.stdout.flush()
                except OSError as e:
                    print(f"\n  ❌ Error downloading: {file}")
                    print(f"     System Message: {e}")
                    errors += 1
                    
    print(f"\n\nVerification Complete: {count} available, {errors} failed.")
    if errors > 0:
        print("⚠ Some files are stuck in the cloud.")
        print("SOLUTION: Right-click the folder in Finder > 'Make Available Offline'.")
        time.sleep(2) # Give user time to read

def get_data_path(target_folder_name=None):
    """Locates data and runs the verification check."""
    print("\n" + "="*60)
    print("LOCATING DATA DIRECTORY")
    print("="*60)

    box_root = get_box_root()
    found_path = None
    
    if box_root and target_folder_name:
        print(f"✓ Detected Box Drive at: {box_root}")
        for root, dirs, files in os.walk(box_root):
            if target_folder_name in dirs:
                found_path = os.path.join(root, target_folder_name)
                print(f"✓ Found project folder: {found_path}")
                break
            if root.count(os.sep) - box_root.count(os.sep) > 3:
                del dirs[:] 
    
    if not found_path:
        print("\n⚠ Could not automatically locate the data.")
        while True:
            user_input = input(f"Drag and drop the '{target_folder_name}' folder here: ").strip()
            if user_input:
                cleaned = user_input.replace('"', '').replace("'", "").strip()
                if os.path.exists(cleaned):
                    found_path = cleaned
                    break
                else:
                    print("❌ Path does not exist.")

    # RUN THE HYDRATION CHECK
    if found_path:
        ensure_files_local(found_path)
        
    return found_path