import os

def rename_images(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    # Filter only image files (you can extend this check based on the image formats you have)
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
    
    # Sort the image files
    image_files.sort()

    # Rename each image file numerically
    for i, old_name in enumerate(image_files, start=1):
        # Create the new file name
        extension = os.path.splitext(old_name)[1]
        new_name = f"{i:03d}{extension}"  # Using 3 digits, e.g., 001, 002, ...

        # Build the full paths
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)

if __name__ == "__main__":
    folder_path = '/path/to/your/images/folder'  # Update this to the path of your image folder
    rename_images(folder_path)
