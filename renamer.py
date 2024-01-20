import os

def rename_files(image_folder, labels_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.jpeg')]
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    if len(image_files) != len(label_files):
        print("Error: Number of images and labels do not match.")
        return

    for i, (image_file, label_file) in enumerate(zip(image_files, label_files), start=1):
        # Rename image file
        original_image_path = os.path.join(image_folder, image_file)
        new_image_name = f"{i}.jpg"
        new_image_path = os.path.join(image_folder, new_image_name)
        os.rename(original_image_path, new_image_path)

        # Rename label file
        original_label_path = os.path.join(labels_folder, label_file)
        new_label_name = f"{i}.txt"
        new_label_path = os.path.join(labels_folder, new_label_name)
        os.rename(original_label_path, new_label_path)

        print(f"Renamed: {original_image_path} to {new_image_path}")
        print(f"Renamed: {original_label_path} to {new_label_path}")

if __name__ == "__main__":
    # Specify the paths to image and labels folders
    image_folder_path = "path/to/image/folder"
    labels_folder_path = "path/to/labels/folder"

    rename_files(image_folder_path, labels_folder_path)
