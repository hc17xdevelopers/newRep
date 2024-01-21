from PIL import Image, ExifTags
import os

def convert_jpeg_to_png(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            # Load the JPEG image
            jpeg_path = os.path.join(input_folder, filename)
            image = Image.open(jpeg_path)

            # Check for orientation information in EXIF data
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(image._getexif().items())
                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # No EXIF data or no orientation information
                pass

            # Create the corresponding PNG filename
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_folder, png_filename)

            # Save as PNG
            image.save(png_path, "PNG")
            print(f"Converted {filename} to {png_filename}")

if __name__ == "__main__":
    # Set your input and output folders
    input_folder = "path/to/jpeg_folder"
    output_folder = "path/to/png_folder"

    convert_jpeg_to_png(input_folder, output_folder)
