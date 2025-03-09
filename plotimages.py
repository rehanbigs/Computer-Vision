import os
from PIL import Image, ExifTags
import matplotlib.pyplot as plt

# Directory containing images
image_dir = '/home/rehanfarooq/cv/images'  # Replace with your images folder path

def apply_exif_orientation(image):
    """Corrects image orientation based on EXIF data, if available."""
    try:
        # Identify the EXIF orientation tag
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        # Apply orientation correction if available
        exif = image._getexif()
        if exif and orientation in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # If no EXIF orientation data is present, leave the image as is
        pass
    return image

# Iterate over all images in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
        image_path = os.path.join(image_dir, filename)  # Full path to the image
        
        # Load the image
        image = Image.open(image_path)
        image = apply_exif_orientation(image)  # Apply EXIF orientation correction

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(filename)
        plt.axis('off')
        
        # Show the image and wait for the plot window to close
        plt.show()

