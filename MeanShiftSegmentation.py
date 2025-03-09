import numpy as np
from skimage import io, color
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import os
import time

# Define the path to the images folder
images_folder = '/home/rehanfarooq/cv/images'
output_folder = '/home/rehanfarooq/cv/output_segmented'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each file in the images folder
for image_name in os.listdir(images_folder):
    # Construct full path to the image file
    image_path = os.path.join(images_folder, image_name)

    # Check if the file is an image (optional: based on file extensions like jpg, png)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing image: {image_name}")

        # Start the timer
        start_time = time.time()

        # Load and convert the image to LAB color space
        image = io.imread(image_path)
        image_lab = color.rgb2lab(image)  # Convert to LAB color space for better color clustering

        # Reshape the image to a 2D array of LAB color and XY position features
        h, w, c = image_lab.shape
        X = np.zeros((h * w, 5))  # Array to hold LAB and XY features
        X[:, :3] = image_lab.reshape(-1, 3)  # LAB color features
        X[:, 3] = np.repeat(np.arange(h), w)  # X coordinates
        X[:, 4] = np.tile(np.arange(w), h)    # Y coordinates

        # Estimate the bandwidth for Mean Shift based on data spread
        bandwidth = estimate_bandwidth(X, quantile=0.09, n_samples=100)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        # Apply Mean Shift clustering
        ms.fit(X)
        labels = ms.labels_.reshape(h, w)  # Reshape labels back to image dimensions

        # Create an output image where each cluster is colored by its mean color
        segmented_image = np.zeros_like(image_lab)
        for label in np.unique(labels):
            mask = (labels == label)
            segmented_image[mask] = np.mean(image_lab[mask], axis=0)

        # Convert segmented image back to RGB for visualization
        segmented_image_rgb = color.lab2rgb(segmented_image)

        # Save the segmented result to output folder
        output_path = os.path.join(output_folder, f"segmented_{image_name}")
        io.imsave(output_path, (segmented_image_rgb * 255).astype(np.uint8))

        # End the timer and compute processing time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processing time for {image_name}: {processing_time:.2f} seconds")

        # Optionally display each segmented image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(mark_boundaries(image, labels))
        ax[0].set_title("Mean Shift Segmentation Boundaries")
        ax[0].axis("off")

        ax[1].imshow(segmented_image_rgb)
        ax[1].set_title("Segmented Image (Mean Colors)")
        ax[1].axis("off")

        plt.show()

        print(f"Segmented image saved as {output_path}")

print("Processing completed for all images.")
