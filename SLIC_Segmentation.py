import torch
import numpy as np
from skimage import io, color
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import time

# Load and preprocess the image
image_path = '/home/rehanfarooq/cv/images/chair.jpg'
print(f"Processing image: {image_path}")

# Start timing
start_time = time.time()

image = io.imread(image_path)  # Load an image
image = color.rgb2lab(image)  # Convert to LAB color space

# Convert to PyTorch tensor and normalize
image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

# Apply SLIC algorithm
num_segments = 100  # Desired number of superpixels
compactness = 40  # Trade-off between color proximity and space proximity
segments = slic(image, n_segments=num_segments, compactness=compactness, start_label=0)

# Convert segments to PyTorch tensor for further processing
segments_tensor = torch.tensor(segments, dtype=torch.int32)

# Calculate processing time for SLIC segmentation
end_time_slic = time.time()
print(f"SLIC segmentation processing time: {end_time_slic - start_time:.2f} seconds")

# Visualize the superpixels
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(mark_boundaries(image, segments))
ax[0].set_title('SLIC Superpixels')
ax[0].axis('off')

# Optionally, visualize each superpixel
superpixel_img = np.zeros_like(image)
for segment_id in np.unique(segments):
    mask = (segments == segment_id)
    superpixel_img[mask] = np.mean(image[mask], axis=0)  # Average color within each superpixel

# Calculate processing time for superpixel average color computation
end_time_superpixel = time.time()
print(f"Superpixel averaging processing time: {end_time_superpixel - end_time_slic:.2f} seconds")

# Convert back to RGB for display
ax[1].imshow(color.lab2rgb(superpixel_img))
ax[1].set_title('Superpixel Average Color')
ax[1].axis('off')

# Display total processing time
plt.show()
total_processing_time = end_time_superpixel - start_time
print(f"Total processing time: {total_processing_time:.2f} seconds")
