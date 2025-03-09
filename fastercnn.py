import os
import time
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from collections import defaultdict

# Load a pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set to evaluation mode

# Directory containing images
image_dir = '/home/rehanfarooq/cv/images'  # Replace with your images folder path

# COCO dataset labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Ground truth data
ground_truths = {
    'bin.jpg': [
        {'box': [2210, 1940, 3860, 6000], 'label': 'trash bin'},
        
    ],
    'closeperson.jpg': [{'box': [1350, 3580, 2920, 8810], 'label': 'person'}],
    'laptop.jpg': [{'box': [90, 2730, 4900, 7150], 'label': 'laptop'},
                   {'box':[20,1440,3080,4520],'label':'tv'},{'box':[1120,710,3250,2150],'label':'person'},{'box':[3320,1970,4810,2960],'label':'person'}],
    'twocars.jpg': [{'box': [494, 375, 2100, 1544], 'label': 'car'},{'box': [1200, 154, 2505, 1139], 'label': 'car'},{'box': [2669, 23, 3264, 417], 'label': 'car'}],
    'bottleandmouse.jpg': [
        {'box': [690, 3930, 1840, 4790], 'label': 'mouse'},
        {'box': [1330, 3330, 5180, 5070], 'label': 'keyboard'},{'box': [2570, 70, 4000, 5190], 'label': 'bottle'},{'box': [1910, 830, 3370, 2010], 'label': 'tv'}
    ],
    'distantperson.jpg': [{'box': [1360, 5120, 1910, 5870], 'label': 'couch'},{'box': [1910, 4460, 2500, 5900], 'label': 'person'},{'box': [2720, 5270, 3050, 5850], 'label': 'bin'}],
    'my.jpg': [{'box': [271, 278, 1086, 1445], 'label': 'person'}],
    'boy.jpg': [{'box': [27, 31, 537, 403], 'label': 'pesron'},{'box': [182, 245, 458, 398], 'label': 'backpack'}],
    'distantshafay.jpg': [{'box': [2390, 3300, 3110, 5460], 'label': 'person'},{'box': [470, 3320, 920, 4100], 'label': 'person'},{'box': [860, 3350, 1430, 4430], 'label': 'person'}],
    'poster.jpg': [{'box': [1819, 4529, 2324, 5025], 'label': 'person'}],
    'chair.jpg': [{'box': [453, 1459, 2385, 4586], 'label': 'chair'}],
    'floortextture.jpg': [{'box': [0, 0, 1896, 1351], 'label': '__background__'}],
    'reflection.jpg': [{'box': [921, 1070, 1253, 1620], 'label': 'cell phone'},{'box': [76, 174, 1121, 2400], 'label': 'person'},{'box': [844, 358, 1535,2123], 'label': 'person'}]
}
# Function to calculate Intersection over Union (IoU)
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1, yi1, xi2, yi2 = max(x1, x1g), max(y1, y1g), min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Initialize counters for overall statistics
total_images = 0
total_ground_truths = 0
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
total_processing_time = 0

# Iterate over all images in the directory
for filename in os.listdir(image_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image file extensions
        total_images += 1
        image_path = os.path.join(image_dir, filename)  # Full path to the image
        print(f"Processing: {image_path}")
        
        # Start time measurement
        start_time = time.time()
        
        # Load the image and apply EXIF orientation
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)  # Automatically handle orientation
        
        # Transform image to tensor
        image_tensor = F.to_tensor(image)

        # Run the model on the sample image
        with torch.no_grad():
            prediction = model([image_tensor])

        # Stop time measurement
        time_taken = time.time() - start_time
        total_processing_time += time_taken

        # Retrieve ground truth boxes and labels
        image_ground_truths = ground_truths.get(filename, [])
        num_ground_truths = len(image_ground_truths)
        total_ground_truths += num_ground_truths
        true_positives = 0
        false_positives = 0
        matched_ground_truths = [False] * num_ground_truths  # Track which ground truths were matched

        # Filter predictions to keep only the highest-confidence boxes for each overlapping object
        filtered_boxes = []
        for i, box in enumerate(prediction[0]['boxes']):
            score = prediction[0]['scores'][i].item()
            if score > 0.75:
                label_idx = int(prediction[0]['labels'][i].item())
                label = COCO_INSTANCE_CATEGORY_NAMES[label_idx]

                # Check if this box overlaps significantly with any higher-confidence box of the same label
                keep = True
                for j, (fbox, flabel, fscore) in enumerate(filtered_boxes):
                    if flabel == label and iou(box.numpy(), fbox) > 0.5:
                        # If an overlapping box has a higher score, discard this one
                        if fscore > score:
                            keep = False
                            break
                        else:
                            # Otherwise, discard the older box and keep the current one
                            filtered_boxes[j] = (box.numpy(), label, score)
                            keep = False
                            break
                if keep:
                    filtered_boxes.append((box.numpy(), label, score))

        # Visualize predictions with labels and bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        # Draw ground truth boxes in green
        for i, gt in enumerate(image_ground_truths):
            gt_box = gt['box']
            gt_label = gt['label']
            # Draw the ground truth box
            plt.plot([gt_box[0], gt_box[2], gt_box[2], gt_box[0], gt_box[0]],
                     [gt_box[1], gt_box[1], gt_box[3], gt_box[3], gt_box[1]], color="green", linewidth=2)
            # Display the ground truth label
            plt.text(gt_box[0], gt_box[1], f"GT: {gt_label}", color="white",
                     backgroundcolor="green", fontsize=10)

        # Draw filtered predicted boxes and labels
        for box, label, score in filtered_boxes:
            plt.plot([box[0], box[2], box[2], box[0], box[0]],
                     [box[1], box[1], box[3], box[3], box[1]], color="red")
            plt.text(box[0], box[1], f"{label}: {score:.2f}", color="white",
                     backgroundcolor="red", fontsize=10)
            
            # Check if this prediction matches any ground truth
            match_found = False
            for j, gt in enumerate(image_ground_truths):
                if not matched_ground_truths[j] and \
                   label == gt['label'] and \
                   box[0] <= gt['box'][2] and box[2] >= gt['box'][0] and \
                   box[1] <= gt['box'][3] and box[3] >= gt['box'][1]:
                    true_positives += 1
                    matched_ground_truths[j] = True
                    match_found = True
                    break
                    
            if not match_found:
                false_positives += 1  # Count as false positive if no match found

        # Calculate false negatives (ground truths not matched)
        false_negatives = matched_ground_truths.count(False)
        
        # Update total counts
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        # Display stats for this image on the plot
        plt.axis("off")
        report_text = f"Image: {filename}\n" \
                      f"Time taken: {time_taken:.2f} seconds\n" \
                      f"Ground Truths: {num_ground_truths}, True Positives: {true_positives},\n" \
                      f"False Positives: {false_positives}, False Negatives: {false_negatives}"
        plt.figtext(0.5, 0.01, report_text, ha="center", fontsize=10, wrap=True)
        plt.show()

# Calculate overall statistics
precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
avg_time_per_image = total_processing_time / total_images if total_images > 0 else 0

# Display final report in a separate figure
plt.figure(figsize=(8, 6))
final_report_text = f"===== Final Report =====\n" \
                    f"Total Images: {total_images}\n" \
                    f"Total Ground Truths: {total_ground_truths}\n" \
                    f"Total True Positives: {total_true_positives}\n" \
                    f"Total False Positives: {total_false_positives}\n" \
                    f"Total False Negatives: {total_false_negatives}\n" \
                    f"Precision: {precision * 100:.2f}%\n" \
                    f"Recall: {recall * 100:.2f}%\n" \
                    f"F1 Score: {f1_score * 100:.2f}%\n" \
                    f"Average Processing Time per Image: {avg_time_per_image:.2f} seconds"
plt.text(0.5, 0.5, final_report_text, ha='center', va='center', fontsize=12)
plt.axis("off")
plt.show()
