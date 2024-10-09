import os
import shutil

# Step 1: Download and Extract the Data
output_dir = '../tiny-imagenet-200/val'


annotations_path = os.path.join(output_dir, 'val_annotations.txt')
images_dir = os.path.join(output_dir, 'images')

# Create a dictionary to store image filenames and their corresponding labels
image_labels = {}

with open(annotations_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        image_filename = parts[0]
        label = parts[1]
        image_labels[image_filename] = label

# Step 3: Organize the Data
organized_dir = 'organized_val_data'

# Create the organized directory if it doesn't exist
if not os.path.exists(organized_dir):
    os.makedirs(organized_dir)

# Create subdirectories for each label
for label in set(image_labels.values()):
    label_dir = os.path.join(organized_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Move images to their corresponding label directories
for image_filename, label in image_labels.items():
    src_path = os.path.join(images_dir, image_filename)
    dst_path = os.path.join(organized_dir, label, image_filename)
    shutil.move(src_path, dst_path)
