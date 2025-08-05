from draw import *
import cv2
import matplotlib.pyplot as plt
import argparse
import os

# ====== CLI utility to visualize a YOLO-labeled image from the command line ======

# Run from terminal: python visualize_yolo.py --image path/to/img.jpg --label path/to/labels.txt
# This will open a Matplotlib window showing the bounding boxes.

def visualize_yolo(image_path, label_path):
    """
    Load an image and its YOLO label file, draw bounding boxes using existing draw_boxes(),
    and display the result in a Matplotlib window.

    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the YOLO-format label file (.txt).
    """
    # Check if the provided image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check if the provided label file exists
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    # Read the image using OpenCV (BGR) and convert to RGB for Matplotlib display
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Get the image dimensions
    h, w = img.shape[:2]

    # Convert YOLO normalized labels to absolute pixel coordinates
    boxes = yolo_to_xyxy(label_path, w, h)

    # Draw bounding boxes on the image
    img_with_boxes = draw_boxes(img, boxes)

    # Display the image with bounding boxes using Matplotlib
    plt.imshow(img_with_boxes)
    plt.axis("off")  # Hide axes/ticks
    plt.title(os.path.basename(image_path))  # Use the filename as the plot title
    plt.show()  # Render the image window


if __name__ == "__main__":
    # Create a CLI argument parser
    parser = argparse.ArgumentParser(description="Visualize YOLO-labeled image.")

    # Required argument: Path to the image file
    parser.add_argument("--image", type=str, required=True, help="Path to image file")

    # Required argument: Path to the YOLO label file
    parser.add_argument("--label", type=str, required=True, help="Path to YOLO label file")

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the function to visualize the labeled image
    visualize_yolo(args.image, args.label)
