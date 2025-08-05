import cv2
import matplotlib
import matplotlib.pyplot as plt

# Get the colormap in a version-safe way
try:
    # Newer Matplotlib
    PAIRED_COLORS = matplotlib.colormaps['Paired'].resampled(12).colors
except AttributeError:
    # Older Matplotlib
    PAIRED_COLORS = plt.get_cmap('Paired', 12).colors

# Uncomment below if your dataset has more than 12 classes
# TAB20_COLORS = plt.cm.get_cmap('tab20', 20).colors  # Supports up to 20 classes
# Then change the reference from PAIRED_COLORS to TAB20_COLORS in get_color()

def get_color(cls_id):
    """
    Returns a consistent BGR color tuple for a given class index using the chosen color palette.

    Args:
        cls_id (int): The class ID to assign a color to.

    Returns:
        tuple: BGR color tuple usable by OpenCV drawing functions.
    """
    color = PAIRED_COLORS[int(cls_id) % len(PAIRED_COLORS)]
    return tuple(int(255 * c) for c in color[:3])  # Convert RGB to BGR


def draw_boxes(image, bboxes):
    """
    Draws bounding boxes with class labels on the given image.

    Args:
        image (np.ndarray): RGB image to draw on.
        bboxes (List[List[float]]): List of bounding boxes with format [x_min, y_min, x_max, y_max, class_id].

    Returns:
        np.ndarray: Image with drawn boxes and labels.
    """
    img = image.copy()
    for box in bboxes:
        x_min, y_min, x_max, y_max, cls = box
        color = get_color(cls)
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(img, str(cls), (int(x_min), int(y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def show_before_after(original, augmented, original_boxes, augmented_boxes):
    """
    Displays a side-by-side comparison of an original and augmented image, both with bounding boxes.

    Args:
        original (np.ndarray): Original input image (RGB).
        augmented (np.ndarray): Augmented image (RGB).
        original_boxes (List[List[float]]): Boxes for the original image [x_min, y_min, x_max, y_max, class_id].
        augmented_boxes (List[List[float]]): Boxes for the augmented image [x_min, y_min, x_max, y_max, class_id].
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(draw_boxes(original, original_boxes))
    ax[0].set_title("Original")
    ax[0].axis('off')

    ax[1].imshow(draw_boxes(augmented, augmented_boxes))
    ax[1].set_title("Augmented")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


def yolo_to_xyxy(label_path, img_width, img_height):
    """
    Convert YOLO labels to [x_min, y_min, x_max, y_max, class_id] format.

    Args:
        label_path (str): Path to YOLO label file.
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.

    Returns:
        List[List[float]]: Bounding boxes in xyxy format.
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            cls, x_c, y_c, w, h = map(float, line.strip().split())
            x_min = (x_c - w / 2) * img_width
            y_min = (y_c - h / 2) * img_height
            x_max = (x_c + w / 2) * img_width
            y_max = (y_c + h / 2) * img_height
            boxes.append([x_min, y_min, x_max, y_max, int(cls)])
    return boxes

