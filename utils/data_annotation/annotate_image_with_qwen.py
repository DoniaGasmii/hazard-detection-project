from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load Qwen model and processor once
QWEN_MODEL_ID = "Qwen/Qwen-VL-Max"
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
qwen_model = AutoModelForVision2Seq.from_pretrained(
    QWEN_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16
)

def extract_json_from_output(text: str):
    start = text.find("[")
    end = text.rfind("]") + 1
    try:
        return json.loads(text[start:end])
    except:
        return []

def draw_bboxes(img, bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)
    for obj in bboxes:
        label = obj.get("label", "unknown")
        x1, y1, x2, y2 = obj["bbox_2d"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, color="red", fontsize=12, weight='bold')
    plt.axis('off')
    plt.title("Qwen-VL Annotations")
    plt.show()

def annotate_image_with_qwen(image_path, msgs, verbose=False):
    """
    Annotate a single image using Qwen-VL based on a message prompt.

    Args:
        image_path (str): Path to the input image.
        msgs (list): The Qwen-VL prompt structure (system + user).
        verbose (bool): Whether to plot the image with predicted boxes.

    Returns:
        list: A list of dicts with keys 'label' and 'bbox_2d' for each detection.
    """
    img = Image.open(image_path).convert("RGB")
    
    # Insert actual image object into msgs
    for msg in msgs:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content["type"] == "image":
                    content["image"] = img

    prompt = qwen_processor.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = qwen_processor(
        text=[prompt],
        images=[img],
        return_tensors="pt"
    ).to(qwen_model.device)

    with torch.no_grad():
        out_ids = qwen_model.generate(**inputs, max_new_tokens=1000)

    output = qwen_processor.batch_decode(
        out_ids[:, inputs.input_ids.shape[-1]:],
        skip_special_tokens=False
    )[0]

    bboxes = extract_json_from_output(output)

    if verbose:
        draw_bboxes(img, bboxes)

    return bboxes
