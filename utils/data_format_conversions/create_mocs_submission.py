def create_mocs_submission(results, output_dir="submission", filename="answer.json"):
    """
    Converts YOLOv8 detection results into COCO-style submission format for MOCS.

    Args:
        results (list): List of ultralytics Result objects from model.predict().
        output_dir (str): Folder to save the final answer.json.
        filename (str): Name of the output file (must be "answer.json" for CodaLab).
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    submission = []

    for image_id, result in enumerate(results):
        if not hasattr(result, "boxes") or result.boxes is None:
            continue

        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            conf = float(result.boxes.conf[i].item())
            x_min, y_min, x_max, y_max = result.boxes.xyxy[i].tolist()
            width = x_max - x_min
            height = y_max - y_min

            submission.append({
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [round(x_min, 1), round(y_min, 1), round(width, 1), round(height, 1)],
                "score": round(conf, 4)
            })

    with open(output_path, "w") as f:
        json.dump(submission, f)
    
    print(f"Submission saved to: {output_path}")




def zip_submission(json_path):
    import zipfile
    import os
    zip_path = os.path.join(os.path.dirname(json_path), "submission.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="answer.json")
    print(f"Zipped submission saved to: {zip_path}")
