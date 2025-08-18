# utils/data_labeling/class_oracles/helmet_oracle.py

from .base_class_labeler import BaseClassLabeler

class HelmetOracle(BaseClassLabeler):
    def __init__(self):
        super().__init__(
            class_name="helmet",
            model_path="path/to/your/helmet/model/best.pt",   # update this
            class_id=0,  # ID for helmet in your unified YOLO class map
            conf_threshold=0.4
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Helmet Oracle Labeler")
    parser.add_argument("--images", type=str, required=True, help="Path to raw images folder")
    parser.add_argument("--out", type=str, required=True, help="Output folder for YOLO labels")
    args = parser.parse_args()

    oracle = HelmetOracle()
    oracle.run_inference(args.images, args.out)
