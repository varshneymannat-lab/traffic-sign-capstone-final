from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from torchvision import transforms


def main():
    parser = argparse.ArgumentParser(description="Classify a photograph of a single traffic sign using the exported ONNX model.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("models/mtsd_classifier"))
    parser.add_argument("--box", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    args = parser.parse_args()
    metadata = json.loads((args.model_dir / "metadata.json").read_text())
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        if args.box:
            x1, y1, x2, y2 = args.box
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                parser.error("The crop rectangle must be inside the image; maxima are exclusive.")
            image = image.crop(args.box)
        preprocess = transforms.Compose([
            transforms.Resize((metadata["image_size"], metadata["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(metadata["normalization_mean"], metadata["normalization_std"]),
        ])
        tensor = preprocess(image).unsqueeze(0).numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.model_dir / "classifier.onnx"), options, providers=["CPUExecutionProvider"])
    logits = session.run(None, {"sign_crop": tensor})[0][0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    index = int(probabilities.argmax())
    print(json.dumps({"image": str(args.image), "predicted_class": metadata["classes"][index],
                      "model_score": float(probabilities[index]),
                      "scope": "Five-class cropped-sign classification; no object detection or unknown-category rejection."}, indent=2))


if __name__ == "__main__":
    main()
