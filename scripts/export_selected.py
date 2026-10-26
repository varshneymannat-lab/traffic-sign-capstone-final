from __future__ import annotations

import argparse
import json
import sys
import time
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.sign_datamodule import TrafficSignDataModule
from src.utils.video_pipeline import load_classifier


def make_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def benchmark(session, size):
    example = np.zeros((1, 3, size, size), dtype=np.float32)
    for _ in range(10):
        session.run(None, {"sign_crop": example})
    measurements = []
    for _ in range(50):
        start = time.perf_counter()
        session.run(None, {"sign_crop": example})
        measurements.append(1000 * (time.perf_counter() - start))
    return {"mean_ms": float(np.mean(measurements)), "median_ms": float(np.median(measurements))}


def main():
    parser = argparse.ArgumentParser(description="Export the validation-selected classifier and verify real ONNX inference.")
    parser.add_argument("--comparison-dir", type=Path, default=Path("outputs/gtsrb_comparison"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/gtsrb"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/gtsrb"))
    parser.add_argument("--report-dir", type=Path, default=Path("report/gtsrb_results"))
    args = parser.parse_args()
    comparison = json.loads((args.comparison_dir / "comparison.json").read_text())
    selected = comparison["selected_architecture"]
    settings = comparison["settings"]
    size = settings["image_size"]
    torch.set_num_threads(2)
    checkpoint = args.comparison_dir / selected / "models/checkpoints/best.ckpt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    classes = list(state["hyper_parameters"]["sign_classes"])
    model = load_classifier(checkpoint, classes, "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state["state_dict"], "hyper_parameters": state["hyper_parameters"]}, args.output_dir / "classifier.ckpt")
    destination = args.output_dir / "classifier.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, size, size), str(destination), dynamo=False,
                      input_names=["sign_crop"], output_names=["sign_logits"],
                      dynamic_axes={"sign_crop": {0: "batch"}, "sign_logits": {0: "batch"}}, opset_version=17)
    exported = onnx.load(str(destination))
    onnx.checker.check_model(exported)
    metadata = {"architecture": selected, "classes": classes, "image_size": size,
                "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225],
                "selection": settings["selection_rule"], "scope": "Cropped GTSRB sign classifier, not an object detector"}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    quantized = args.output_dir / "classifier_dense_int8.onnx"
    quantize_dynamic(str(destination), str(quantized), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    sessions = {"float32": make_session(destination), "dense_int8": make_session(quantized)}
    data = TrafficSignDataModule(str(args.data_dir), str(args.data_dir / "processed/annotations_split.csv"), size, 32, 0, classes)
    data.setup()
    actual, predictions = [], {name: [] for name in sessions}
    maximum_error = 0.0
    for index, batch in enumerate(data.test_dataloader()):
        values = batch["image"].numpy()
        actual.extend(batch["sign_label"].tolist())
        for name, session in sessions.items():
            output = session.run(None, {"sign_crop": values})[0]
            predictions[name].extend(output.argmax(1).tolist())
            if name == "float32" and index == 0:
                with torch.inference_mode():
                    reference = model(batch["image"]).numpy()
                maximum_error = float(np.max(np.abs(reference - output)))
                np.testing.assert_allclose(reference, output, atol=1e-4, rtol=1e-3)
    results = {name: {"test_samples": len(actual), "accuracy": accuracy_score(actual, predicted),
                      "macro_f1": f1_score(actual, predicted, average="macro"),
                      "bytes": (destination if name == "float32" else quantized).stat().st_size,
                      "forward_latency": benchmark(sessions[name], size)} for name, predicted in predictions.items()}
    report = {"architecture": selected, "first_test_batch_max_abs_logit_difference": maximum_error,
              "sessions": results, "runtime": ort.__version__, "threads": 2,
              "quantization_scope": "Dynamic INT8 of dense MatMul/Gemm weights only; convolution layers remain float32. Not full-model INT8.",
              "latency_scope": "CPU ONNX Runtime single-image forward only, 10 warm-ups and 50 repetitions; excludes loading and preprocessing."}
    report["software"] = {"python": platform.python_version(), **{name: version(name) for name in ["torch", "torchvision", "lightning", "scikit-learn", "onnx", "onnxruntime"]}}
    (args.report_dir / "onnx_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
