#!/usr/bin/env python3
"""Convert Arc2Face ArcFace ONNX model to PyTorch via onnx2torch.

After conversion, runs a numerical equivalence check on random inputs
to ensure the PyTorch model produces the same outputs as the ONNX model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx2torch import convert

ONNX_PATH = Path(__file__).resolve().parent / "pretrained" / "arcface.onnx"
PTH_PATH = Path(__file__).resolve().parent / "pretrained" / "arc2face_arcface_r100_model.pth"


def convert_and_verify(
    onnx_path: Path = ONNX_PATH,
    pth_path: Path = PTH_PATH,
    n_tests: int = 10,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    print(f"Loading ONNX model from {onnx_path} …")
    onnx_model = onnx.load(str(onnx_path))

    # --- ONNX Runtime reference session ---
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape          # e.g. [batch, 3, 112, 112]
    output_name = sess.get_outputs()[0].name
    print(f"ONNX input: name={input_name}, shape={input_shape}")
    print(f"ONNX output: name={output_name}")

    # --- Convert to PyTorch ---
    print("Converting with onnx2torch …")
    torch_model = convert(onnx_model)
    torch_model.eval()
    print(f"Conversion done — {sum(p.numel() for p in torch_model.parameters()):,} parameters")

    # --- Numerical equivalence ---
    print(f"\nRunning {n_tests} equivalence checks (atol={atol}, rtol={rtol}) …")
    max_abs_err = 0.0
    max_rel_err = 0.0

    for i in range(n_tests):
        # Random BGR input in [-1, 1]
        np_input = np.random.uniform(-1, 1, size=(2, 3, 112, 112)).astype(np.float32)

        # ONNX Runtime
        onnx_out = sess.run([output_name], {input_name: np_input})[0]

        # PyTorch
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(np_input)).numpy()

        abs_err = np.abs(onnx_out - torch_out).max()
        denom = np.abs(onnx_out).max()
        rel_err = abs_err / denom if denom > 0 else 0.0
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)

        ok = np.allclose(onnx_out, torch_out, atol=atol, rtol=rtol)
        status = "PASS" if ok else "FAIL"
        print(f"  test {i + 1}/{n_tests}: abs_err={abs_err:.2e}  rel_err={rel_err:.2e}  [{status}]")

        if not ok:
            raise RuntimeError(
                f"Equivalence check FAILED on test {i + 1}: "
                f"max_abs_err={abs_err:.2e}, max_rel_err={rel_err:.2e}"
            )

    print(f"\nAll {n_tests} tests passed!")
    print(f"  worst-case abs error: {max_abs_err:.2e}")
    print(f"  worst-case rel error: {max_rel_err:.2e}")

    # --- Print named modules (useful for finding tap points) ---
    print("\n--- Named modules (for perceptual loss tap points) ---")
    for name, mod in torch_model.named_modules():
        if name:
            print(f"  {name}  ->  {mod.__class__.__name__}")

    # --- Save ---
    print(f"\nSaving PyTorch model to {pth_path} …")
    torch.save(torch_model, pth_path)
    print(f"Done — {pth_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    convert_and_verify()
