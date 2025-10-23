#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch

def main():
    ap = argparse.ArgumentParser(description="Download DeepLabv3_ResNet50 weights locally")
    ap.add_argument("--outdir", default="baseline_deeplabv3_resnet50",
                    help="output directory for model files")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Download via torchvision weights API (goes to outdir/torch/hub/checkpoints)
    #    by temporarily setting TORCH_HOME
    import os
    os.environ["TORCH_HOME"] = str(outdir)

    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).eval()

    # 2) Save a plain state_dict for fully offline loading (no cache needed)
    sd_path = outdir / "deeplabv3_resnet50_state_dict.pt"
    torch.save(model.state_dict(), sd_path)

    # 3) Save a TorchScript copy (fast to load, portable)
    ts_path = outdir / "deeplabv3_resnet50.torchscript.pt"
    scripted = torch.jit.script(model)
    scripted.save(str(ts_path))

    # 4) Minimal metadata
    meta = {
        "weights_enum": str(weights),
        "transforms_note": "Use torchvision weights.transforms() at inference",
        "num_params": sum(p.numel() for p in model.parameters())
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("✅ Downloaded & saved:")
    print(" - cache dir (weights):", outdir / "torch" / "hub" / "checkpoints")
    print(" - state_dict:", sd_path)
    print(" - torchscript:", ts_path)

if __name__ == "__main__":
    main()

