"""Plot test reconstruction (BCE and MSE) vs. updates for K-FAC and SGD.

Reads JSONL run logs from ``logs/`` and writes ``plots/recon_vs_steps.png``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def _load(tag: str) -> list[dict]:
    path = HERE / "logs" / f"{tag}.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kfac-tag", default="kfac_asdl_main")
    parser.add_argument("--sgd-tag", default="sgd_main")
    parser.add_argument("--out", default="plots/recon_vs_steps.png")
    args = parser.parse_args()

    kfac = _load(args.kfac_tag)
    sgd = _load(args.sgd_tag)

    kfac_steps = [r["step"] for r in kfac]
    kfac_bce = [r["test_loss"] for r in kfac]
    kfac_mse = [r["test_mse"] for r in kfac]

    sgd_steps = [r["step"] for r in sgd]
    sgd_bce = [r["test_loss"] for r in sgd]
    sgd_mse = [r["test_mse"] for r in sgd]

    fig, (ax_bce, ax_mse) = plt.subplots(1, 2, figsize=(12, 5))

    ax_bce.plot(kfac_steps, kfac_bce, label=f"K-FAC ({args.kfac_tag})", color="C0")
    ax_bce.plot(sgd_steps, sgd_bce, label=f"SGD-momentum ({args.sgd_tag})", color="C1")
    ax_bce.set_xlabel("update")
    ax_bce.set_ylabel("test BCE (sum/pixel, mean/example)")
    ax_bce.set_yscale("log")
    ax_bce.set_title("Test reconstruction (BCE) vs updates")
    ax_bce.grid(True, which="both", alpha=0.3)
    ax_bce.legend()

    ax_mse.plot(kfac_steps, kfac_mse, label=f"K-FAC ({args.kfac_tag})", color="C0")
    ax_mse.plot(sgd_steps, sgd_mse, label=f"SGD-momentum ({args.sgd_tag})", color="C1")
    ax_mse.set_xlabel("update")
    ax_mse.set_ylabel("test MSE (sum/example)")
    ax_mse.set_yscale("log")
    ax_mse.set_title("Test reconstruction (MSE) vs updates")
    ax_mse.grid(True, which="both", alpha=0.3)
    ax_mse.legend()

    fig.suptitle("Hinton & Salakhutdinov 2006 deep MNIST autoencoder")
    fig.tight_layout()

    out_path = HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
