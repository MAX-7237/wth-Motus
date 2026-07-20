#!/usr/bin/env python3
"""Generate task-level WAN T5 caches for local LeRobot v3 datasets."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import torch


def _iter_dataset_roots(root: Path) -> Iterable[Path]:
    if (root / "meta" / "info.json").is_file():
        yield root
        return
    for info_path in sorted(root.glob("**/meta/info.json")):
        ds_root = info_path.parent.parent
        if (ds_root / "meta" / "tasks.parquet").is_file():
            yield ds_root


def _task_texts(dataset_root: Path) -> list[str]:
    table = pq.read_table(dataset_root / "meta" / "tasks.parquet")
    data = table.to_pydict()
    text_col = "task" if "task" in data else "__index_level_0__"
    return sorted({str(x) for x in data.get(text_col, []) if str(x).strip()})


def _init_wan_t5_encoder(wan_path: str, device: str, text_len: int) -> Any:
    try:
        from bak.wan.modules.t5 import T5EncoderModel  # type: ignore
    except Exception:
        bak_root = str((Path(__file__).resolve().parents[2] / "bak").resolve())
        if bak_root not in sys.path:
            sys.path.insert(0, bak_root)
        from wan.modules.t5 import T5EncoderModel  # type: ignore

    ckpt = os.path.join(wan_path, "Wan2.2-TI2V-5B", "models_t5_umt5-xxl-enc-bf16.pth")
    tok = os.path.join(wan_path, "Wan2.2-TI2V-5B", "google/umt5-xxl")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return T5EncoderModel(
        text_len=int(text_len),
        dtype=dtype,
        device=device,
        checkpoint_path=ckpt,
        tokenizer_path=tok,
    )


def _encode_t5(encoder: Any, instruction: str, device: str) -> torch.Tensor:
    with torch.no_grad():
        out = encoder([instruction], device)
    emb = out[0] if isinstance(out, list) else out
    if isinstance(emb, torch.Tensor) and emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb.squeeze(0)
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb)
    return emb.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root or directory containing task dataset roots")
    parser.add_argument("--wan_path", default=os.environ.get("WAN_PATH") or os.environ.get("WAN_ROOT") or "/mnt/gyc_ckp")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text_len", type=int, default=512)
    parser.add_argument("--t5_folder_name", default="t5_embedding")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    roots = list(_iter_dataset_roots(Path(args.root)))
    if not roots:
        raise FileNotFoundError(f"No LeRobot v3 dataset roots found under {args.root}")

    instructions = []
    for ds_root in roots:
        for text in _task_texts(ds_root):
            instructions.append((ds_root, text))

    encoder = None
    updated = 0
    skipped = 0
    for ds_root, text in instructions:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        out_path = ds_root / args.t5_folder_name / f"task_{digest}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if encoder is None:
            print(f"Loading WAN T5 encoder from {args.wan_path} on {args.device} ...")
            encoder = _init_wan_t5_encoder(args.wan_path, args.device, args.text_len)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_encode_t5(encoder, text, args.device), out_path)
        updated += 1
        if updated % 10 == 0:
            print(f"updated={updated} skipped={skipped}")

    print(f"Done. dataset_roots={len(roots)} updated={updated} skipped={skipped}")


if __name__ == "__main__":
    main()
