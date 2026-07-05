import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import argparse

import torch

from changedetection.configs.config import get_config
from changedetection.script.script_utils import get_cgssg_kwargs, get_vssm_kwargs, load_checkpoint
from changedetection.models.ChangeMambaSCD import ChangeMambaSCD


def main():
    parser = argparse.ArgumentParser(description="Profile MambaSCD inference cost")
    parser.add_argument("--cfg", type=str, default="changedetection/configs/vssm1/vssm_small_224.yaml")
    parser.add_argument("--encoder_pretrained_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--opts", default=None, nargs="+")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    config = get_config(args)
    model = ChangeMambaSCD(
        output_cd=2,
        output_clf=7,
        pretrained=args.encoder_pretrained_path,
        **get_vssm_kwargs(config),
        **get_cgssg_kwargs(config),
    ).to(args.device)
    if args.model_checkpoint_path:
        load_checkpoint(model, args.model_checkpoint_path)
    model.eval()

    inputs = (
        torch.randn(args.batch_size, 3, args.crop_size, args.crop_size, device=args.device),
        torch.randn(args.batch_size, 3, args.crop_size, args.crop_size, device=args.device),
    )
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    flops = None
    try:
        from thop import profile

        flops, _ = profile(model, inputs=inputs, verbose=False)
    except Exception as exc:
        print(f"FLOPs: unavailable ({exc})")

    with torch.no_grad():
        for _ in range(args.warmup):
            model(*inputs)
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.iters):
            model(*inputs)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / args.iters
    fps = args.batch_size * args.iters / elapsed
    print(f"Params: {params / 1e6:.3f} M")
    print(f"Trainable params: {trainable_params / 1e6:.3f} M")
    print(f"Model size: {model_size_mb:.2f} MB")
    if flops is not None:
        print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Latency: {latency_ms:.2f} ms/image-batch")
    print(f"FPS: {fps:.2f}")
    if args.device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak memory: {memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
