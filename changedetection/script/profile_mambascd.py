import os
import sys
import time
import argparse
import warnings
from typing import Dict, Optional, Tuple

import torch

# Make "changedetection" importable when this script is called from either:
#   1) ChangeMamba/
#   2) ChangeMamba/changedetection/
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from changedetection.configs.config import get_config
from changedetection.models.ChangeMambaSCD import ChangeMambaSCD


DEFAULT_CFG = os.path.abspath(
    os.path.join(CUR_DIR, "..", "configs", "vssm1", "vssm_small_224.yaml")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile MambaSCD Params(M), FLOPs(G), and Infer(s/100runs)."
    )

    parser.add_argument(
        "--cfg",
        type=str,
        default=DEFAULT_CFG,
        help="Path to VSSM config yaml."
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs="+",
        help="Modify config options by adding KEY VALUE pairs."
    )
    parser.add_argument(
        "--encoder_pretrained_path",
        type=str,
        default=None,
        help="Path to encoder/backbone pretrained weights. Can be omitted for profiling."
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=None,
        help="Optional full MambaSCD checkpoint path."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size used for profiling. For paper tables, usually use 1."
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=512,
        help="Input crop size. SECOND/MambaSCD usually uses 512."
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup forward passes before timing."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of timed forward passes."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device, e.g. cuda, cuda:0, or cpu."
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision for inference timing."
    )
    parser.add_argument(
        "--flop_tool",
        type=str,
        default="thop",
        choices=["thop", "fvcore", "none"],
        help="Tool used to calculate FLOPs."
    )
    parser.add_argument(
        "--output_cd",
        type=int,
        default=2,
        help="Number of binary change detection classes."
    )
    parser.add_argument(
        "--output_clf",
        type=int,
        default=7,
        help="Number of semantic classes for SECOND."
    )

    return parser.parse_args()


def get_vssm_kwargs(config) -> Dict:
    """Extract VSSM backbone parameters from config."""
    return dict(
        patch_size=config.MODEL.VSSM.PATCH_SIZE,
        in_chans=config.MODEL.VSSM.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=config.MODEL.VSSM.DEPTHS,
        dims=config.MODEL.VSSM.EMBED_DIM,
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=(
            "auto"
            if config.MODEL.VSSM.SSM_DT_RANK == "auto"
            else int(config.MODEL.VSSM.SSM_DT_RANK)
        ),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )


def get_cgssg_kwargs(config) -> Dict:
    """Extract optional CGSSG parameters from config."""
    if not hasattr(config.MODEL, "CGSSG"):
        return dict(
            use_cgssg=False,
            cgssg_stages=[4],
            cgssg_reduction=4,
            cgssg_alpha_init=0.5,
            cgssg_learnable_alpha=True,
            cgssg_return_mask=True,
        )

    return dict(
        use_cgssg=config.MODEL.CGSSG.ENABLED,
        cgssg_stages=list(config.MODEL.CGSSG.STAGES),
        cgssg_reduction=config.MODEL.CGSSG.REDUCTION,
        cgssg_alpha_init=config.MODEL.CGSSG.ALPHA_INIT,
        cgssg_learnable_alpha=config.MODEL.CGSSG.LEARNABLE_ALPHA,
        cgssg_return_mask=config.MODEL.CGSSG.RETURN_MASK,
    )


def normalize_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if str(path).lower() in {"", "none", "null"}:
        return None
    return path


def load_checkpoint_safely(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """Load common checkpoint formats without relying on project helper files."""
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model", "model_state_dict", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    state_dict = {}
    for key, value in checkpoint.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        state_dict[new_key] = value

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")


def build_model(args, device: torch.device) -> torch.nn.Module:
    config = get_config(args)

    model = ChangeMambaSCD(
        output_cd=args.output_cd,
        output_clf=args.output_clf,
        pretrained=normalize_path(args.encoder_pretrained_path),
        **get_vssm_kwargs(config),
        **get_cgssg_kwargs(config),
    )

    model.to(device)

    ckpt = normalize_path(args.model_checkpoint_path)
    if ckpt is not None:
        load_checkpoint_safely(model, ckpt, device)

    model.eval()
    return model


def count_params_m(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def make_inputs(args, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    t1 = torch.randn(
        args.batch_size,
        3,
        args.crop_size,
        args.crop_size,
        device=device,
    )
    t2 = torch.randn(
        args.batch_size,
        3,
        args.crop_size,
        args.crop_size,
        device=device,
    )
    return t1, t2


def calc_flops_thop(model, inputs) -> Optional[float]:
    try:
        from thop import profile
    except ImportError:
        print("[Warning] thop is not installed. Install with: pip install thop")
        return None

    try:
        # thop reports MACs for many layers. Many papers report this value as FLOPs(G).
        macs, _ = profile(model, inputs=inputs, verbose=False)
        return macs / 1e9
    except Exception as exc:
        print(f"[Warning] thop FLOPs calculation failed: {exc}")
        return None


def calc_flops_fvcore(model, inputs) -> Optional[float]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("[Warning] fvcore is not installed. Install with: pip install fvcore")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            analyzer = FlopCountAnalysis(model, inputs)
            flops = analyzer.total()

        unsupported = analyzer.unsupported_ops()
        if len(unsupported) > 0:
            print("[Warning] fvcore has unsupported ops. FLOPs may be underestimated.")
            for op_name, op_count in unsupported.items():
                print(f"  {op_name}: {op_count}")

        return flops / 1e9
    except Exception as exc:
        print(f"[Warning] fvcore FLOPs calculation failed: {exc}")
        return None


def calc_flops(model, inputs, tool: str) -> Optional[float]:
    if tool == "none":
        return None
    if tool == "thop":
        return calc_flops_thop(model, inputs)
    if tool == "fvcore":
        return calc_flops_fvcore(model, inputs)
    raise ValueError(f"Unsupported FLOP tool: {tool}")


def synchronize_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_infer_s_per_100runs(model, inputs, args, device: torch.device) -> float:
    t1, t2 = inputs

    for _ in range(args.warmup):
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            _ = model(t1, t2)

    synchronize_if_needed(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for _ in range(args.iters):
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            _ = model(t1, t2)

    synchronize_if_needed(device)

    elapsed = time.perf_counter() - start

    return elapsed * 100.0 / args.iters


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Use CPU instead.")
        args.device = "cpu"

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    model = build_model(args, device)
    inputs = make_inputs(args, device)

    params_m = count_params_m(model)

    with torch.no_grad():
        flops_g = calc_flops(model, inputs, args.flop_tool)

    infer_s_100 = benchmark_infer_s_per_100runs(
        model=model,
        inputs=inputs,
        args=args,
        device=device,
    )

    flops_text = "N/A" if flops_g is None else f"{flops_g:.3f}"

    print("\n" + "=" * 72)
    print("MambaSCD Profile Result")
    print("=" * 72)
    print(f"Config      : {args.cfg}")
    print(f"Input       : {args.batch_size} x 3 x {args.crop_size} x {args.crop_size}")
    print(f"Device      : {device}")
    print(f"AMP         : {args.amp}")
    print(f"Runs        : {args.iters}")
    print("-" * 72)
    print("Params(M)\tFLOPs(G)\tInfer(s/100runs)")
    print(f"{params_m:.3f}\t\t{flops_text}\t\t{infer_s_100:.4f}")
    print("=" * 72)

    if device.type == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak memory : {memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
