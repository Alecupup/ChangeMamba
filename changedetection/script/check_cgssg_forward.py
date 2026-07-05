import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import argparse

import torch

from changedetection.configs.config import get_config
from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
from changedetection.script.script_utils import get_cgssg_kwargs, get_vssm_kwargs


def main():
    parser = argparse.ArgumentParser(description="Check MambaSCD + CGSSG forward shapes")
    parser.add_argument("--cfg", type=str, default="changedetection/configs/vssm1/vssm_small_224_cgssg_stage4.yaml")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--opts", default=None, nargs="+")
    parser.add_argument("--encoder_pretrained_path", type=str, default=None)
    args = parser.parse_args()

    config = get_config(args)
    model = ChangeMambaSCD(
        output_cd=2,
        output_clf=7,
        pretrained=args.encoder_pretrained_path,
        **get_vssm_kwargs(config),
        **get_cgssg_kwargs(config),
    )
    model.eval()
    pre = torch.randn(args.batch_size, 3, args.crop_size, args.crop_size)
    post = torch.randn_like(pre)
    with torch.no_grad():
        output_cd, output_t1, output_t2 = model(pre, post)

    print("output_cd:", tuple(output_cd.shape))
    print("output_t1:", tuple(output_t1.shape))
    print("output_t2:", tuple(output_t2.shape))
    for stage, mask in model.last_cgssg_masks.items():
        print(f"cgssg_mask_stage{stage}:", tuple(mask.shape))


if __name__ == "__main__":
    main()
