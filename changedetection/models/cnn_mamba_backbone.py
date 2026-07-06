import torch
import torch.nn as nn

from .Mamba_backbone import Backbone_VSSM


class ConvBNAct(nn.Sequential):
    """Conv-BN-GELU block used by the CNN part."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        if padding is None:
            padding = kernel_size // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class DepthwiseSeparableConvBlock(nn.Module):
    """Lightweight CNN block for high-resolution stages."""

    def __init__(self, channels, mlp_ratio=2.0, drop=0.0):
        super().__init__()

        hidden_channels = int(channels * mlp_ratio)

        self.dwconv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.pwconv = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.act(x + identity)
        return x


class Backbone_CNNMambaStage4(nn.Module):
    """
    Hybrid backbone for ChangeMamba SCD.

    stage1-stage3: CNN
    stage4: original VSSM/Mamba blocks

    Output feature dimensions are kept the same as VSSM:
        [C, 2C, 4C, 8C]
    so the original ChangeDecoder and SemanticDecoder can be reused.
    """

    def __init__(
        self,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=(2, 2, 9, 2),
        dims=96,
        cnn_depths=(2, 2, 2),
        cnn_mlp_ratio=2.0,
        cnn_drop=0.0,
        norm_layer="ln",
        **vssm_kwargs,
    ):
        super().__init__()

        if patch_size != 4:
            raise ValueError("Backbone_CNNMambaStage4 currently supports patch_size=4 only.")

        if norm_layer.lower() != "ln":
            raise ValueError(
                "Backbone_CNNMambaStage4 currently assumes MODEL.VSSM.NORM_LAYER='ln'."
            )

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(len(depths))]
        else:
            dims = list(dims)

        if len(dims) != 4:
            raise ValueError(f"Expected 4-stage dims, got {dims}.")

        if len(cnn_depths) != 3:
            raise ValueError(f"cnn_depths should contain 3 values, got {cnn_depths}.")

        self.out_indices = tuple(out_indices)
        self.dims = dims
        self.num_layers = 4
        self.num_features = dims[-1]

        # Keep consistent with original Backbone_VSSM when norm_layer='ln':
        # encoder outputs are NCHW tensors for decoder usage.
        self.channel_first = False

        stem_hidden = max(dims[0] // 2, 16)

        self.stem = nn.Sequential(
            ConvBNAct(in_chans, stem_hidden, kernel_size=3, stride=2),
            ConvBNAct(stem_hidden, dims[0], kernel_size=3, stride=2),
        )

        self.stage1 = self._make_cnn_stage(dims[0], cnn_depths[0], cnn_mlp_ratio, cnn_drop)
        self.down1 = ConvBNAct(dims[0], dims[1], kernel_size=3, stride=2)

        self.stage2 = self._make_cnn_stage(dims[1], cnn_depths[1], cnn_mlp_ratio, cnn_drop)
        self.down2 = ConvBNAct(dims[1], dims[2], kernel_size=3, stride=2)

        self.stage3 = self._make_cnn_stage(dims[2], cnn_depths[2], cnn_mlp_ratio, cnn_drop)
        self.down3 = ConvBNAct(dims[2], dims[3], kernel_size=3, stride=2)

        # Reuse original VSSM stage4 and its outnorm3.
        vssm = Backbone_VSSM(
            out_indices=(3,),
            pretrained=pretrained,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            norm_layer=norm_layer,
            **vssm_kwargs,
        )

        self.stage4 = vssm.layers[3]
        self.outnorm3 = getattr(vssm, "outnorm3")

        self._init_cnn_weights()

    @staticmethod
    def _make_cnn_stage(channels, depth, mlp_ratio, drop):
        return nn.Sequential(
            *[
                DepthwiseSeparableConvBlock(
                    channels=channels,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                )
                for _ in range(depth)
            ]
        )

    def _init_cnn_weights(self):
        modules = [
            self.stem,
            self.stage1,
            self.down1,
            self.stage2,
            self.down2,
            self.stage3,
            self.down3,
        ]

        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1.0)
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        outs = []

        # stage1: CNN, stride 4
        x = self.stem(x)
        x = self.stage1(x)
        if 0 in self.out_indices:
            outs.append(x)

        # stage2: CNN, stride 8
        x = self.down1(x)
        x = self.stage2(x)
        if 1 in self.out_indices:
            outs.append(x)

        # stage3: CNN, stride 16
        x = self.down2(x)
        x = self.stage3(x)
        if 2 in self.out_indices:
            outs.append(x)

        # stage4: VSSM/Mamba, stride 32
        x = self.down3(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.stage4.blocks(x)

        if 3 in self.out_indices:
            out = self.outnorm3(x)
            out = out.permute(0, 3, 1, 2).contiguous()
            outs.append(out)

        return outs
