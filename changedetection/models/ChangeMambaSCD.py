import torch.nn as nn

from .ChangeDecoder import ChangeDecoder
from .SemanticDecoder import SemanticDecoder
from .builders import build_backbone, build_head, resolve_decoder_components, resize_to_input
from .cgssg import ChangeGuidedSparseSpatialGate


class ChangeMambaSCD(nn.Module):
    def __init__(
        self,
        output_cd,
        output_clf,
        pretrained,
        use_cgssg=False,
        cgssg_stages=None,
        cgssg_reduction=4,
        cgssg_alpha_init=0.5,
        cgssg_learnable_alpha=True,
        cgssg_return_mask=True,
        **kwargs,
    ):
        super().__init__()
        self.encoder = build_backbone(pretrained=pretrained, **kwargs)
        norm_layer, ssm_act_layer, mlp_act_layer, clean_kwargs = resolve_decoder_components(kwargs)
        self.use_cgssg = use_cgssg
        self.cgssg_return_mask = cgssg_return_mask
        self.last_cgssg_masks = {}

        if self.use_cgssg:
            self.cgssg_stage_indices = self._normalize_cgssg_stages(cgssg_stages, len(self.encoder.dims))
            self.cgssg_gates = nn.ModuleDict(
                {
                    str(stage_idx): ChangeGuidedSparseSpatialGate(
                        channels=self.encoder.dims[stage_idx],
                        reduction=cgssg_reduction,
                        alpha_init=cgssg_alpha_init,
                        learnable_alpha=cgssg_learnable_alpha,
                    )
                    for stage_idx in self.cgssg_stage_indices
                }
            )
        else:
            self.cgssg_stage_indices = []
            self.cgssg_gates = nn.ModuleDict()

        self.decoder_bcd = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.decoder_T1 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )
        self.decoder_T2 = SemanticDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs,
        )

        self.main_clf_cd = build_head(out_channels=output_cd)
        self.aux_clf = build_head(out_channels=output_clf)

    @staticmethod
    def _normalize_cgssg_stages(stages, num_stages):
        if stages is None:
            stages = [num_stages]
        if isinstance(stages, int):
            stages = [stages]

        # Public config uses 1-based stage IDs: stage4 is the deepest feature.
        # Keep 0-based lists working for callers that explicitly include 0.
        zero_based = any(stage == 0 for stage in stages)
        stage_indices = []
        for stage in stages:
            stage_idx = stage if zero_based else stage - 1
            if stage_idx < 0 or stage_idx >= num_stages:
                raise ValueError(f"CGSSG stage {stage} is out of range for {num_stages} encoder stages.")
            stage_indices.append(stage_idx)
        return sorted(set(stage_indices))

    def _apply_cgssg(self, pre_features, post_features):
        if not self.use_cgssg:
            self.last_cgssg_masks = {}
            return pre_features, post_features

        pre_features = list(pre_features)
        post_features = list(post_features)
        masks = {}
        for stage_idx in self.cgssg_stage_indices:
            gated_pre, gated_post, mask = self.cgssg_gates[str(stage_idx)](
                pre_features[stage_idx],
                post_features[stage_idx],
            )
            pre_features[stage_idx] = gated_pre
            post_features[stage_idx] = gated_post
            if self.cgssg_return_mask:
                masks[stage_idx + 1] = mask
        self.last_cgssg_masks = masks
        return pre_features, post_features

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        pre_features, post_features = self._apply_cgssg(pre_features, post_features)

        output_bcd = resize_to_input(self.main_clf_cd(self.decoder_bcd(pre_features, post_features)), pre_data)
        output_T1 = resize_to_input(self.aux_clf(self.decoder_T1(pre_features)), pre_data)
        output_T2 = resize_to_input(self.aux_clf(self.decoder_T2(post_features)), post_data)
        return output_bcd, output_T1, output_T2
