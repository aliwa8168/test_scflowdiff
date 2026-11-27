import math
from typing import Literal

import torch
import torch.nn as nn

from scflowdiff.core.layers import (
    Block,
    CrossAttentionBlock,
    FinalLayerDit,
    TimestepEmbedder,
    get_1d_sincos_pos_embed,
)


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer: int,
        n_inducing_points: int,
        n_embed: int,
        n_embed_latent: int,
        n_head: int,
        n_head_cross: int,
        dropout: float,
        bias: bool,
        multiple_of: int,
        layernorm_eps: float,
        norm_layer: str,
        positional_encoding: bool = False,
    ):
        super().__init__()

        self.latent_embedding = n_embed_latent
        self.latent_dim = n_inducing_points
        self.encoder_layers = nn.ModuleList([])
        self.pos_embed: torch.Tensor | None
        if positional_encoding:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, n_inducing_points, n_embed), requires_grad=False
            )
        else:
            self.pos_embed = None

        self.ca_layer = CrossAttentionBlock(
            n_embed=n_embed,
            n_inducing_points=n_inducing_points,
            n_head=n_head_cross,
            dropout=dropout,
            bias=bias,
            norm_layer=norm_layer,
            multiple_of=multiple_of,
            layernorm_eps=layernorm_eps,
        )

        for _ in range(n_layer):
            self.encoder_layers.append(
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                )
            )

        self.encoder_latent_input = nn.Sequential(
            nn.Linear(n_embed, n_embed_latent, bias=bias),
            nn.LayerNorm(n_embed_latent, eps=layernorm_eps, elementwise_affine=False),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.ca_layer(x, key_padding_mask=key_padding_mask)
        if isinstance(self.pos_embed, nn.Parameter):
            x = x + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        h = self.encoder_latent_input(x)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_embed: int,
        n_embed_latent: int,
        n_head: int,
        n_head_cross: int,
        n_layer: int,
        n_inducing_points: int,
        dropout: float,
        bias: bool,
        multiple_of: int,
        layernorm_eps: float,
        norm_layer: str,
        shared_embedding: bool,
        use_adaln: bool = False,
    ):
        super().__init__()

        self.gene_embedding = (
            nn.Embedding(n_genes + 1, n_embed) if not shared_embedding else nn.Identity()
        )
        self.decoder_layers = nn.ModuleList([])

        self.decoder_latent_input = nn.Sequential(
            nn.LayerNorm(n_embed_latent, eps=layernorm_eps, elementwise_affine=False),
            nn.Linear(n_embed_latent, n_embed, bias=bias),
        )

        for _ in range(n_layer):
            self.decoder_layers.append(
                Block(
                    n_embed=n_embed,
                    n_head=n_head,
                    dropout=dropout,
                    bias=bias,
                    norm_layer=norm_layer,
                    multiple_of=multiple_of,
                    layernorm_eps=layernorm_eps,
                    use_adaln=use_adaln,
                )
            )
        self.decoder_cross_attention = CrossAttentionBlock(
            n_embed=n_embed,
            n_inducing_points=0,
            n_head=n_head_cross,
            dropout=dropout,
            bias=bias,
            norm_layer=norm_layer,
            multiple_of=multiple_of,
            layernorm_eps=layernorm_eps,
            use_adaln=use_adaln,
        )

    def forward(
        self,
        x: torch.Tensor,
        genes: torch.Tensor,
        condition: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.decoder_latent_input(x)
        for layer in self.decoder_layers:
            x = layer(x, condition)
        q = self.gene_embedding(genes)
        output = self.decoder_cross_attention(x, q, condition)
        return output


class CrossAttentionBlock_cell(nn.Module):
    """Bidirectional cross-attention between RNA and ATAC latent channels."""

    def __init__(self, channels: int, feature_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.feature_dim = feature_dim

        self.q_rna = nn.Linear(1, self.feature_dim, bias=False)
        self.k_rna = nn.Linear(1, self.feature_dim, bias=False)
        self.v_rna = nn.Linear(1, self.feature_dim, bias=False)

        self.q_atac = nn.Linear(1, self.feature_dim, bias=False)
        self.k_atac = nn.Linear(1, self.feature_dim, bias=False)
        self.v_atac = nn.Linear(1, self.feature_dim, bias=False)

        self.trans_rna = nn.Linear(self.feature_dim, 1, bias=False)
        self.trans_atac = nn.Linear(self.feature_dim, 1, bias=False)

    def forward(
        self, video: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video = video.transpose(-1, -2).unsqueeze(-1)  # (B, M, D, 1)
        audio = audio.transpose(-1, -2).unsqueeze(-1)  # (B, M, D, 1)

        q_rna = self.q_rna(video)
        k_rna = self.k_rna(video)
        v_rna = self.v_rna(video)

        q_atac = self.q_atac(audio)
        k_atac = self.k_atac(audio)
        v_atac = self.v_atac(audio)

        scale = math.sqrt(self.feature_dim)
        att_rna_atac = torch.softmax(torch.matmul(q_rna, k_atac.transpose(-1, -2)) / scale, dim=-1)
        rna_new = torch.matmul(att_rna_atac, v_atac)

        att_atac_rna = torch.softmax(torch.matmul(q_atac, k_rna.transpose(-1, -2)) / scale, dim=-1)
        atac_new = torch.matmul(att_atac_rna, v_rna)

        video = video + self.trans_rna(rna_new)
        audio = audio + self.trans_atac(atac_new)

        video = video.squeeze(-1).transpose(-1, -2)
        audio = audio.squeeze(-1).transpose(-1, -2)

        return video, audio


class MultimodalDiT(nn.Module):
    """Two-stream RNA/ATAC DiT velocity model with cross-attention."""

    _DIRECTION_TO_ID = {"joint": 0, "rna_to_atac": 1, "atac_to_rna": 2}

    def __init__(
        self,
        n_embed: int,
        n_embed_input: int,
        n_layer: int,
        n_head: int,
        seq_len: int,
        dropout: float,
        bias: bool,
        norm_layer: str,
        multiple_of: int,
        layernorm_eps: float,
        class_vocab_sizes: dict[str, int],
        cfg_dropout_prob: float = 0.1,
        condition_strategy: Literal["mutually_exclusive", "joint"] = "mutually_exclusive",
        feature_dim: int = 64,
        backbone: Literal["flat", "uvit"] = "flat",
    ):
        super().__init__()
        if backbone not in {"flat", "uvit"}:
            raise ValueError("backbone must be either 'flat' or 'uvit'")
        if backbone == "uvit" and (n_layer < 2 or n_layer % 2 != 0):
            raise ValueError("U-ViT backbone requires an even n_layer >= 2")
        self.class_vocab_sizes = class_vocab_sizes
        self.cfg_dropout_prob = cfg_dropout_prob
        self.condition_strategy = condition_strategy
        self.backbone = backbone
        self.class_embeddings = nn.ModuleDict()
        for class_name, vocab_size in class_vocab_sizes.items():
            use_cfg_embedding = int(cfg_dropout_prob > 0)
            self.class_embeddings[class_name] = nn.Embedding(
                vocab_size + use_cfg_embedding, n_embed
            )

        self.t_embedder = TimestepEmbedder(n_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, n_embed), requires_grad=False)
        self.n_embed = n_embed
        self.seq_len = seq_len

        self.input_proj_rna = nn.Linear(n_embed_input, n_embed, bias=bias)
        self.input_proj_atac = nn.Linear(n_embed_input, n_embed, bias=bias)

        def make_dual_block() -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    "rna_block": Block(
                        n_embed=n_embed,
                        n_head=n_head,
                        dropout=dropout,
                        bias=bias,
                        norm_layer=norm_layer,
                        multiple_of=multiple_of,
                        layernorm_eps=layernorm_eps,
                        use_adaln=True,
                        elementwise_affine=False,
                    ),
                    "atac_block": Block(
                        n_embed=n_embed,
                        n_head=n_head,
                        dropout=dropout,
                        bias=bias,
                        norm_layer=norm_layer,
                        multiple_of=multiple_of,
                        layernorm_eps=layernorm_eps,
                        use_adaln=True,
                        elementwise_affine=False,
                    ),
                    "cross_attn": CrossAttentionBlock_cell(
                        channels=n_embed,
                        feature_dim=feature_dim,
                    ),
                }
            )

        if self.backbone == "flat":
            self.blocks = nn.ModuleList([make_dual_block() for _ in range(n_layer)])
            self.encoder_blocks = nn.ModuleList()
            self.decoder_blocks = nn.ModuleList()
            self.rna_skip_projs = nn.ModuleList()
            self.atac_skip_projs = nn.ModuleList()
            self.direction_embedding = None
        else:
            n_encoder = n_layer // 2
            n_decoder = n_layer // 2
            self.blocks = nn.ModuleList()
            self.encoder_blocks = nn.ModuleList([make_dual_block() for _ in range(n_encoder)])
            self.decoder_blocks = nn.ModuleList([make_dual_block() for _ in range(n_decoder)])
            self.rna_skip_projs = nn.ModuleList(
                [nn.Linear(2 * n_embed, n_embed, bias=False) for _ in range(n_decoder)]
            )
            self.atac_skip_projs = nn.ModuleList(
                [nn.Linear(2 * n_embed, n_embed, bias=False) for _ in range(n_decoder)]
            )
            self.direction_embedding = nn.Embedding(len(self._DIRECTION_TO_ID), n_embed)

        self.final_layer_rna = FinalLayerDit(n_embed, n_embed_input, bias, layernorm_eps)
        self.final_layer_atac = FinalLayerDit(n_embed, n_embed_input, bias, layernorm_eps)

        self.initialize_weights()

    def _iter_block_dicts(self):
        if self.backbone == "flat":
            yield from self.blocks
        else:
            yield from self.encoder_blocks
            yield from self.decoder_blocks

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_1d_sincos_pos_embed(self.n_embed, self.seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for embedding in self.class_embeddings.values():
            nn.init.normal_(embedding.weight, std=0.02)
        if isinstance(self.direction_embedding, nn.Embedding):
            nn.init.normal_(self.direction_embedding.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for block_dict in self._iter_block_dicts():
            nn.init.constant_(block_dict["rna_block"].adaln_modulation[-1].weight, 0)
            if block_dict["rna_block"].adaln_modulation[-1].bias is not None:
                nn.init.constant_(block_dict["rna_block"].adaln_modulation[-1].bias, 0)
            nn.init.constant_(block_dict["atac_block"].adaln_modulation[-1].weight, 0)
            if block_dict["atac_block"].adaln_modulation[-1].bias is not None:
                nn.init.constant_(block_dict["atac_block"].adaln_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer_rna.adaln_modulation[-1].weight, 0)
        if self.final_layer_rna.adaln_modulation[-1].bias is not None:
            nn.init.constant_(self.final_layer_rna.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_rna.linear.weight, 0)
        if self.final_layer_rna.linear.bias is not None:
            nn.init.constant_(self.final_layer_rna.linear.bias, 0)

        nn.init.constant_(self.final_layer_atac.adaln_modulation[-1].weight, 0)
        if self.final_layer_atac.adaln_modulation[-1].bias is not None:
            nn.init.constant_(self.final_layer_atac.adaln_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer_atac.linear.weight, 0)
        if self.final_layer_atac.linear.bias is not None:
            nn.init.constant_(self.final_layer_atac.linear.bias, 0)

    def _get_condition_embedding(
        self,
        condition: dict[str, torch.Tensor] | None,
        force_drop_ids: bool = True,
    ) -> torch.Tensor:
        if not condition:
            raise ValueError(
                "_get_condition_embedding requires batch metadata; call _empty_condition_embedding instead"
            )
        batch_size = next(iter(condition.values())).shape[0]
        device = next(iter(condition.values())).device
        if self.condition_strategy == "joint":
            return self._get_joint_condition_embedding(condition, batch_size, device)
        else:
            return self._get_mutually_exclusive_condition_embedding(
                condition, batch_size, device, force_drop_ids
            )

    def _empty_condition_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the null conditioning embedding used for CFG."""
        return torch.zeros(batch_size, 1, self.n_embed, device=device)

    def _get_mutually_exclusive_condition_embedding(
        self,
        condition: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        force_drop_ids: bool = True,
    ) -> torch.Tensor:
        available_classes = [
            name for name in sorted(self.class_vocab_sizes.keys()) if name in condition
        ]
        if not available_classes:
            return torch.zeros(batch_size, 1, self.n_embed, device=device)

        selected_class_idx = torch.randint(0, len(available_classes), (), device=device)
        if force_drop_ids:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob

        embeddings = []
        class_names = sorted(self.class_vocab_sizes.keys())
        for class_name in class_names:
            null_token = self.class_vocab_sizes[class_name]
            if class_name in available_classes:
                i = available_classes.index(class_name)
                is_selected_b = (selected_class_idx == i).expand(batch_size)
                cond_vals = condition[class_name]
                null_vals = torch.full_like(cond_vals, null_token)
                if force_drop_ids:
                    cond_or_null = torch.where(drop_mask, null_vals, cond_vals)
                    final_vals = torch.where(is_selected_b, cond_or_null, null_vals)
                else:
                    final_vals = torch.where(is_selected_b, cond_vals, null_vals)
            else:
                final_vals = torch.full((batch_size,), null_token, device=device, dtype=torch.long)

            class_emb = self.class_embeddings[class_name](final_vals)
            embeddings.append(class_emb)

        return sum(embeddings).unsqueeze(1)

    def _get_joint_condition_embedding(
        self, condition: dict[str, torch.Tensor], batch_size: int, device: torch.device
    ) -> torch.Tensor:
        available_classes = [
            name for name in sorted(self.class_vocab_sizes.keys()) if name in condition
        ]
        if not available_classes:
            return torch.zeros(batch_size, 1, self.n_embed, device=device)

        embeddings = []
        class_names = sorted(self.class_vocab_sizes.keys())

        if self.training:
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_prob
        else:
            drop_mask = torch.ones(batch_size, device=device) < 0

        for class_name in class_names:
            condition_values = condition[class_name]
            null_token = self.class_vocab_sizes[class_name]
            final_condition_values = torch.where(drop_mask, null_token, condition_values)
            class_emb = self.class_embeddings[class_name](final_condition_values)
            embeddings.append(class_emb)

        return sum(embeddings).unsqueeze(1)

    def _get_direction_embedding(
        self,
        mode: Literal["joint", "rna_to_atac", "atac_to_rna"] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(self.direction_embedding, nn.Embedding):
            return torch.zeros(batch_size, 1, self.n_embed, device=device)
        direction = "joint" if mode is None else mode
        if direction not in self._DIRECTION_TO_ID:
            raise ValueError(f"Unsupported multimodal flow direction: {direction}")
        direction_id = self._DIRECTION_TO_ID[direction]
        ids = torch.full((batch_size,), direction_id, device=device, dtype=torch.long)
        return self.direction_embedding(ids).unsqueeze(1)

    def _apply_dual_block(
        self,
        h_rna: torch.Tensor,
        h_atac: torch.Tensor,
        t_embedding: torch.Tensor,
        block_dict: nn.ModuleDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_rna = block_dict["rna_block"](x=h_rna, condition=t_embedding)
        h_atac = block_dict["atac_block"](x=h_atac, condition=t_embedding)

        video = h_rna.transpose(-1, -2)
        audio = h_atac.transpose(-1, -2)
        video, audio = block_dict["cross_attn"](video, audio)
        h_rna = video.transpose(-1, -2)
        h_atac = audio.transpose(-1, -2)
        return h_rna, h_atac

    def forward(
        self,
        x_rna: torch.Tensor,
        x_atac: torch.Tensor,
        t: torch.Tensor,
        condition: dict[str, torch.Tensor] | None = None,
        force_drop_ids: bool | None = None,
        mode: Literal["joint", "rna_to_atac", "atac_to_rna"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if force_drop_ids is None:
            force_drop_ids = self.training
        t_embedding = self.t_embedder(t).unsqueeze(1)

        if condition:
            condition_embedding = self._get_condition_embedding(condition, force_drop_ids)
        else:
            condition_embedding = self._empty_condition_embedding(x_rna.shape[0], x_rna.device)
        if condition_embedding is not None:
            t_embedding = t_embedding + condition_embedding
        if self.backbone == "uvit":
            t_embedding = t_embedding + self._get_direction_embedding(
                mode, x_rna.shape[0], x_rna.device
            )

        h_rna = self.input_proj_rna(x_rna) + self.pos_embed
        h_atac = self.input_proj_atac(x_atac) + self.pos_embed

        if self.backbone == "flat":
            for block_dict in self.blocks:
                h_rna, h_atac = self._apply_dual_block(h_rna, h_atac, t_embedding, block_dict)
        else:
            rna_skips: list[torch.Tensor] = []
            atac_skips: list[torch.Tensor] = []
            for block_dict in self.encoder_blocks:
                h_rna, h_atac = self._apply_dual_block(h_rna, h_atac, t_embedding, block_dict)
                rna_skips.append(h_rna)
                atac_skips.append(h_atac)

            for idx, block_dict in enumerate(self.decoder_blocks):
                h_rna_skip = rna_skips[-(idx + 1)]
                h_atac_skip = atac_skips[-(idx + 1)]
                h_rna = self.rna_skip_projs[idx](torch.cat([h_rna, h_rna_skip], dim=-1))
                h_atac = self.atac_skip_projs[idx](torch.cat([h_atac, h_atac_skip], dim=-1))
                h_rna, h_atac = self._apply_dual_block(h_rna, h_atac, t_embedding, block_dict)

        out_rna = self.final_layer_rna(h_rna, t_embedding)
        out_atac = self.final_layer_atac(h_atac, t_embedding)

        return out_rna, out_atac
