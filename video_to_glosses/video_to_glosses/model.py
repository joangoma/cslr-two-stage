import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import CLIPModel  # Full CLIP model for projection head access


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """
    Parameter-free sinusoidal positional encoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class VideoToGlossModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        hidden_dim: int = 512,
        transformer_layers: int = 6,
        transformer_heads: int = 8,
        transformer_ffn_dim: int = 2048,
        transformer_dropout: float = 0.1,
    ):
        super().__init__()

        print(f"Loading visual backbone: {clip_model_name}")
        # Use full CLIPModel to access the projection head (get_image_features → 512-dim)
        self.backbone = CLIPModel.from_pretrained(clip_model_name)

        # Drop the text side entirely (mirrors the framework's FeatureExtractor)
        self.backbone.text_model = None
        self.backbone.text_projection = None

        # self.backbone.vision_model.gradient_checkpointing_enable()

        # Projection head output is always 512 for clip-vit-base-patch32,
        # regardless of the ViT hidden size (768). This matches the framework.
        visual_feat_dim = self.backbone.config.projection_dim  # 512

        # CNN architecture unchanged — only the input channel count drops 768 → 512
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(visual_feat_dim, 2048, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(2048, hidden_dim, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )

        self.ctc_head = nn.Linear(hidden_dim, num_classes)
        self.aux_frame_head = nn.Linear(hidden_dim, num_classes)

        nn.init.constant_(self.ctc_head.bias[0], 2.0)

    def forward(
        self,
        video_frames: torch.Tensor,   # (B, T, C, H, W)
        video_lengths: torch.Tensor,  # (B,) real frame counts (unpadded)
    ):
        B, T, C, H, W = video_frames.shape

        flat_frames = video_frames.view(B * T, C, H, W)

        # get_image_features: runs vision encoder + projection head → (B*T, 512), L2-normalised
        feats = self.backbone.get_image_features(pixel_values=flat_frames)
        feats = feats.view(B, T, -1)                       # (B, T, 512)

        # Zero-out padded frame positions
        pad_mask = (
            torch.arange(T, device=feats.device).unsqueeze(0)
            >= video_lengths.unsqueeze(1)
        )                                                  # (B, T) True = padded
        feats = feats.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        # ---- 2. Temporal CNN (local motion + downsampling) ---------------
        feats = feats.permute(0, 2, 1)                     # (B, 512, T)
        feats = self.temporal_cnn(feats)                   # (B, hidden_dim, T_down)
        feats = feats.permute(0, 2, 1)                     # (B, T_down, hidden_dim)

        T_down = feats.shape[1]

        def conv_out_len(length, kernel, stride, padding, dilation=1):
            return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

        input_lengths = conv_out_len(video_lengths, 5, 2, 2, dilation=1)
        input_lengths = conv_out_len(input_lengths, 5, 1, 4, dilation=2)
        input_lengths = input_lengths.clamp(min=1, max=T_down)

        # ---- 3. Positional encoding --------------------------------------
        feats = self.pos_encoding(feats)                   # (B, T_down, hidden_dim)

        # ---- 4. Temporal Transformer (long-range dependencies) -----------
        transformer_pad_mask = (
            torch.arange(T_down, device=feats.device).unsqueeze(0)
            >= input_lengths.unsqueeze(1)
        )                                                  # (B, T_down)

        feats = self.temporal_transformer(
            feats,
            src_key_padding_mask=transformer_pad_mask,
        )                                                  # (B, T_down, hidden_dim)

        # ---- 5. Prediction heads -----------------------------------------
        frame_logits = self.aux_frame_head(feats.detach())  # (B, T_down, num_classes)
        ctc_logits   = self.ctc_head(feats).permute(1, 0, 2)  # (T_down, B, num_classes)

        return ctc_logits, frame_logits, input_lengths