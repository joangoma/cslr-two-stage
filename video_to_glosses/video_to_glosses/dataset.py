import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms as T
import torch.nn.functional as F
import json


class SignVideoDataset(Dataset):
    def __init__(
        self,
        json_path,
        vocab_path,
        max_frames=700,
        target_height=224,
        target_width=224,
        skip_frames_stride=2,
        augmentation_pipeline=None,

    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.max_frames = max_frames
        self.skip_frames_stride = skip_frames_stride
        self.augmentation_pipeline=augmentation_pipeline
        
        self.resize = T.Resize((target_height, target_width), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if self.augmentation_pipeline is not None:
            print(f"[Dataset] Augmentation pipeline: ACTIVE")
        else:
            print(f"[Dataset] Augmentation pipeline: DISABLED (eval mode)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_path = item["video_path"]
        end_pts = None
        if item.get("timeline"):
            last_end = max(e["end"] for e in item["timeline"])
            end_pts = last_end + 1.0  # small buffer
            start_pts = max(0.0, item["timeline"][0]["start"] - 0.5)
        else:
            start_pts = 0.0

        try:
            if end_pts is not None:
                video, _, info = io.read_video(
                    video_path,
                    start_pts=start_pts,
                    end_pts=end_pts,
                    pts_unit="sec",
                    output_format="TCHW",
                )
            else:
                video, _, info = io.read_video(
                    video_path, pts_unit="sec", output_format="TCHW"
                )
            fps = info.get("video_fps", 25.0)
        except Exception as e:
            print(f"CRITICAL ERROR loading video {video_path}: {e}")
            return None

        if video.size(0) == 0:
            print(f"WARNING: Video {video_path} has 0 frames.")
            return None

        frame_offset = int(round(start_pts * fps)) if end_pts is not None else 0

        # Temporal subsampling before resize (cheaper on large-resolution videos)
        if self.skip_frames_stride > 1:
            video = video[:: self.skip_frames_stride]
            fps_effective = fps / self.skip_frames_stride
        else:
            fps_effective = fps

        video = video[: self.max_frames]

        video = video.float() / 255.0
        video = self.resize(video)
        if self.augmentation_pipeline is not None:
            # Pipeline expects (C, T, H, W)
            video = video.permute(1, 0, 2, 3)
            video = self.augmentation_pipeline(video)
            video = video.permute(1, 0, 2, 3)  # back to (T, C, H, W)
        
        T_len = video.size(0)
        video = self.normalize(video)

        unk_id = self.vocab.get("<unknown>", 1)

        try:
            ctc_targets = [self.vocab[g] for g in item["gloss_sequence"]]
        except KeyError as e:
            print(f"WARNING: Unknown gloss {e} in {video_path}, skipping sample.")
            return None

        frame_targets = torch.full((T_len,), fill_value=-1, dtype=torch.long)

        for entry in item["timeline"]:
            gloss_id = self.vocab.get(entry["gloss"], unk_id)

            # Adjust for start_pts offset and stride
            start_frame = int(round((entry["start"] * fps - frame_offset) / self.skip_frames_stride))
            end_frame = int(round((entry["end"] * fps - frame_offset) / self.skip_frames_stride))

            start_frame = max(0, min(start_frame, T_len - 1))
            end_frame = max(0, min(end_frame, T_len - 1))

            if start_frame <= end_frame:
                frame_targets[start_frame : end_frame + 1] = gloss_id

        cnn_stride = 2
        downsampled_len = max(1, T_len // cnn_stride)
        if downsampled_len < len(ctc_targets):
            print(
                f"WARNING: Skipping {video_path} — downsampled length {downsampled_len} "
                f"< gloss count {len(ctc_targets)}. Video too short for CTC."
            )
            return None

        return {
            "video": video,                                              # (T, C, H, W)
            "ctc_targets": torch.tensor(ctc_targets, dtype=torch.long), # (S,)
            "frame_targets": frame_targets,                              # (T,)
        }


class SignLanguageCollate:
    """
    Handles variable-length batch padding.
    Pass the vocab dict so the collator can build human-readable reference
    strings from ctc_targets — needed for WER / metric computation.
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        self.id2token: dict[int, str] = {}
        if vocab is not None:
            self.id2token = {v: k for k, v in vocab.items()}

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return {}

        B = len(batch)

        video_lengths = torch.tensor([b["video"].size(0) for b in batch], dtype=torch.long)
        ctc_lengths = torch.tensor([b["ctc_targets"].size(0) for b in batch], dtype=torch.long)

        max_T = video_lengths.max().item()
        max_S = ctc_lengths.max().item()

        C, H, W = batch[0]["video"].shape[1:]

        padded_videos = torch.zeros(B, max_T, C, H, W, dtype=torch.float32)

        padded_ctc = torch.full((B, max_S), fill_value=-1, dtype=torch.long)

        # -1 → ignored by CrossEntropyLoss(ignore_index=-1)
        padded_frames = torch.full((B, max_T), fill_value=-1, dtype=torch.long)

        for i, item in enumerate(batch):
            v_len = video_lengths[i].item()
            c_len = ctc_lengths[i].item()
            padded_videos[i, :v_len] = item["video"]
            padded_ctc[i, :c_len] = item["ctc_targets"]
            padded_frames[i, :v_len] = item["frame_targets"]

        ctc_targets_packed = torch.cat(
            [batch[i]["ctc_targets"] for i in range(B)], dim=0
        )


        references = [
            " ".join(
                self.id2token.get(idx.item(), "<unknown>")
                for idx in batch[i]["ctc_targets"]
            )
            for i in range(B)
        ]

        return {
            "videos":       padded_videos,        # (B, T, C, H, W)
            "video_lengths": video_lengths,        # (B,)
            "ctc_targets":  ctc_targets_packed,   # (sum_S,)  1-D for CTCLoss
            "ctc_lengths":  ctc_lengths,          # (B,)
            "frame_targets": padded_frames,       # (B, T)
            "references":   references,           # list[str] for metrics
        }