from __future__ import annotations
import torch
from torchmetrics import Metric
from torchmetrics.text import WordErrorRate 

# ---------------------------------------------------------------------------
# Custom Sequence Accuracy Metric 
# ---------------------------------------------------------------------------
class SequenceAccuracy(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: list[str], targets: list[str]) -> None:
        for p, t in zip(preds, targets):
            if p.strip() == t.strip():
                self.correct += 1
            self.total += 1

    def compute(self) -> torch.Tensor:
        return self.correct.float() / torch.clamp(self.total, min=1)


# ---------------------------------------------------------------------------
# Initializing the Metric Pipeline
# ---------------------------------------------------------------------------
from torchmetrics import MetricCollection

def get_ctc_metric_collection() -> MetricCollection:
    return MetricCollection({
        "wer": WordErrorRate(),
        "seq_acc": SequenceAccuracy()
    })


# ---------------------------------------------------------------------------
# CTC greedy decoder
# ---------------------------------------------------------------------------
def batch_ctc_greedy_decode(
    ctc_logits: torch.Tensor,       # (T, B, C)
    input_lengths: torch.Tensor,    # (B,)
    id2token: dict[int, str],
    blank_id: int = 0,
) -> list[str]:
    logits_bcT = ctc_logits.permute(1, 0, 2).detach().cpu()
    results = []
    for i, length in enumerate(input_lengths.tolist()):
        sample_logits = logits_bcT[i, :length, :]
        ids = sample_logits.argmax(dim=-1).tolist()
        
        tokens = []
        prev = None
        for idx in ids:
            if idx == blank_id:
                prev = idx
                continue
            if idx != prev:
                tokens.append(id2token.get(idx, "<unknown>"))
            prev = idx
        results.append(" ".join(tokens))
    return results


def batch_ctc_beam_decode(
    ctc_logits: torch.Tensor,       # (T, B, C)  — raw logits from model
    input_lengths: torch.Tensor,    # (B,)
    id2token: dict[int, str],
    beam_width: int = 10,
    blank_id: int = 0,
) -> list[str]:
    """
    CTC beam search decoder — no external dependencies.
    Drop-in replacement for batch_ctc_greedy_decode.

    Algorithm: standard CTC beam search (Graves 2006).
    Beam state: dict mapping token-ID tuple → (p_blank, p_non_blank)
      - p_blank     : cumulative prob of this prefix ending in a blank
      - p_non_blank : cumulative prob of this prefix ending in a non-blank
    """
    # Work in log-space for numerical stability
    log_probs = torch.log_softmax(ctc_logits, dim=-1)   # (T, B, C)
    log_probs = log_probs.detach().cpu()

    B = log_probs.size(1)
    results = []

    for i in range(B):
        T_i = input_lengths[i].item()
        # Shape: (T_i, C)
        lp = log_probs[:T_i, i, :]

        # ---- Initialise beam ----
        # Each entry: prefix_tuple → [log_p_blank, log_p_non_blank]
        NEG_INF = float("-inf")
        beam = {(): [0.0, NEG_INF]}   # empty prefix, starts with p_blank=1 (log=0)

        for t in range(T_i):
            next_beam = {}

            # Pre-fetch log probs for this frame as a plain list (faster lookup)
            frame_lp = lp[t].tolist()

            for prefix, (log_pb, log_pnb) in beam.items():
                # Total log-prob of this prefix so far
                log_p_total = _log_add(log_pb, log_pnb)

                # ---- Extend with blank ----
                # Blank can follow anything; result keeps the same prefix
                new_log_pb = log_p_total + frame_lp[blank_id]
                _beam_update(next_beam, prefix, new_log_pb, NEG_INF)

                # ---- Extend with each non-blank token ----
                for c in range(len(frame_lp)):
                    if c == blank_id:
                        continue

                    lp_c = frame_lp[c]

                    if prefix and prefix[-1] == c:
                        # Same token as last in prefix:
                        #   - can only extend cleanly if last frame was blank
                        new_log_pnb = log_pb + lp_c
                    else:
                        # Different token: extend from either blank or non-blank ending
                        new_log_pnb = log_p_total + lp_c

                    new_prefix = prefix + (c,)
                    _beam_update(next_beam, new_prefix, NEG_INF, new_log_pnb)

                    # If same token as last: also allow extending the SAME prefix
                    # (i.e. the repeated token doesn't add a new symbol)
                    if prefix and prefix[-1] == c:
                        new_log_pnb_same = log_pnb + lp_c
                        _beam_update(next_beam, prefix, NEG_INF, new_log_pnb_same)

            # ---- Prune to top beam_width prefixes ----
            beam = dict(
                sorted(
                    next_beam.items(),
                    key=lambda kv: _log_add(kv[1][0], kv[1][1]),
                    reverse=True,
                )[:beam_width]
            )

        # ---- Extract best prefix ----
        best_prefix = max(beam, key=lambda p: _log_add(beam[p][0], beam[p][1]))
        tokens = [id2token.get(idx, "<unknown>") for idx in best_prefix]
        results.append(" ".join(tokens))

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _beam_update(
    beam: dict,
    prefix: tuple,
    log_pb: float,
    log_pnb: float,
) -> None:
    """Add or accumulate a (log_p_blank, log_p_non_blank) entry into the beam."""
    if prefix not in beam:
        beam[prefix] = [float("-inf"), float("-inf")]
    existing = beam[prefix]
    existing[0] = _log_add(existing[0], log_pb)
    existing[1] = _log_add(existing[1], log_pnb)


def format_metrics(metrics: dict[str, torch.Tensor]) -> str:
    return (
        f"WER={metrics['wer']*100:.1f}%  "
        f"SeqAcc={metrics['seq_acc']*100:.1f}%"
    )