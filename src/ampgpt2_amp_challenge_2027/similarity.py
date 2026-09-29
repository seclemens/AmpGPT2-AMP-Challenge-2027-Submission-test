"""
Local-alignment sequence similarity used for novelty and diversity filtering.

The implementation follows the similarity scoring used by the AMP-Diffusion starter kit.
"""

from __future__ import annotations

from Bio import Align

_aligner = Align.PairwiseAligner()
_aligner.mode = "local"
_aligner.match_score = 1.0
_aligner.mismatch_score = -1.0
_aligner.open_gap_score = -1.0
_aligner.extend_gap_score = -1.0


def local_similarity(a: str, b: str) -> float:
    """Normalized Smith-Waterman local-alignment similarity between two peptides, in [0, 1]."""
    score = _aligner.score(a, b)
    return max(0.0, score) / max(len(a), len(b))