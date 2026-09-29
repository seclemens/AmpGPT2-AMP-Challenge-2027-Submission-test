"""AMP Challenge 2027 generation and candidate-selection pipeline for AmpGPT2.

AmpGPT2 generates a broad antimicrobial-peptide library. Generated sequences
are filtered for validity, uniqueness, reference-set matches, and predicted
antibacterial activity using APEX. Top candidates are selected by predicted
MIC, novelty, and sequence diversity following the filtering strategy of the
AMP-Diffusion starter kit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import Levenshtein

from .model import AmpGPT2Generator
from .scoring import score_apex
from .similarity import local_similarity
from transformers import set_seed as hf_set_seed
import torch
import numpy as np
import random
import os


ROOT = Path(__file__).resolve().parents[2]
MODEL_CHECKPOINT = ROOT / "checkpoint" / "AmpGPT2"
ANTIBACTERIAL_FASTA = ROOT / "data" / "antibacterial.fasta"
COMPASS_FASTA = ROOT / "data" / "compass_shortlist.fasta"
APEX_DIR = ROOT / "apex"

AA20 = set("ACDEFGHIKLMNPQRSTVWY")
MIN_AA = 8
MAX_AA = 50

MIN_LENGTH = 8
MAX_LENGTH = 50
SIMILARITY_THRESHOLD = 0.8 # submission requirement
MIC_THRESHOLD_TOP_100 = 64.0 # (1) activity: keep APEX mean MIC <= 64 uM
NOVELTY_THRESHOLD = 0.60  # (2) exclude local-alignment similarity > 0.60 to known AMPs
DIVERSITY_THRESHOLD = 0.40  # (3) among survivors, > 0.40 pair -> keep the lower-MIC peptide
MIC_THRESHOLD_50000_SET = 256.0 # model is not only trained on antibacterial



def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    hf_set_seed(seed, True) #just in case something is missed


def read_fasta(path: Path) -> list[str]:
    sequences = []
    current = []

    with path.open() as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current = []
            else:
                current.append(line)

    if current:
        sequences.append("".join(current).upper())

    return sequences


def write_fasta(
    sequences: list[str],
    path: Path,
    prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as handle:
        for i, sequence in enumerate(sequences, start=1):
            handle.write(f">{prefix}_{i:05d}\n{sequence}\n")


def valid_sequence(sequence: str) -> bool:
    return (
        MIN_AA <= len(sequence) <= MAX_AA
        and set(sequence) <= AA20
    )


def challenge_identity_ok(
    sequence: str,
    antibacterial: list[str],
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """
    Top-100 challenge guard.

    Requirement is that no top sequence exceeds 80% Levenshtein ratio
    against antibacterial.fasta.
    """
    return all(
        Levenshtein.ratio(sequence, known) <= threshold
        for known in antibacterial
    )


def select_top100(
    library_scores: dict[str, float],
    *,
    compass: list[str],
    antibacterial: list[str],
    top_k: int = 100,
    challenge_identity_threshold: float = SIMILARITY_THRESHOLD,
    novelty_threshold: float = NOVELTY_THRESHOLD,
    diversity_threshold: float = DIVERSITY_THRESHOLD,
) -> list[str]:
    """
    based on diffusion starter kit:

    Order:
      1. APEX mean MIC ascending
      2. mandatory Levenshtein <= 0.80 vs antibacterial
      3. local similarity <= 0.60 vs references
      4. local similarity <= 0.40 vs already selected peptides

    If fewer than top_k survive, raise an error
    """
    import Levenshtein

    # Lowest predicted MIC first.
    ranked = sorted(
        library_scores,
        key=lambda seq: library_scores[seq],
    )

    selected: list[str] = []

    for seq in ranked:
        if any(Levenshtein.ratio(seq, ref) > challenge_identity_threshold for ref in antibacterial):
            continue

        if any(local_similarity(seq, other) > diversity_threshold for other in selected):
            continue

        if any(local_similarity(seq, ref) > novelty_threshold for ref in compass):
            continue

        if any(local_similarity(seq, ref) > novelty_threshold for ref in antibacterial):
            continue

        selected.append(seq)

        if len(selected) == top_k:
            return selected

    raise RuntimeError(
        f"Only {len(selected)} of {top_k} peptides survived the filtering"
    )

def generate_library(args, device, antibacterial, compass) -> Any:
    generator = AmpGPT2Generator(checkpoint=Path(args.model_checkpoint), device=device)

    library: list[str] = []
    library_scores: dict[str, float] = {}

    seen: set[str] = set()

    print(
        f"Generating AmpGPT2 candidates until "
        f"{args.n_sequences:,} pass the <= "
        f"{MIC_THRESHOLD_50000_SET:g} µM APEX gate,"
        f"removing sequences that are too short/long or "
        f"100% identical to the references(antibacterial.fasta and compass_shortlist.fasta)"
        f"seed={args.seed:}, device={device} ..."
    )


    while len(library) < args.n_sequences:
        raw = generator.generate(
            5000,
            batch_size=args.batch_size,
        )

        candidates: list[str] = []

        for sequence in raw:
            if not valid_sequence(sequence):
                continue

            if sequence in seen:
                continue

            seen.add(sequence)

            # Mandatory full-library exact-match exclusion.
            if sequence in antibacterial or sequence in compass:
                continue

            candidates.append(sequence)

        if not candidates:
            continue

        scores = score_apex(
            candidates,
            gpu=False,
        )

        for row in scores.itertuples(index=False):
            sequence = row.sequence
            mean_mic = float(row.mean_mic)

            if mean_mic > MIC_THRESHOLD_50000_SET:
                continue

            library.append(sequence)
            library_scores[sequence] = mean_mic

            if len(library) == args.n_sequences:
                break

        print(
            f"library: {len(library):,}/{args.n_sequences:,} "
            f"| unique generated: {len(seen):,}"
        )
    return library, library_scores

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-sequences", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-checkpoint", default=str(MODEL_CHECKPOINT))
    parser.add_argument("--antibacterial-fasta", default=str(ANTIBACTERIAL_FASTA))
    parser.add_argument("--compass-fasta", default=str(COMPASS_FASTA))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir",default="generate")

    args = parser.parse_args()

    for path in (args.model_checkpoint, args.antibacterial_fasta, args.compass_fasta):
        if not Path(path).exists():
            sys.exit(f"ERROR: required path not found (cwd: {os.getcwd()}): {path}")

    device = torch.device(args.device)
    set_seed(args.seed)

    antibacterial = list(set(read_fasta(Path(args.antibacterial_fasta))))
    compass = list(set(read_fasta(Path(args.compass_fasta))))

    library, library_scores = generate_library(args, device, antibacterial, compass)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_fasta(
        library,
        output_dir / "library.fasta",
        "ampgpt2",
    )

    print("Selecting top candidates...")

    top = select_top100(
        library_scores,
        compass=compass,
        antibacterial=antibacterial,
        top_k=args.top_k,
    )

    write_fasta(
        top,
        output_dir / "top.fasta",
        "ampgpt2_top",
    )

    print(f"Wrote {len(library):,} sequences to:")
    print(output_dir / "library.fasta")

    print(f"Wrote {len(top):,} top candidates to:")
    print(output_dir / "top.fasta")
