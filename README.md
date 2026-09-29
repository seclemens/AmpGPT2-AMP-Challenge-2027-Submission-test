# AmpGPT2 Base — AMP Challenge 2027

Reproducible AMP Challenge 2027 submission based on the published
[AmpGPT2](https://doi.org/10.1038/s44259-026-00218-3) model. The model was trained on sequences collected in [COMPASS](https://compass.imi.uni-muenster.de/)
Refer to the Paper for detailed training and data description.

## Abstract

We submit AmpGPT2, an autoregressive language model trained on antimicrobial
peptide sequences, as a broad AMP generator for the AMP Challenge 2027.
Because AmpGPT2 is not specific to antibacterial activity against the challenge
pathogen panel, generated peptides are computationally enriched using APEX.

Valid, unique peptides of 8–50 amino acids are generated until 50,000 sequences
with a mean APEX-predicted MIC <=256 µM are obtained. Exact matches to the
challenge antibacterial reference set and the additional COMPASS reference set
are excluded.

Top candidates are ranked by ascending mean APEX-predicted MIC. Candidates are
then filtered using the challenge <=80% Levenshtein-similarity requirement
against the antibacterial reference set, local-alignment novelty against the
antibacterial and COMPASS reference sets, and sequence diversity within the
selected set.

Generation and filtering are fully automated and reproducible from a fixed seed.

## Data description

`data/antibacterial.fasta` contains the antibacterial reference sequences
provided for the AMP Challenge.

`data/compass_shortlist.fasta` contains additional known peptide sequences from
COMPASS restricted to lengths of 8–50 amino acids and excluding exact sequences
already present in `antibacterial.fasta`, that were part of the models training data.

The model itself is the previously published AmpGPT2 checkpoint; no additional
training is performed for this submission.

Computational filtering uses APEX-predicted MIC values together with sequence
novelty and diversity filters.

## Top candidates selection procedure 
based on the Diffusion Selection Process

The 50,000-member library is ranked by ascending mean APEX-predicted MIC.

Starting from the lowest predicted MIC, a peptide is retained only if it:

1. has Levenshtein similarity <=0.80 to every sequence in the challenge
   antibacterial reference set;
2. has local-alignment similarity <=0.60 to every sequence in the antibacterial
   and COMPASS reference sets; and
3. has local-alignment similarity <=0.40 to every previously selected top
   candidate.

Selection continues in MIC order until 100 peptides are obtained.

## Usage

```bash
git lfs install
git lfs pull
uv sync
uv run generate
```
AmpGPT2 generation uses CUDA when available. APEX scoring is executed separately
in CPU mode.
## License

MIT(see [LICENSE](LICENSE)). The included AmpGPT2 checkpoint retains its Apache-2.0 license. APEX and any
adapted code from the challenge starter kits retain their respective upstream licenses.