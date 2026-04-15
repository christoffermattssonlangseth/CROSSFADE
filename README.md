# CROSSFADE

**C**apturing **R**eorganization **O**f **S**patial **S**tates: **F**actors **A**cross **D**isease **E**volution

A temporal transition model for spatial transcriptomics that learns what spatial features of tissue organization drive transitions between disease states.

## Overview

CROSSFADE builds on pretrained spatial foundation models (Novae) to analyze how tissue spatial organization evolves over time. Rather than treating each spatial transcriptomics snapshot in isolation, CROSSFADE models the transitions between disease stages to identify what niche-level features predict whether tissue progresses toward recovery or chronic disease.

## The problem

Spatial transcriptomics foundation models answer: *"what is this cell/niche?"*

CROSSFADE answers: *"what is this niche becoming, and why?"*

## Data

Built on a relapsing-remitting EAE (Experimental Autoimmune Encephalomyelitis) spatial transcriptomics dataset:

- **877,141 cells** across **107 samples**
- **5,101 gene** Xenium panel
- **Two disease models**: MOG (chronic) and PLP (relapsing-remitting with 3 peaks and 2 remissions)
- **Three spinal cord regions** (cervical, lumbar, thoracic) per animal
- Unified biological staging across both models

## Approach

```
Raw spatial transcriptomics data
         |
    [ Novae (pretrained, frozen) ]
    Spatial cell embeddings + domain assignments
         |
    [ Niche aggregation ]
    Region-level representations per sample
         |
    [ CROSSFADE temporal transition model ]
    Transition predictions + factor attributions
```

1. **Spatial encoding** -- Pretrained Novae model encodes cells in their spatial context
2. **Niche aggregation** -- Cell embeddings aggregated to spatial domain level
3. **Transition modeling** -- Attention-based model learns what niche features predict the next disease state
4. **Factor attribution** -- Attention weights identify which niches and cell types drive each transition

## Key questions

- What spatial features at disease peak predict remission (PLP) vs. chronicity (MOG)?
- Do relapses reuse the same spatial transition programs?
- Does the tissue retain spatial memory between disease cycles?
- Which niches are the leading indicators of state transitions?

## Installation

```bash
pip install -e .
```

## Usage

See notebooks:
- `notebooks/01_explore_and_embed.ipynb` -- Data exploration, Novae embedding, niche aggregation
- `notebooks/02_temporal_transitions.ipynb` -- Transition analysis, temporal model, factor attribution

## Project structure

```
crossfade/
├── CONCEPT.md                           # Detailed project concept
├── README.md
├── pyproject.toml
├── notebooks/
│   ├── 01_explore_and_embed.ipynb       # Spatial embedding pipeline
│   └── 02_temporal_transitions.ipynb    # Temporal transition model
└── src/temporal_foundation/
    ├── config.py                        # Disease models, parameters
    ├── data.py                          # Data loading and organization
    ├── embeddings.py                    # Novae embedding pipeline
    └── aggregation.py                   # Niche-level aggregation
```

## References

- [Novae](https://github.com/MICS-Lab/novae) -- Graph-based spatial transcriptomics foundation model (Nature Methods, 2025)
- [Nicheformer](https://github.com/theislab/nicheformer) -- Foundation model for single-cell and spatial omics
- [SpatialCorpus-110M](https://huggingface.co/datasets/theislab/SpatialCorpus-110M) -- Pretraining dataset for spatial models
- [scMOBA](https://github.com/Sherryweiran/scMOBA) -- Conversational foundation model for single-cell multi-omics
