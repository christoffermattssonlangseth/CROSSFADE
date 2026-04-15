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

## Key findings

### Relapses are not replays

Each PLP disease peak is driven by a distinct macrophage program, not a repeated inflammatory episode:

| Peak | Lesion compartment | Molecular program | Key genes |
|------|--------------------|-------------------|-----------|
| Peak 1 | APC lesion | IFN-gamma / antigen presentation | *Stat1*, *Irf1*, *Gbp2*, *H2-Ab1*, *Psmb8* |
| Peak 2 | Lipid/ECM macrophage | Myelin debris processing | *Abca1*, *Gpnmb*, *Grn*, *Trem2* |
| Peak 3 | Foamy-lipid droplet | Chronic lipid accumulation | *Gpnmb*, *Plin2*, *Trem2*, *Cd68*, *Hexa* |

This represents a maturation of the pathological macrophage response across relapse cycles: acute immune attack, then debris cleanup, then chronic foam cell pathology. The Trem2 pathway (implicated in Alzheimer's and other neurodegeneration) becomes dominant at the third peak.

### Remission is active, not passive

- **Remission 1** is an "armed truce" — MHC class II still elevated, but inflammatory effectors (*Ly6i*, *Fpr1/2*) shut down and immune checkpoints (*Lag3*) engage
- **Remission 2** is a deeper molecular remission — circadian anti-inflammatory programs (*Dbp*, *Nr1d1*) return to near-baseline, neural genes recover

### Circadian clock disruption tracks disease activity

Clock output genes (*Dbp*, *Nr1d1*/Rev-Erb-alpha, *Hlf*) are anti-correlated with inflammatory markers across the trajectory. The core oscillator (*Arntl*, *Clock*) remains stable — only the anti-inflammatory output arm is selectively suppressed at peaks. This is consistent with NF-kB-mediated clock disruption, and Rev-Erb-alpha agonists have been shown to reduce EAE severity in mice.

### Early cholesterol biosynthesis response

The earliest disease stages (onset 1-2) activate the cholesterol/lipid biosynthesis pathway (*Hmgcr*, *Srebf2*, *Idi1*, *Ldlr*, *Fasn*), likely representing a myelin repair attempt that fails as immune damage escalates.

## Installation

```bash
pip install -e .
```

## Usage

See notebooks:
- `notebooks/01_explore_and_embed.ipynb` -- Data exploration, Novae embedding, niche aggregation
- `notebooks/02_temporal_transitions.ipynb` -- Transition analysis, temporal model, factor attribution
- `notebooks/03_compartment_transitions.ipynb` -- Compartment-level transition dynamics, disease direction classification
- `notebooks/04_gene_transitions.ipynb` -- Gene-level transition programs, compartment markers, circadian analysis

## Project structure

```
crossfade/
├── CONCEPT.md                           # Detailed project concept
├── README.md
├── pyproject.toml
├── notebooks/
│   ├── 01_explore_and_embed.ipynb       # Spatial embedding pipeline
│   ├── 02_temporal_transitions.ipynb    # Temporal transition model
│   ├── 03_compartment_transitions.ipynb # Compartment dynamics
│   └── 04_gene_transitions.ipynb        # Gene programs
└── src/temporal_foundation/
    ├── config.py                        # Disease models, parameters
    ├── data.py                          # Data loading and organization
    ├── embeddings.py                    # Novae embedding pipeline
    └── aggregation.py                   # Niche-level aggregation
```

## Public datasets for pretraining

Candidate datasets for pretraining a temporal transition model on EAE/MS data.

### Spatial transcriptomics with temporal EAE sampling

| Dataset | Technology | Model / Tissue | Timepoints | Accession |
|---------|-----------|----------------|------------|-----------|
| Kukanja et al., Cell 2024 | ISS (239 genes) + Xenium | MOG-EAE, mouse spinal cord | 4 (day 8, onset, peak, late) | [Zenodo 10.5281/zenodo.8037425](https://doi.org/10.5281/zenodo.8037425) |
| Gadani et al., eLife 2024 | 10x Visium | PLP-EAE, mouse brain | 3 (weeks 6, 8, 10) | [GSE272362](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272362), [GSE236963](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236963) |

### Single-cell RNA-seq with EAE timepoints

| Dataset | Technology | Model / Tissue | Timepoints | Accession |
|---------|-----------|----------------|------------|-----------|
| Jordao et al., Science 2019 | scRNA-seq | MOG-EAE, mouse CNS, ~3.5k myeloid cells | 4 (naive, pre-clinical, onset, peak) | [GSE118948](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE118948) |
| Falcao et al., Nat Med 2018 | scRNA-seq | EAE, mouse spinal cord, ~2.2k OL lineage | 3 (naive, early, chronic) | [GSE113973](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113973) |
| Floriddia et al., Nat Commun 2020 | scRNA-seq + ISS | SCI + EAE, mouse spinal cord + brain | Multiple | [GSE128525](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128525) |

### Demyelination / remyelination (cuprizone)

| Dataset | Technology | Model / Tissue | Timepoints | Accession |
|---------|-----------|----------------|------------|-----------|
| Biogen, 2024 | snRNA-seq + spatial | Cuprizone, mouse brain | 3 (control, 4wk cuprizone, recovery) | [GSE255370](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255370) |

### Human MS (translational validation)

| Dataset | Technology | Tissue | Lesion types | Accession |
|---------|-----------|--------|-------------|-----------|
| Absinta et al., Nature 2021 | snRNA-seq, 66k nuclei | Human brain WM | Chronic active, inactive, periplaque, normal | [GSE180759](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE180759) |
| Lerma-Martin et al., Nat Neurosci 2024 | snRNA-seq + Visium, 34 samples | Human subcortical WM | Chronic active, inactive | [GSE279183](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279183) |
| Feng et al., Immunity 2025 | MERFISH (500 genes), ~400k cells | Human brain | Chronic active MS lesions | [GSE284005](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE284005) |

### Priority for pretraining

1. **Kukanja et al.** -- same tissue (spinal cord), same model (MOG-EAE), spatial, 4 timepoints. Castelo-Branco lab.
2. **Gadani et al.** -- PLP-EAE (same disease model as ours), Visium, 3 timepoints. Tests generalization to brain.
3. **GSE255370 (cuprizone)** -- clean demyelination/remyelination without immune complexity. Spatial + snRNA-seq.

## References

- [Novae](https://github.com/MICS-Lab/novae) -- Graph-based spatial transcriptomics foundation model (Nature Methods, 2025)
- [Nicheformer](https://github.com/theislab/nicheformer) -- Foundation model for single-cell and spatial omics
- [SpatialCorpus-110M](https://huggingface.co/datasets/theislab/SpatialCorpus-110M) -- Pretraining dataset for spatial models
- [scMOBA](https://github.com/Sherryweiran/scMOBA) -- Conversational foundation model for single-cell multi-omics
