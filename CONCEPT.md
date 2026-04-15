# Spatial-Temporal Transition Model for Disease Dynamics

A model that learns what spatial features of tissue organization drive transitions
between disease states, built on top of pretrained spatial transcriptomics
foundation models.

## Motivation

Spatial transcriptomics foundation models (Nicheformer, Novae) can encode cells in
their spatial context, but they treat each timepoint as a static snapshot. They
answer "what is this cell/niche?" but not "what is this niche becoming, and why?"

We have a unique dataset: two EAE (Experimental Autoimmune Encephalomyelitis)
mouse models of Multiple Sclerosis with distinct disease trajectories captured at
multiple timepoints with spatial transcriptomics. One model (PLP) follows a
relapsing-remitting course with three disease peaks and two remissions. The other
(MOG) follows a chronic course. Both are profiled with the same 5101-gene spatial
panel across multiple spinal cord regions.

This paired design lets us ask: **what spatial features at a given disease state
predict whether the tissue will progress toward recovery or remain chronically
diseased?**

## Data

### Disease models

```
MOG-EAE (chronic):
  MOG CFA ─→ non symptomatic ─→ early onset ─→ chronic peak ─→ chronic long

PLP-EAE (relapsing-remitting):
  PLP CFA ─→ onset I ─→ onset II ─→ monophasic
                                     peak I ─→ remitt I ─→ peak II ─→ remitt II ─→ peak III
```

### Summary

| Property | Value |
|---|---|
| Total samples | ~107 |
| Total cells | ~900,000 |
| Gene panel | 5,101 genes |
| Animals per timepoint | ~3 (varies) |
| Regions per animal | Multiple spinal cord regions |
| Species | Mouse |

### What makes this dataset valuable

- **Repeated transitions**: PLP gives three peaks and two remissions — the same
  type of transition observed multiple times, providing natural replicates.
- **Chronic vs. relapsing comparison**: MOG and PLP at comparable clinical severity
  but divergent outcomes. Spatial differences at these matched timepoints are
  candidate explanations for why one remits and the other doesn't.
- **Within-animal spatial variation**: Multiple spinal cord regions per animal means
  lesion-adjacent, distal, and spared tissue are all captured at the same systemic
  timepoint. This separates local spatial dynamics from systemic animal-level state.

### Developmental data (planned)

Incorporating public developmental spatial transcriptomics data (e.g., mouse CNS
developmental atlases) to learn baseline spatial programs. Remyelination during
remission may recapitulate developmental myelination programs, and disease may
disrupt developmental spatial patterning. A model trained on both can quantify
how far disease deviates from normal development and how completely remission
restores it.

## Approach

### Overview

Rather than training a spatial foundation model from scratch (which requires
100M+ cells and large compute), we build a temporal transition model on top of
existing pretrained spatial encoders. This separates the problem into a solved
part (spatial cell encoding) and an unsolved part (temporal dynamics), and keeps
compute requirements modest.

```
┌─────────────────────────────────────────────────┐
│  Level 1: Pretrained spatial encoder             │
│  (Nicheformer / Novae, frozen or lightly tuned)  │
│  Input: gene expression + spatial context        │
│  Output: cell embeddings                         │
├─────────────────────────────────────────────────┤
│  Level 2: Niche / region aggregation             │
│  Group cells into spatial neighborhoods          │
│  Produce region-level representations per        │
│  timepoint                                       │
├─────────────────────────────────────────────────┤
│  Level 3: Temporal transition model              │
│  Input: region representations at time t         │
│  Output: predicted state at t+1,                 │
│          transition factor attributions           │
│  THIS IS THE NOVEL CONTRIBUTION                  │
└─────────────────────────────────────────────────┘
```

### Level 1: Spatial cell encoding

Use a pretrained model to embed each cell in its spatial context. Both Nicheformer
and Novae are publicly available with pretrained weights.

**Primary recommendation: Novae**, for these reasons:
- Panel-invariant — the 5101-gene panel works without vocabulary mapping.
- Graph-native — spatial relationships between cells are the input structure,
  not a side feature. Natural fit for spatial organization questions.
- Built-in batch correction via optimal transport — important across 107 samples.
- Spatial domain assignment is a direct output, feeding straight into Level 2.

**Secondary option: Nicheformer**, which has complementary strengths:
- Trained on 3.5x more data (110M vs 30M cells).
- Includes dissociated scRNA-seq in training, which helps disentangle
  cell-intrinsic changes (gene program shifts) from spatial changes (cell
  reorganization). Both matter during disease transitions.
- More flexible Transformer architecture for potential fine-tuning.

**Validation approach**: Run a few representative samples (one baseline, one peak,
one remission) through both encoders. Visualize embeddings — if disease states
separate cleanly in one and not the other, that decides it. If both work well,
concatenating embeddings from both encoders gives the temporal model access to
complementary information.

No training required at this level — just inference.

### Level 2: Niche/region aggregation

Aggregate cell embeddings into spatial neighborhood representations. This defines
the "units" that the temporal model operates on. Options to explore:

- **Spatial domain clustering**: Cluster cells within each timepoint into spatial
  domains (using the spatial encoder's embeddings), then represent each domain by
  its mean embedding + composition statistics.
- **Grid binning**: Overlay a regular grid on each tissue section, aggregate cells
  per bin. Simpler, less biologically motivated, but reproducible across samples.
- **Anatomical landmarks**: If spinal cord regions have consistent anatomical
  structure (grey matter, white matter tracts, meninges), aggregate by known
  anatomical compartments. Most interpretable.
- **Learned aggregation**: A small attention-pooling layer that learns to weight
  cells within a neighborhood. Most flexible but requires more data.

The right choice depends on how consistent the spatial structure is across
samples. Start simple (anatomical or grid-based), move to learned if needed.

### Level 3: Temporal transition model

This is the core contribution. Given region-level representations at timepoint t,
the model learns:

1. **What the region will look like at t+1** (predictive)
2. **What features of the current state drive the transition** (interpretive)

#### Continuous state space, not discrete labels

Disease stages are modeled as positions in a continuous embedding space, not
discrete categories. An animal labeled "peak I" that is spatially similar to
"remitt I" animals may be an early remitter — the model should capture this
rather than forcing categorical boundaries.

Clinical scores and other continuous phenotyping data serve as soft anchors in
this space, not hard classification targets.

#### Training objectives

Several self-supervised and supervised objectives to explore:

- **Next-state prediction**: Given spatial state at t, predict state at t+1.
  The most direct objective for learning transitions. Works with consecutive
  timepoint pairs.
- **Temporal ordering**: Given shuffled snapshots from a trajectory, recover the
  correct temporal order. Forces the model to learn progression structure.
- **Contrastive transitions**: Same-region-across-time pairs should be closer than
  different-region pairs. Teaches temporal correspondence without requiring
  exact prediction.
- **Transition direction classification**: Given a spatial state, predict whether
  the trajectory leads to remission or chronic disease. The most clinically
  relevant objective.

These can be combined in a multi-task setup.

#### Interpretability

The central question — "what factors drive transitions" — requires interpretable
outputs, not just accurate predictions. Design choices that support this:

- Attention-based architecture: attention weights over spatial features indicate
  what the model considers important for each transition.
- Factor decomposition: decompose each predicted transition into contributions
  from gene programs, cell type composition, and spatial organization.
- Comparison analysis: for matched disease severity between MOG (→chronic) and
  PLP (→remission), the model's differential attention highlights candidate
  mechanisms of remission.

## Key scientific questions this enables

### 1. What drives remission vs. chronicity?

At comparable clinical severity, MOG goes chronic while PLP remits. Project both
into the same spatial embedding space. Differences in niche composition, spatial
organization, or gene program activity are candidate explanations.

### 2. Are relapses mechanistically identical?

PLP peak I, peak II, and peak III — do the same spatial programs activate each
time? Or does the tissue find different paths to the same clinical phenotype?
Does accumulated tissue damage change the spatial signature of later peaks?

### 3. Does the tissue retain spatial memory?

At remission, does the tissue return to its baseline spatial organization, or do
spatial signatures persist that predict where the next relapse will occur?

### 4. Does remission recapitulate development?

If trained jointly on developmental data: do remission niches activate
developmental myelination/patterning programs? If so, which developmental stages
does remission most resemble?

### 5. What distinguishes the first onset from relapse?

PLP onset I (naive tissue encountering disease) vs. remitt I → peak II (tissue
that has recovered once and is relapsing) — are the spatial transition programs
different? This asks whether the tissue has fundamentally changed.

## Practical considerations

### Compute requirements

- Level 1 (inference through pretrained model): single GPU, hours not days
- Level 2 (aggregation): CPU, minutes
- Level 3 (temporal model training): single GPU, ~hundreds of region-level
  observations — this is a small model

No cluster-scale compute needed.

### Validation strategy

With ~3 animals per timepoint, overfitting is the primary risk. Strategies:

- **Leave-one-animal-out cross-validation**: Train on all animals except one at
  each timepoint, predict the held-out animal's transitions. This is the
  strictest test.
- **Cross-model validation**: Train on MOG transitions, test whether the model
  generalizes to PLP (and vice versa). Shared programs (onset) should transfer;
  model-specific programs (remission) should not.
- **Within-animal region consistency**: Multiple regions per animal provide
  internal replication. Predictions should be consistent across regions at
  similar disease stages within the same animal.

### Relationship to existing work

| Model | What it does | How we differ |
|---|---|---|
| Nicheformer | Static spatial cell embeddings | We add temporal dynamics |
| Novae | Spatial domain identification | We model how domains transition |
| scMOBA | Multi-task conversational model | We focus on temporal prediction |
| RNA velocity | Infers dynamics from splicing | We use explicit multi-timepoint spatial data |
| CellRank | Trajectory inference | We operate in spatial context with real timepoints |

The closest conceptual predecessor is trajectory inference (Monocle, CellRank),
but those methods operate on dissociated single-cell data and infer pseudotime.
We work with real timepoints and real spatial coordinates.

## Next steps

1. **Data inventory**: Catalog all 107 samples with metadata (model, timepoint,
   animal ID, region, cell count, clinical score if available). Identify gaps.
2. **Gene panel overlap**: Check how the 5101-gene panel maps to Nicheformer and
   Novae training vocabularies. This determines which pretrained encoder to use.
3. **Baseline embeddings**: Run a few representative samples through both
   pretrained encoders. Visualize embeddings, check if disease states separate.
4. **Niche definition**: Experiment with aggregation strategies on a subset of
   samples. Determine what spatial resolution is meaningful for temporal modeling.
5. **Prototype transition model**: Start with simple approaches (e.g., linear
   regression on niche embeddings across consecutive timepoints) before building
   more complex architectures. If linear models already capture signal, the
   temporal structure is strong.
6. **Developmental data integration**: Identify suitable public datasets and
   assess compatibility with the disease data (gene overlap, spatial technology).

## References and inspiration

- Nicheformer: Schaar, Tejada-Lapuerta et al. (2024). Foundation model for
  single-cell and spatial omics. Theis Lab.
  - Code: https://github.com/theislab/nicheformer
  - Data: https://huggingface.co/datasets/theislab/SpatialCorpus-110M
- Novae: Graph-based foundation model for spatial transcriptomics.
  Nature Methods (2025). https://www.nature.com/articles/s41592-025-02899-6
- scMOBA: Wei et al. (2025). Conversational foundation model for single-cell
  multi-omics. https://github.com/Sherryweiran/scMOBA
