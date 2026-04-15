"""Aggregate cell-level embeddings to niche/region-level representations."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from rich.console import Console

from .config import (
    MODEL_KEY,
    REGION_KEY,
    SAMPLE_KEY,
    STAGE_KEY,
    AggregationConfig,
    model_short_name,
)

console = Console()


# ---------------------------------------------------------------------------
# Niche representation
# ---------------------------------------------------------------------------


class NicheRepresentation:
    """Region-level representation for a single sample at a single timepoint.

    Attributes:
        sample_id: unique sample identifier
        model: disease model short name ("MOG" or "PLP")
        stage: unified biological stage (e.g. "PEAK1", "REMISSION1")
        region: spinal cord region ("C", "L", "T")
        niche_ids: array of niche/domain identifiers
        embeddings: (n_niches, embedding_dim) mean embedding per niche
        composition: (n_niches, n_cell_types) cell type proportions per niche
        cell_counts: (n_niches,) number of cells per niche
        centroids: (n_niches, 2) spatial centroid of each niche
        clinical_score: optional continuous clinical score
    """

    def __init__(
        self,
        sample_id: str,
        model: str,
        stage: str,
        region: str,
        niche_ids: np.ndarray,
        embeddings: np.ndarray,
        cell_counts: np.ndarray,
        centroids: np.ndarray,
        composition: np.ndarray | None = None,
        cell_type_names: list[str] | None = None,
        clinical_score: float | None = None,
    ):
        self.sample_id = sample_id
        self.model = model
        self.stage = stage
        self.region = region
        self.niche_ids = niche_ids
        self.embeddings = embeddings
        self.cell_counts = cell_counts
        self.centroids = centroids
        self.composition = composition
        self.cell_type_names = cell_type_names
        self.clinical_score = clinical_score

    @property
    def n_niches(self) -> int:
        return len(self.niche_ids)

    @property
    def embedding_dim(self) -> int:
        return self.embeddings.shape[1]

    def __repr__(self) -> str:
        return (
            f"NicheRepresentation(sample={self.sample_id}, "
            f"stage={self.stage}, region={self.region}, "
            f"n_niches={self.n_niches}, embed_dim={self.embedding_dim})"
        )


# ---------------------------------------------------------------------------
# Internal: extract sample-level metadata from obs
# ---------------------------------------------------------------------------


def _extract_sample_meta(obs: pd.DataFrame) -> dict:
    """Pull sample-level metadata from an obs DataFrame."""
    return {
        "sample_id": obs[SAMPLE_KEY].iloc[0] if SAMPLE_KEY in obs.columns else "unknown",
        "model": model_short_name(obs[MODEL_KEY].iloc[0]) if MODEL_KEY in obs.columns else "unknown",
        "stage": obs[STAGE_KEY].iloc[0] if STAGE_KEY in obs.columns else "unknown",
        "region": obs[REGION_KEY].iloc[0] if REGION_KEY in obs.columns else "unknown",
        "clinical_score": (
            float(obs["score_sacrifice"].iloc[0])
            if "score_sacrifice" in obs.columns and pd.notna(obs["score_sacrifice"].iloc[0])
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------


def aggregate_by_spatial_domains(
    adata: ad.AnnData,
    config: AggregationConfig | None = None,
) -> NicheRepresentation:
    """Aggregate cell embeddings by spatial domain assignments."""
    config = config or AggregationConfig()

    if config.domain_key not in adata.obs.columns:
        raise ValueError(f"Domain column '{config.domain_key}' not found in adata.obs")
    if config.embedding_key not in adata.obsm:
        raise ValueError(f"Embedding key '{config.embedding_key}' not found in adata.obsm")

    domains = adata.obs[config.domain_key].values
    embeddings = adata.obsm[config.embedding_key]
    spatial = adata.obsm["spatial"][:, :2]
    unique_domains = np.unique(domains)

    has_cell_types = config.include_composition and config.cell_type_key in adata.obs.columns
    if has_cell_types:
        all_cell_types = sorted(adata.obs[config.cell_type_key].dropna().unique())
        cell_type_to_idx = {ct: i for i, ct in enumerate(all_cell_types)}
    else:
        all_cell_types = None

    mean_embeddings = []
    cell_counts = []
    centroids = []
    compositions = []

    for domain in unique_domains:
        mask = domains == domain
        n_cells = mask.sum()

        mean_embeddings.append(np.mean(embeddings[mask], axis=0))
        cell_counts.append(n_cells)
        centroids.append(np.mean(spatial[mask], axis=0))

        if has_cell_types:
            cts = adata.obs.loc[mask, config.cell_type_key].values
            comp = np.zeros(len(all_cell_types))
            for ct in cts:
                if ct in cell_type_to_idx:
                    comp[cell_type_to_idx[ct]] += 1
            comp = comp / n_cells
            compositions.append(comp)

    meta = _extract_sample_meta(adata.obs)

    return NicheRepresentation(
        sample_id=meta["sample_id"],
        model=meta["model"],
        stage=meta["stage"],
        region=meta["region"],
        niche_ids=unique_domains,
        embeddings=np.stack(mean_embeddings),
        cell_counts=np.array(cell_counts),
        centroids=np.stack(centroids),
        composition=np.stack(compositions) if compositions else None,
        cell_type_names=all_cell_types,
        clinical_score=meta["clinical_score"],
    )


def aggregate_by_grid(
    adata: ad.AnnData,
    config: AggregationConfig | None = None,
) -> NicheRepresentation:
    """Aggregate cell embeddings by a regular spatial grid."""
    config = config or AggregationConfig()

    if config.embedding_key not in adata.obsm:
        raise ValueError(f"Embedding key '{config.embedding_key}' not found in adata.obsm")

    embeddings = adata.obsm[config.embedding_key]
    spatial = adata.obsm["spatial"][:, :2]

    bin_x = (spatial[:, 0] // config.grid_size_um).astype(int)
    bin_y = (spatial[:, 1] // config.grid_size_um).astype(int)
    bin_ids = np.array([f"{x}_{y}" for x, y in zip(bin_x, bin_y)])
    unique_bins = np.unique(bin_ids)

    has_cell_types = config.include_composition and config.cell_type_key in adata.obs.columns
    if has_cell_types:
        all_cell_types = sorted(adata.obs[config.cell_type_key].dropna().unique())
        cell_type_to_idx = {ct: i for i, ct in enumerate(all_cell_types)}
    else:
        all_cell_types = None

    mean_embeddings = []
    cell_counts = []
    centroids = []
    compositions = []

    for bin_id in unique_bins:
        mask = bin_ids == bin_id
        n_cells = mask.sum()

        mean_embeddings.append(np.mean(embeddings[mask], axis=0))
        cell_counts.append(n_cells)
        centroids.append(np.mean(spatial[mask], axis=0))

        if has_cell_types:
            cts = adata.obs.loc[mask, config.cell_type_key].values
            comp = np.zeros(len(all_cell_types))
            for ct in cts:
                if ct in cell_type_to_idx:
                    comp[cell_type_to_idx[ct]] += 1
            comp = comp / n_cells
            compositions.append(comp)

    meta = _extract_sample_meta(adata.obs)

    return NicheRepresentation(
        sample_id=meta["sample_id"],
        model=meta["model"],
        stage=meta["stage"],
        region=meta["region"],
        niche_ids=unique_bins,
        embeddings=np.stack(mean_embeddings),
        cell_counts=np.array(cell_counts),
        centroids=np.stack(centroids),
        composition=np.stack(compositions) if compositions else None,
        cell_type_names=all_cell_types,
        clinical_score=meta["clinical_score"],
    )


# ---------------------------------------------------------------------------
# Main aggregation dispatcher
# ---------------------------------------------------------------------------


def aggregate_sample(
    adata: ad.AnnData,
    config: AggregationConfig | None = None,
) -> NicheRepresentation:
    """Aggregate a single sample using the configured method."""
    config = config or AggregationConfig()

    if config.method == "spatial_domains":
        return aggregate_by_spatial_domains(adata, config)
    elif config.method == "grid":
        return aggregate_by_grid(adata, config)
    else:
        raise ValueError(f"Unknown aggregation method: {config.method}")


def aggregate_all_samples(
    samples: dict[str, ad.AnnData],
    config: AggregationConfig | None = None,
) -> list[NicheRepresentation]:
    """Aggregate all samples to niche-level representations."""
    config = config or AggregationConfig()
    representations = []

    for sample_id, adata in samples.items():
        try:
            rep = aggregate_sample(adata, config)
            representations.append(rep)
        except (ValueError, KeyError) as e:
            console.print(f"[yellow]Skipping {sample_id}: {e}[/yellow]")

    console.print(f"Aggregated {len(representations)} samples")
    total_niches = sum(r.n_niches for r in representations)
    console.print(f"Total niches: {total_niches} (avg {total_niches / max(len(representations), 1):.0f} per sample)")
    return representations


# ---------------------------------------------------------------------------
# Summary utilities
# ---------------------------------------------------------------------------


def summarize_representations(reps: list[NicheRepresentation]) -> pd.DataFrame:
    """Build a summary table of niche representations."""
    rows = []
    for rep in reps:
        rows.append({
            "sample_id": rep.sample_id,
            "model": rep.model,
            "stage": rep.stage,
            "region": rep.region,
            "n_niches": rep.n_niches,
            "total_cells": rep.cell_counts.sum(),
            "mean_cells_per_niche": rep.cell_counts.mean(),
            "embedding_dim": rep.embedding_dim,
            "clinical_score": rep.clinical_score,
        })
    return pd.DataFrame(rows)
