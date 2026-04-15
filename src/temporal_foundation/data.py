"""Data loading and organization for EAE spatial transcriptomics."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
from rich.console import Console
from rich.table import Table

from .config import (
    COURSE_KEY,
    DATA_PATH,
    MODEL_KEY,
    MODEL_LABELS,
    MODELS,
    REGION_KEY,
    REGIONS,
    SAMPLE_KEY,
    STAGE_KEY,
    get_all_stages,
    model_short_name,
)

console = Console()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_adata(path: str | Path | None = None) -> ad.AnnData:
    """Load the combined EAE dataset."""
    path = Path(path) if path else DATA_PATH
    console.print(f"Loading {path}...")
    adata = ad.read_h5ad(path)
    console.print(f"Loaded {adata.n_obs:,} cells, {adata.n_vars} genes, {adata.obs[SAMPLE_KEY].nunique()} samples")
    return adata


def split_by_sample(adata: ad.AnnData) -> dict[str, ad.AnnData]:
    """Split combined AnnData into per-sample AnnData objects.

    Returns dict mapping sample_id -> AnnData subset.
    """
    samples: dict[str, ad.AnnData] = {}
    for sample_id in adata.obs[SAMPLE_KEY].unique():
        mask = adata.obs[SAMPLE_KEY] == sample_id
        samples[sample_id] = adata[mask].copy()
    console.print(f"Split into {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------


def build_sample_table(adata: ad.AnnData) -> pd.DataFrame:
    """Build a summary table of all samples with their metadata."""
    rows = []
    for sample_id in adata.obs[SAMPLE_KEY].unique():
        mask = adata.obs[SAMPLE_KEY] == sample_id
        subset = adata.obs.loc[mask]
        row = {
            "sample_id": sample_id,
            "n_cells": mask.sum(),
            "model": model_short_name(subset[MODEL_KEY].iloc[0]),
            "stage": subset[STAGE_KEY].iloc[0],
            "course": subset[COURSE_KEY].iloc[0],
            "region": subset[REGION_KEY].iloc[0],
        }
        if "score_sacrifice" in subset.columns:
            row["clinical_score"] = subset["score_sacrifice"].iloc[0]
        if "day_of_sacrifice" in subset.columns:
            row["day_of_sacrifice"] = subset["day_of_sacrifice"].iloc[0]
        if "sex" in subset.columns:
            row["sex"] = subset["sex"].iloc[0]
        rows.append(row)

    df = pd.DataFrame(rows)
    df["region_name"] = df["region"].map(REGIONS)
    return df.sort_values(["model", "course", "region", "sample_id"]).reset_index(drop=True)


def print_coverage(sample_table: pd.DataFrame) -> None:
    """Print a summary of sample coverage across models, stages, and regions."""
    all_stages = get_all_stages()

    for model_name, stages in all_stages.items():
        table = Table(title=f"{model_name} coverage")
        table.add_column("Stage", style="bold")
        table.add_column("Samples", justify="right")
        table.add_column("Regions", justify="center")
        table.add_column("Cells", justify="right")
        if "clinical_score" in sample_table.columns:
            table.add_column("Score (mean)", justify="right")

        model_rows = sample_table[sample_table["model"] == model_name]

        for stage in stages:
            stage_rows = model_rows[model_rows["stage"] == stage]
            n_samples = len(stage_rows)
            n_cells = stage_rows["n_cells"].sum()
            regions = ",".join(sorted(stage_rows["region"].unique())) if n_samples > 0 else "-"

            row_data = [stage, str(n_samples), regions, f"{n_cells:,}"]

            if "clinical_score" in sample_table.columns and n_samples > 0:
                mean_score = stage_rows["clinical_score"].mean()
                row_data.append(f"{mean_score:.1f}")
            elif "clinical_score" in sample_table.columns:
                row_data.append("-")

            style = "red" if n_samples == 0 else ("yellow" if n_samples < 3 else "green")
            table.add_row(*row_data, style=style)

        console.print(table)
        console.print()


# ---------------------------------------------------------------------------
# Sample access
# ---------------------------------------------------------------------------


def get_samples_for_stage(
    adata: ad.AnnData,
    stage: str,
    model_name: str | None = None,
) -> ad.AnnData:
    """Get all cells for a specific disease stage.

    If model_name is given (e.g. "MOG"), also filters by model.
    """
    mask = adata.obs[STAGE_KEY] == stage
    if model_name:
        model_label = {v: k for k, v in MODEL_LABELS.items()}.get(model_name, model_name)
        mask = mask & (adata.obs[MODEL_KEY] == model_label)
    return adata[mask].copy()


def get_transition_data(
    adata: ad.AnnData,
    model_name: str,
    from_stage: str,
    to_stage: str,
) -> tuple[ad.AnnData, ad.AnnData]:
    """Get data for a specific transition (e.g., PEAK1 -> REMISSION1)."""
    from_data = get_samples_for_stage(adata, from_stage, model_name)
    to_data = get_samples_for_stage(adata, to_stage, model_name)
    return from_data, to_data
