"""Disease model definitions and project constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = Path("/Volumes/processing2/RRmap/data/rrmap.companion.ready.h5ad")

# ---------------------------------------------------------------------------
# Disease models
# ---------------------------------------------------------------------------

# model column in adata.obs
MODEL_LABELS = {
    "CHRONIC": "MOG",
    "RELAPSE REMITTING": "PLP",
}

# Stage column provides unified biological staging across both models.
# Some stages are shared (PEAK1 has both MOG and PLP samples), others
# are model-specific.
MODELS = {
    "MOG": {
        "baseline": "MOG CFA",
        "stages": [
            "NONSYMPTOM",
            "OS1",
            "PEAK1",
            "LONG",
        ],
    },
    "PLP": {
        "baseline": "PLP CFA",
        "stages": [
            "ONSET1",
            "ONSET2",
            "MONOPHASIC",
            "PEAK1",
            "REMISSION1",
            "PEAK2",
            "REMISSION2",
            "PEAK3",
        ],
    },
}

# Stages present in both models (enables cross-model comparison)
SHARED_STAGES = {"PEAK1"}

# Column names in adata.obs
MODEL_KEY = "model"       # "CHRONIC" or "RELAPSE REMITTING"
STAGE_KEY = "stage"       # unified biological staging (primary)
COURSE_KEY = "course"     # original experimental labels (reference)
REGION_KEY = "region"     # "C", "L", "T"
SAMPLE_KEY = "sample_id"  # unique sample identifier

# Spinal cord regions
REGIONS = {"C": "cervical", "L": "lumbar", "T": "thoracic"}

# Cell type annotation columns (finest to broadest)
CELL_TYPE_KEYS = {
    "L1": "anno_L1",   # Glia, Myeloid, Neuron, Vascular, ...
    "L2": "anno_L2",   # Oligodendrocyte, Microglia, Astrocyte, ...
    "L3": "anno_L3",   # Myelinating Oligodendrocyte, CD4+ T Cell, ...
}


def model_short_name(model_label: str) -> str:
    """Convert adata model label to short name: 'CHRONIC' -> 'MOG'."""
    return MODEL_LABELS.get(model_label, model_label)


def get_all_stages() -> dict[str, list[str]]:
    """Return the full ordered stage list per model (baseline + stages)."""
    return {
        name: [cfg["baseline"]] + cfg["stages"]
        for name, cfg in MODELS.items()
    }


def get_transition_pairs(model_name: str) -> list[tuple[str, str]]:
    """Return consecutive (from_stage, to_stage) pairs for a disease model."""
    stages = get_all_stages()[model_name]
    return list(zip(stages[:-1], stages[1:]))


# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------

SPATIAL_KEY = "spatial"
N_GENES = 5101


# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    """Configuration for pretrained encoder inference."""

    encoder: str = "novae"
    novae_model_id: str = "MICS-Lab/novae-mouse-0"
    novae_technology: str = "xenium"
    novae_domain_level: int = 7

    # Spatial graph construction
    neighbor_radius: float | None = 200.0   # max edge length in um (None = no cutoff)
    remove_long_links: bool = True          # use cellcharter to remove outlier edges
    long_link_percentile: float = 99.0      # percentile threshold for long link removal

    # Device for inference
    accelerator: str = "cpu"  # "cpu" is fastest for Novae's small-subgraph workload on Apple Silicon
    num_workers: int = 0       # data loading workers (must be 0 for MPS, can be >0 for CUDA)

    # Batch correction across samples
    batch_correct: bool = True

    # Key names for stored results
    latent_key: str = "novae_latent"
    domain_key: str = "novae_domains"


# ---------------------------------------------------------------------------
# Aggregation configuration
# ---------------------------------------------------------------------------

@dataclass
class AggregationConfig:
    """Configuration for niche/region-level aggregation."""

    method: str = "spatial_domains"  # "spatial_domains", "grid"

    # Grid binning parameters
    grid_size_um: float = 100.0

    # Spatial domain parameters (uses Novae domain assignments)
    domain_key: str = "novae_domains"

    # What to aggregate
    embedding_key: str = "novae_latent"
    include_composition: bool = True
    cell_type_key: str = "anno_L2"
