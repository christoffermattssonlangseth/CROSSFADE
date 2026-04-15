"""Extract cell embeddings using pretrained spatial foundation models."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
from rich.console import Console
from rich.progress import Progress

from .config import EmbeddingConfig, SAMPLE_KEY

console = Console()


# ---------------------------------------------------------------------------
# Novae encoder
# ---------------------------------------------------------------------------


def _build_spatial_graph(adata: ad.AnnData, config: EmbeddingConfig, slide_key: str | None = None) -> None:
    """Build spatial neighbor graph, handling existing coordinates and long links."""
    import novae

    kwargs = {}

    # If spatial coordinates already exist, don't pass technology
    if "spatial" in adata.obsm:
        console.print("Building spatial graph (using existing coordinates)...")
    else:
        console.print(f"Building spatial graph (technology={config.novae_technology})...")
        kwargs["technology"] = config.novae_technology

    # Use slide_key for multi-sample processing
    if slide_key is not None:
        kwargs["slide_key"] = slide_key

    # Set a radius to avoid spurious long-distance edges from Delaunay
    if config.neighbor_radius is not None:
        kwargs["radius"] = config.neighbor_radius

    novae.spatial_neighbors(adata, **kwargs)


def _remove_long_links(
    adata: ad.AnnData,
    config: EmbeddingConfig,
    connectivity_key: str = "spatial_connectivities",
    distance_key: str = "spatial_distances",
) -> None:
    """Remove long-distance edges from the spatial graph.

    Computes a distance threshold at the given percentile of all nonzero edge
    distances, then removes any edge exceeding that threshold from both the
    distance and connectivity matrices.
    """
    if distance_key not in adata.obsp:
        console.print(f"[yellow]No '{distance_key}' in adata.obsp — skipping long link removal[/yellow]")
        return

    dists = adata.obsp[distance_key]
    threshold = np.percentile(dists.data, config.long_link_percentile)

    n_before = dists.nnz

    # Zero out long edges in distances
    dists.data[dists.data > threshold] = 0
    dists.eliminate_zeros()

    # Sync connectivities: keep only edges that survive in distances
    if connectivity_key in adata.obsp:
        surviving = dists.copy()
        surviving.data[:] = 1
        adata.obsp[connectivity_key] = adata.obsp[connectivity_key].multiply(surviving)
        adata.obsp[connectivity_key].eliminate_zeros()

    n_after = dists.nnz
    console.print(
        f"Removed long links: {n_before - n_after} edges removed "
        f"(threshold={threshold:.1f}um, percentile={config.long_link_percentile})"
    )


def compute_novae_embeddings_combined(
    adata: ad.AnnData,
    config: EmbeddingConfig | None = None,
    slide_key: str = SAMPLE_KEY,
) -> ad.AnnData:
    """Compute Novae embeddings on the full combined dataset at once.

    Processing all samples together ensures:
    - Consistent spatial domain assignments across samples
    - Batch correction is possible
    - Embeddings are directly comparable

    Args:
        adata: Combined AnnData with all samples
        config: Embedding configuration
        slide_key: obs column identifying individual samples/slides

    After running:
        adata.obsm["novae_latent"]           -- cell-neighborhood embeddings
        adata.obsm["novae_latent_corrected"]  -- batch-corrected embeddings (if enabled)
        adata.obs["novae_domains"]            -- spatial domain assignments
    """
    import novae

    config = config or EmbeddingConfig()

    # Build spatial graph with slide_key so Novae doesn't connect cells across samples
    _build_spatial_graph(adata, config, slide_key=slide_key)

    # Remove long-distance edges
    if config.remove_long_links:
        _remove_long_links(adata, config)

    # Load pretrained model
    console.print(f"Loading Novae model: {config.novae_model_id}")
    model = novae.Novae.from_pretrained(config.novae_model_id)

    # Compute representations on GPU
    console.print(f"Computing cell embeddings ({adata.n_obs:,} cells, accelerator={config.accelerator})...")
    model.compute_representations(adata, zero_shot=True, accelerator=config.accelerator, num_workers=config.num_workers)

    # Batch correction across samples
    if config.batch_correct:
        console.print("Applying batch correction...")
        novae.batch_effect_correction(adata)

    # Assign spatial domains
    console.print(f"Assigning spatial domains (level={config.novae_domain_level})...")
    domain_key = model.assign_domains(adata, level=config.novae_domain_level)

    # Rename to our standard keys if different from Novae defaults
    if config.latent_key != "novae_latent" and "novae_latent" in adata.obsm:
        adata.obsm[config.latent_key] = adata.obsm["novae_latent"]
    if domain_key != config.domain_key:
        adata.obs[config.domain_key] = adata.obs[domain_key]

    n_domains = adata.obs[config.domain_key].nunique()
    latent_key = "novae_latent_corrected" if config.batch_correct and "novae_latent_corrected" in adata.obsm else config.latent_key
    console.print(
        f"Done: {adata.obsm[latent_key].shape[1]}d embeddings, "
        f"{n_domains} spatial domains, {adata.n_obs:,} cells"
    )
    return adata


def compute_novae_embeddings(
    adata: ad.AnnData,
    config: EmbeddingConfig | None = None,
) -> ad.AnnData:
    """Compute embeddings for a single sample. Prefer compute_novae_embeddings_combined."""
    import novae

    config = config or EmbeddingConfig()

    _build_spatial_graph(adata, config)

    if config.remove_long_links:
        _remove_long_links(adata, config)

    console.print(f"Loading Novae model: {config.novae_model_id}")
    model = novae.Novae.from_pretrained(config.novae_model_id)

    console.print(f"Computing cell embeddings (accelerator={config.accelerator})...")
    model.compute_representations(adata, zero_shot=True, accelerator=config.accelerator, num_workers=config.num_workers)

    console.print(f"Assigning spatial domains (level={config.novae_domain_level})...")
    domain_key = model.assign_domains(adata, level=config.novae_domain_level)

    if config.latent_key != "novae_latent" and "novae_latent" in adata.obsm:
        adata.obsm[config.latent_key] = adata.obsm["novae_latent"]
    if domain_key != config.domain_key:
        adata.obs[config.domain_key] = adata.obs[domain_key]

    n_domains = adata.obs[config.domain_key].nunique()
    console.print(
        f"Done: {adata.obsm[config.latent_key].shape[1]}d embeddings, "
        f"{n_domains} spatial domains"
    )
    return adata


def compute_novae_batch(
    samples: dict[str, ad.AnnData],
    config: EmbeddingConfig | None = None,
) -> dict[str, ad.AnnData]:
    """Compute Novae embeddings for all samples individually."""
    import novae

    config = config or EmbeddingConfig()

    # Load model once, reuse for all samples
    console.print(f"Loading Novae model: {config.novae_model_id}")
    model = novae.Novae.from_pretrained(config.novae_model_id)

    with Progress() as progress:
        task = progress.add_task("Computing embeddings...", total=len(samples))
        for sample_id, adata in samples.items():
            progress.update(task, description=f"Embedding {sample_id}...")

            _build_spatial_graph(adata, config)
            if config.remove_long_links:
                _remove_long_links(adata, config)

            model.compute_representations(adata, zero_shot=True, accelerator=config.accelerator, num_workers=config.num_workers)
            domain_key = model.assign_domains(adata, level=config.novae_domain_level)

            if config.latent_key != "novae_latent" and "novae_latent" in adata.obsm:
                adata.obsm[config.latent_key] = adata.obsm["novae_latent"]
            if domain_key != config.domain_key:
                adata.obs[config.domain_key] = adata.obs[domain_key]

            progress.advance(task)

    total_cells = sum(a.n_obs for a in samples.values())
    console.print(f"Embedded {len(samples)} samples, {total_cells:,} cells total")
    return samples


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------


def save_embedded(
    adata: ad.AnnData,
    output_path: str | Path,
) -> None:
    """Save the combined embedded AnnData to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output_path)
    console.print(f"Saved embedded data to {output_path}")


def load_embedded(path: str | Path) -> ad.AnnData:
    """Load a previously embedded AnnData."""
    path = Path(path)
    adata = ad.read_h5ad(path)
    console.print(f"Loaded {adata.n_obs:,} cells from {path}")
    return adata


# Kept for backwards compatibility
def save_embeddings(
    samples: dict[str, ad.AnnData],
    output_dir: str | Path,
    config: EmbeddingConfig | None = None,
) -> None:
    """Save per-sample embedded AnnData objects to disk."""
    config = config or EmbeddingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id, adata in samples.items():
        if config.latent_key not in adata.obsm:
            console.print(f"[yellow]Skipping {sample_id}: no embeddings found[/yellow]")
            continue
        path = output_dir / f"{sample_id}.h5ad"
        adata.write_h5ad(path)

    console.print(f"Saved {len(samples)} samples to {output_dir}")


def load_embeddings(
    embedding_dir: str | Path,
    config: EmbeddingConfig | None = None,
) -> dict[str, ad.AnnData]:
    """Load per-sample embedded AnnData objects."""
    config = config or EmbeddingConfig()
    embedding_dir = Path(embedding_dir)

    samples: dict[str, ad.AnnData] = {}
    for path in sorted(embedding_dir.glob("*.h5ad")):
        adata = ad.read_h5ad(path)
        if config.latent_key not in adata.obsm:
            console.print(f"[yellow]WARNING: {path.name} has no '{config.latent_key}' in obsm[/yellow]")
        sample_id = path.stem
        samples[sample_id] = adata

    console.print(f"Loaded {len(samples)} embedded samples")
    return samples
