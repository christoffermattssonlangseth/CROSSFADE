"""Embed the full dataset with Novae in combined mode.

Processes all 107 samples together so that:
- Spatial domain assignments are consistent across samples
- Batch correction (optimal transport) is applied
- Embeddings are directly comparable

Usage:
    python scripts/embed_combined.py
    python scripts/embed_combined.py --accelerator mps    # use Metal on Apple Silicon
    python scripts/embed_combined.py --domain-level 15    # fewer, coarser domains
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from temporal_foundation.config import EmbeddingConfig
from temporal_foundation.data import load_adata
from temporal_foundation.embeddings import compute_novae_embeddings_combined, save_embedded


def main():
    parser = argparse.ArgumentParser(description="Embed full dataset with Novae (combined mode)")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input h5ad (default: config.DATA_PATH)")
    parser.add_argument("--output", type=str, default="data/embedded_combined.h5ad",
                        help="Output path for embedded h5ad")
    parser.add_argument("--model-id", type=str, default="MICS-Lab/novae-mouse-0",
                        help="Novae model ID")
    parser.add_argument("--accelerator", type=str, default="cpu",
                        help="Device: cpu, mps, or cuda")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 for mps, can be >0 for cuda)")
    parser.add_argument("--domain-level", type=int, default=30,
                        help="Novae domain granularity (higher = more domains)")
    parser.add_argument("--neighbor-radius", type=float, default=50.0,
                        help="Max edge length in um for spatial graph")
    parser.add_argument("--no-batch-correct", action="store_true",
                        help="Skip batch correction")
    args = parser.parse_args()

    config = EmbeddingConfig(
        novae_model_id=args.model_id,
        novae_technology="xenium",
        novae_domain_level=args.domain_level,
        neighbor_radius=args.neighbor_radius,
        remove_long_links=True,
        long_link_percentile=99.0,
        accelerator=args.accelerator,
        num_workers=args.num_workers,
        batch_correct=not args.no_batch_correct,
    )

    print(f"Config: {config}")
    print(f"Output: {args.output}")

    # Load
    adata = load_adata(args.input)

    # Embed
    t0 = time.time()
    adata = compute_novae_embeddings_combined(adata, config)
    elapsed = time.time() - t0
    print(f"\nEmbedding took {elapsed / 60:.1f} minutes")

    # Summary
    print(f"\nResult:")
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    novae_obs = [c for c in adata.obs.columns if "novae" in c.lower()]
    print(f"  novae obs columns: {novae_obs}")
    if "novae_domains" in adata.obs.columns:
        print(f"  Domains: {adata.obs['novae_domains'].nunique()}")
    if "novae_latent_corrected" in adata.obsm:
        print(f"  Batch-corrected latent dim: {adata.obsm['novae_latent_corrected'].shape[1]}")
    elif "novae_latent" in adata.obsm:
        print(f"  Latent dim: {adata.obsm['novae_latent'].shape[1]}")

    # Save
    save_embedded(adata, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
