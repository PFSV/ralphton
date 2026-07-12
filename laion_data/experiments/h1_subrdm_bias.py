"""H1 — is the paper's 100-image sub-RDM sampler an unbiased estimator of the full RDM?

Every RSA number in the paper is a mean over 100x100 sub-RDMs (4,950 pairs each), never the
full RDM. The reproduction plan lists this as hidden assumption H1 and notes that the paper
gives NO bias analysis. This matters directly for replication: if the sampler is biased, a
"correct" full-RDM reimplementation will legitimately disagree with the published numbers,
and a replicator would wrongly conclude they had made an error.

For subj01 the full RDM has 49,995,000 pairs. Computing it with scipy's `pdist` is
prohibitively slow (single-threaded). But correlation distance is just a Gram matrix of
z-scored rows:

    1 - corr(x_i, x_j) = 1 - <z_i, z_j>,   with z = (x - mean(x)) / ||x - mean(x)||

so the whole RDM is a single GEMM, which an A100 does in under a second.

Run:  python -m experiments.h1_subrdm_bias
"""

from __future__ import annotations

import numpy as np
import torch

from src.revision import config as C
from src.revision import embeddings as E
from src.revision import nsd_data, roi, rsa

DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"


def full_rdm(X: np.ndarray, iu: torch.Tensor) -> torch.Tensor:
    """Condensed correlation-distance RDM over ALL pairs, via one GEMM."""
    Z = torch.as_tensor(X, device=DEVICE, dtype=torch.float32)
    Z = Z - Z.mean(1, keepdim=True)
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-8)
    G = 1.0 - (Z @ Z.T)
    return G[iu[0], iu[1]]


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main(subj: str = "subj01") -> None:
    betas = roi.load_betas(subj)
    mask = roi.streams_mask()
    emb = E.load("captions")
    _, keep = nsd_data.get_conditions_3rep(subj)
    M = emb[keep]
    n = len(keep)

    iu = torch.triu_indices(n, n, offset=1, device=DEVICE)
    model_full = full_rdm(M, iu)

    print("H1 — the 100-image sub-RDM sampler vs the FULL RDM")
    print(f"{subj}: {n} images -> full RDM = {n * (n - 1) // 2:,} pairs "
          f"(the paper only ever uses 4,950 at a time)\n")
    print(f"{'ROI':<9} {'full-RDM r':>12} {'sampled r':>12} {'bias':>10} {'% error':>9}")

    splits = rsa.make_splits(subj)
    for r in C.STREAMS_MAIN_ROIS:
        X = roi.roi_patterns(betas, mask, r).astype(np.float32)
        full = pearson(full_rdm(X, iu), model_full)
        sampled = float(np.mean([rsa.corr_rdms(rsa.rdm(X[i]), rsa.rdm(M[i])) for i in splits]))
        bias = sampled - full
        print(f"{r:<9} {full:12.4f} {sampled:12.4f} {bias:+10.4f} {100 * bias / full:8.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
