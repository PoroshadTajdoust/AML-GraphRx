# Supplied analysis runner

normalization_aligned_ablation.py is based on the user-supplied source preserved byte-for-byte in provenance/user_supplied_source.txt. The associated notebook is a newly assembled, unexecuted Colab wrapper, not the original executed notebook.

## What was corrected

The original final CSV inventory includes sensitivity_outputs_manifest.csv if the analysis is run again after completion. Its write-once check then rejects the changed manifest. The runnable copy excludes that manifest from its own inventory. No normalization, seed, threshold, clustering, matching or scientific calculation was changed.

## Required inputs

The script retains the author's original absolute Colab paths and strict input-hash checks. It requires the frozen audit files listed near its start, the original discovery ledger and rank matrix, and all nine raw gene_embedding32.npy files in the original per-arm/per-seed folders. The results folder supplied with this repository is insufficient by itself to run the full re-evaluation.

The runner evaluates existing trained embeddings; it is not an encoder-training script. It contains the projection, graph construction, Leiden partitioning, edge-perturbation recurrence assessment and pairwise matching functions. Do not change seed values or bypass fidelity checks to obtain a successful run.

The source depends on NumPy, pandas, SciPy, scikit-learn, python-igraph and leidenalg. The original source did not pin their complete historical versions. The separately pinned requirements-verification.txt belongs to the verification utilities, not to a proven historical clustering environment.

## Checks performed

Syntax and the recorded arm, seed and resampling settings matched. The uploaded matching function reproduced every saved field for all nine within-arm seed pairs from the provided assignment caches. The corrected inventory matches the existing output manifest. Projection, clustering and encoder training were not rerun during this review.

See provenance/runner_review.json for the source hashes, checks and limits.
