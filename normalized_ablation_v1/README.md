# Normalization aligned graph message ablation

This addition supports the revised AML-GraphRx BMC Bioinformatics manuscript, Figure 3 and Supplementary Tables S4A, S4B, S5, S8, and S9. It contains authentic archived outputs and new verification and plotting utilities. It also includes the author-supplied normalization runner and an unexecuted Colab notebook assembled from it. The complete frozen input tree is required for a full re-evaluation.

## Reproduce the checks and figure

From this folder, using Python 3.10 or later:

```bash
python -m pip install -r requirements-verification.txt
python verify_normalized_ablation.py
python plot_normalized_ablation.py
```

No GPU, Drive credentials, or expression matrix is required for these commands. The requirements pin the environment used to test these verification utilities, not the historical encoder-training environment. The plot command writes 600 dpi PNG, vector SVG and vector PDF to `generated/`. The verifier recalculates coverage, community counts, joint membership, all-profile partition ARI and size-weighted Hungarian-matched Jaccard from the nine saved assignment caches. It checks file hashes and recorded embedding identities. A matching identity string does not independently validate unavailable embedding bytes.

## Experimental scope

Nine previously trained embeddings were evaluated after float32 column centering and standard-deviation scaling followed by gene-row L2 normalization exactly once. The original frozen discovery basis was not normalized again. The same cohort projection and recurrence criteria were retained. The three encoder seeds were 20261031, 20261101 and 20261102, corresponding to Initialization A, B and C in the manuscript. Seed integers are not execution dates. One relation-wise degree-preserving rewired source topology was reused across all three initializations. All conditions retained biological decoder targets; the no-message condition retained trainable gene coordinates and root transformations. No encoder was retrained for the aligned readout.

| Condition | Mean recurrent coverage | Median matched Jaccard | Evaluable Jaccard pairs |
|---|---:|---:|---:|
| Biological messages | 77.96% | 0.391 | 3 |
| Degree-preserving rewiring | 60.05% | 0.267 | 3 |
| No neighbor messages | 22.49% | 0.000 | 2 |

Biological coverage exceeded rewired coverage in two of three paired fits. Rewiring exceeded biological messages by one profile in Initialization B. One no-message pair has no qualifying communities in either fit: its matched Jaccard is undefined (NA), while joint recurrent coverage is zero. The overlapping pairs are descriptive and are not independent replicates for a superiority test.

## Supplied runner and provenance

The original `results/sensitivity_protocol_and_inputs.json` and manifest are preserved byte-for-byte. Historical output identifiers remain inside these original files to preserve their hashes. Display names use scientific terminology. No extra analyses are introduced.

The author supplied the Python source for this re-evaluation after the initial package review. See analysis/normalization_aligned_ablation.py and analysis/normalization_aligned_ablation.ipynb. The exact supplied bytes are preserved in analysis/provenance/user_supplied_source.txt. The runnable copy contains one documented restart fix: the output manifest is excluded from its own inventory. The notebook is a newly assembled, unexecuted wrapper, not a historical execution record. Its source settings match the saved protocol and its matching function reproduces the archived pairwise results. Full re-execution still requires all nine raw embeddings, the frozen rank matrix and audit files, and the historical dependency versions. This full rerun was not performed in the review. Existing raw-latent message-ablation scripts remain historical analyses.

These outputs concern exploratory recurrence within the discovery data. They do not validate discrete external subtypes, clinical utility, or universal superiority over other representations. The frozen S03/S06 definition and BeatAML scoring were not changed.
