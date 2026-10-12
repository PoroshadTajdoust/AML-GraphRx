# Random-number settings and manuscript concordance

The explicit seed values in the uploaded main manuscript and Supplementary Tables S4A, S4B and S5 agree with the supplied code and saved run tables. The first two rows of Table S4A use the wording “Locked / fixed” rather than a printed integer. Both refer to 20261004 in the executed code; for explicit numerical consistency, replace those two labels with 20261004. No reported result or random seed needs changing. Unspecified implementation offsets are documented below; they are not silently substituted for the reported initialization seeds.

| Operation | Exact setting | Record in manuscript / supplement |
|---|---|---|
| Original gene encoder | 20261004 | Methods 2.3 |
| Original discovery Leiden partition | 20261004 | Code detail for Methods 2.4 |
| Discovery edge-subsampling generator | 20261004 + 9000 = 20270004 | 30 perturbations; code detail |
| Leiden in perturbation b | 20261004 + b + 1; b = 0,…,29 | 20261005,…,20261034; Methods 2.4 |
| Expression PCA randomized SVD | 20261004 | Methods 2.5; Table S4A (PCA shown as Locked / fixed) |
| Five Gaussian bases | 20261011, 20261012, 20261013, 20261014, 20261015 | Methods 2.5; Table S4A |
| Five gene-row shuffles | 20261021, 20261022, 20261023, 20261024, 20261025 | Methods 2.5; Table S4A |
| Three paired encoder initializations | 20261031, 20261101, 20261102 | Methods 2.5; Tables S4B and S5; Figure 3 caption |
| Rewiring by relation | 20261029 + r; r = 0,1,2,3 in signaling, PPI, TF regulation, kinase–substrate order | One rewired topology; seed is an additional code detail |
| All ablation downstream Leiden / recurrence | Same 20261004, +9000 and +b+1 settings above | Fixed downstream operations in Methods 2.5–2.6 |
| Positive training-edge subsampling | training seed + 100 × epoch + r | Original encoder and message ablation; epoch = 1,…,30 |
| Negative-edge sampling | training seed + 1000 × epoch + r | Original encoder and message ablation |
| Reactome preranked enrichment | 20261005 + state_number; state_number = 1,…,6 | Methods 2.7; 1,000 permutations |
| GO BP preranked enrichment | 20261005 + 100 + state_number; state_number = 1,…,6 | Methods 2.7; 1,000 permutations |
| Three-cohort assignment bootstrap | 20261005 + 1000 × gse_number; gse_number = 1,2,3 | 1,000 resamples; Table S3 parent analysis |
| Corrective LOCO discovery | 20261009 + 1000 × fold_no; fold_no = 1,…,8 | Table S3; fold_no follows the source enumeration |
| All-eight-cohort corrective candidate | 20261009 + 99000 | Descriptive candidate in Table S3 lineage |
| BeatAML pooled subject bootstrap | 20261009 | Methods 2.8; 1,000 resamples |
| Rank-matched paired gene-label null | 20261010 | Methods 2.8; 1,000 permutations |
| Secondary unrestricted gene-label null | 20261011 | Methods 2.8; 1,000 permutations |
| Mutation ridge fivefold shuffle | 20261010 | Methods 2.9; Table S3 |
| Six-cell-score ridge fivefold shuffle | 20261010 | Methods 2.9; Supplementary methods |
| Conditional rank bootstrap and residual permutation | 20261010, one shared sequential NumPy generator | Methods 2.9; Supplementary methods; 2,000 of each per wave |
| Focused DepMap bootstrap | 20261010 | Supplementary dependency analysis; 1,000 resamples |
| Global response bootstrap | 20261010 | Supplementary response analysis; 2,000 subject-cluster resamples |
| Global response cross-validation | First 8 hex digits of SHA-256(response_id + "\|" + "20261010"), interpreted as an integer; identical for baseline and augmented fits of each endpoint | Supplementary response analysis |

The exact enrichment state enumeration, assignment-cohort enumeration, LOCO enumeration and global-response CV call are retained in the source. No reseeding is inserted between conditional bootstrap and permutation loops: Waves1+2 bootstrap → Waves1+2 permutation → Waves3+4 bootstrap → Waves3+4 permutation. Restarting the generator for each loop would change the stored draws despite showing the same base seed.

`config/random_settings.json` extracts the literal seed and resampling constants from every released analysis. The pathway freezing, fixed external scoring and clinical-annotation tests have deterministic components that do not require a random seed. The bridge script's historical seed declaration belongs to the omitted graph-null extension; the retained expression/pathway comparison is deterministic.

Reproducibility paragraph suitable for supplementary methods:

> Integer random-number settings were retained from the executed analyses. Original discovery and the expression-derived PCA comparator used 20261004. Gaussian controls used 20261011–20261015 and gene-row shuffles used 20261021–20261025. Neighbor-message comparisons used paired initialization seeds 20261031, 20261101 and 20261102; relation-wise rewiring used 20261029 plus the relation index, with one rewired topology shared across those fits. Downstream partitioning and edge perturbations retained the original discovery settings. The pooled external bootstrap used 20261009; the matched and unrestricted gene-label nulls used 20261010 and 20261011, respectively. Conditional rank analysis used a single sequential generator initialized at 20261010 for the recorded bootstrap and permutation order. Seed settings and offsets are supplied with the code; no seed was selected on the basis of a favorable result.
