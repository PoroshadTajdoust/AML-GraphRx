# Frozen inputs and execution boundary

The analysis code expects the existing AML_GraphRx project structure. It does not synthesize data or reconstruct missing frozen inputs from approximate substitutes.

## Discovery and ablations

Under `data/patient_graphs/20260823T123040Z/`:

- `20260823T123040Z_patient_rank_raw_float64.npy`
- `20260823T123040Z_gene_presence_mask_bool.npy`
- `20260823T123040Z_shared_locked_topology.pt`
- `20260823T123040Z_node_index.csv`
- `20260823T123040Z_analysis_unit_index.csv` for assignment assessment

Under `results/audit/`, retain the frozen discovery ledger (452), original internal split (453), reconciled labels (317), and the relevant protocol/manifest chain. The discovery data comprise 1,733 AML profiles in eight specified cohorts and 9,554 genes. The frozen GNN basis (483), saved projection (486), MNN edges (488), raw assignments (489) and original manifest (497) are required by the ablation fidelity checks. Cross-seed comparison additionally requires the nine saved `gene_embedding32.npy` outputs and the message-ablation run tables (760–767).

## Signatures, pathway definition and clinical context

Enrichment uses the original frozen GMT snapshots and Entrez-to-symbol map (428), state meta-statistics and the assessment outputs. Downloading a current gene-set collection is not an equivalent substitute for the recorded Reactome 2022 and GO BP 2025 snapshots. The code checks the historical catalog sizes and hashes.

The clinical script requires `config/verified_clinical_analysis_unit_map_v1.csv`; it does not choose the first available sample-level table. The successful one-to-one merge and final odds-ratio labels were recovered from the earlier complete notebook because the new copy ends before those cells.

The frozen pathway definition (605) is included in `source_tables/context/`. Retain its full component membership and gene weights. Scoring is not based on re-fitting external clusters.

## External RNA-seq and contextual analyses

BeatAML inputs are obtained from the documented [BeatAML2 resource](https://biodev.github.io/BeatAML2/) under its data-access terms. The original external cache is `/content/beataml2.0_data_r7b`; override it with `AML_GRAPHRX_BEATAML_DIR`. Required files depend on the selected stage: expression, clinical workbook, WES mutations, six-cell-score workbook, or response data. External scoring also expects the original pre-scoring protocol (608) and source audit (609).

The global-response script validates the genuine response payload: `beataml_probit_curve_fits_v4_dbgap.txt`, 19,047,078 bytes, SHA-256 `d4bc5f0d91f66314107411e0f2511adc987e29df5b83d9b03df56d3d12928314`. A Git LFS pointer is rejected. The uploaded notebook stopped when that payload was missing; the manuscript reports its subsequent completed execution, whose actual summary and bootstrap tables are included.

DepMap analyses require the authorized frozen Public 26Q3 expression, dependency and model files and the pre-existing 533-gene candidate family. The supplementary focused subset has 33 genes. The candidate-family inputs are not rebuilt from a later database release.

## Data identity

`config/frozen_file_hashes.json` consolidates recorded SHA-256 identities. Historical manifests retain their original paths; compare by basename where the original code does so. Do not rewrite a hash to accept a different input. The included source tables support immediate verification and figure reconstruction; they are not all raw inputs required to retrain every model.

Upstream expression-preparation audits (131, 136, 147, 153, 158, 161, 163, 169) and network-source audits (170, 171, 184, 250) are included. The release does not claim a new end-to-end re-execution of that historical preparation.
