# Packaging changes

Scientific constants, seeds, thresholds, optimizer settings and retained analytical functions were preserved. `provenance/source_lineage.json` identifies exact notebook/cell hashes and records function-level AST comparisons.

1. Scripts have descriptive filenames and standard Python formatting.
2. Drive mounting is performed in the launcher notebook. Project/cache locations are configurable through environment variables.
3. In-run package installation is replaced by an explicit dependency error. A print fallback is supplied for notebook `display()` calls.
4. Frozen assignment combines the completed prefix of the original cell with its executed dtype-repair continuation, excluding the failed merge. The continuation's scientific code is unchanged.
5. Clinical comparison combines the verified analysis-unit merge, final statistical table and executed odds-ratio-label correction. The earlier failed sample-level joins are excluded.
6. Pathway contrast ends after the reported expression and shared-pathway outputs. Its unreported network-bridge candidate extension is excluded. The unused prospective-cohort helper in the corrective transfer script is excluded.
7. Old supervised classifiers, alternative encoder experiments, the later exploratory branch not described by these manuscripts, and the proposed normalization-aligned rerun are excluded.
8. The existing scientific figure code and the underlying source tables are retained; no plot points or statistical values are simulated.

Historical analysis scripts retain their required parent checks and their existing audit outputs. Some parent artifacts originate in older preparation stages; those prerequisites are documented rather than recreated by an invented pipeline. Original audit identifiers remain in filenames to preserve traceability.
