## Updated normalization aligned ablation

The current BMC Bioinformatics revision reports the normalization-aligned message-ablation readout in Figure 3. The previous raw-latent message-ablation outputs are retained as historical results. See [normalized_ablation_v1](normalized_ablation_v1/README.md) for all three paired initializations, original result tables, hashes, assignment caches, and tested verification and plotting commands. This addition introduces no new scientific analysis beyond the reported normalization sensitivity. The author-supplied analysis runner and a Colab notebook assembled from it are included under analysis/. A documented manifest-only restart fix leaves scientific calculations unchanged. Full re-execution requires the frozen expression inputs, all nine raw embeddings, and the historical dependency versions; the notebook is supplied without execution outputs.

```bash
python -m pip install -r normalized_ablation_v1/requirements-verification.txt
python normalized_ablation_v1/verify_normalized_ablation.py
python normalized_ablation_v1/plot_normalized_ablation.py
```
