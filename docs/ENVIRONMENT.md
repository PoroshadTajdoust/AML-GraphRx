# Environment provenance

The message-ablation notebook output records **PyTorch 2.11.0+cu130** on CUDA. The project notebooks explicitly request **PyTorch Geometric 2.8.0**; the enrichment runner requires **GSEApy 1.3.0**. The upstream network acquisition table records R 4.6.1, Bioconductor 3.23 and OmnipathR 4.0.0. These are observed project records, not a claim that a complete original environment lock exists.

`requirements-replay.txt` pins the environment used for the release's CPU table verification and figure reconstruction. It is **not** the original GPU training environment. `requirements-analysis.txt` lists the analysis dependencies; most historical versions were not fully recorded, so unknown versions are deliberately not fabricated. The earlier project's scikit-learn 1.6.1 installation is not sufficient evidence for the version used in every later notebook.

In the original Colab environment, capture the actual installed environment before changing packages:

```python
import sys, subprocess, pathlib
out = pathlib.Path('/content/AML_GraphRx_environment.txt')
out.write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
print(sys.version)
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print(out)
```

Use the appropriate CUDA-compatible PyTorch installation for GPU training, then install the remaining requirements. The released scripts do not install packages during a scientific run. Set `AML_GRAPHRX_ROOT` for project files and, if necessary, `AML_GRAPHRX_BEATAML_DIR` for the BeatAML cache. Default values retain the original Colab locations.

Matching seeds alone cannot promise bitwise identity across different PyTorch/CUDA/library versions. No deterministic-kernel switch, new seed, normalization step or optimizer setting has been added to force a different execution to resemble the reported results.
