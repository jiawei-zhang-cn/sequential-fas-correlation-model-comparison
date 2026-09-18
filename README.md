# Sequential FAS Correlation-Model Comparison

This repository contains the code, parameters, saved numerical results, and
figures for comparing three correlation models in a sequentially observed
fluid antenna system (FAS):

- Common-scatterer (CS) reference model
- Tx/Rx-side decoupled (TRD) model
- Space-time separable (STS) model

## Structure

```text
├── code/          Analysis and plotting scripts
├── parameters/    Numerical parameters used in the paper
├── results/       Saved numerical results
├── figures/       Generated PDF and PNG figures
├── requirements.txt
└── LICENSE
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Generate figures

To recompute the main outage-decomposition results:

```bash
python code/literature_anchored_outage_decomposition.py
```

```bash
python code/plot_literature_anchored_outage_curves.py
python code/plot_literature_anchored_applicability_map.py
python code/plot_validation_figure.py
```

Generated files are written to `figures/`.

To recompute the validation data before plotting:

```bash
python code/plot_validation_figure.py --recompute
```

## License

MIT License. See [`LICENSE`](LICENSE).
