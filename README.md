# FedCTI-LLM v3

Calibration-aware federated LoRA fine-tuning of small LLMs for cyber threat intelligence.

| Experiment | Task | Data |
| --- | --- | --- |
| Exp 1 | Report-level extraction of APT, CVE, campaign | Di Tizio APT corpus |
| Exp 2 | Sentence-level mapping to MITRE ATT&CK techniques | TRAM (CTID) + ATT&CK Enterprise v15.1 |

Both experiments share one engine: FFA-LoRA (A frozen, only B trained and averaged), and the
aggregators FedAvg, FedProx and Cal-FedAvg v3. Full design: the "FedCTI-LLM v3 - Methodology
Pipeline" doc.

> Private repository. Keep it private until the provisional patent is filed.

## Layout

```
configs/            base.yaml (shared engine) + per-experiment YAMLs
src/fedcti/         importable code used by every notebook
  config.py         YAML loading/merging, storage root (Drive on Colab)
  paths.py          data / run / results directory layout
  repro.py          seeds, git commit, run metadata
  stages.py         cached stages: finished stages are skipped after a Colab restart
  data/ retrieval/ fl/ eval/
notebooks/
  00_colab_setup.ipynb   run first in every Colab session
  exp1/  exp2/           experiment notebooks
tests/              pytest unit tests (run locally)
docs/               workflow notes
```

Data, adapters, predictions and figures are never committed. They live on Google Drive:

```
/content/drive/MyDrive/fedcti_v3/
  data/<exp>/{raw,clean,splits,clients,gt}/
  runs/<exp>/<method>/<model>/alpha<a>/seed<s>/{adapters,logs,preds}/
  results/
```

## Workflow

1. Code and notebooks are written in this repo locally, committed, and pushed to GitHub.
2. In Colab, run `notebooks/00_colab_setup.ipynb`; it pulls the latest commit and installs the package.
3. Open the experiment notebook from GitHub in Colab and run it; outputs go to Drive.

Details, including the one-time GitHub token setup: [docs/colab_workflow.md](docs/colab_workflow.md).

## Local development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .
pytest
nbstripout --install   # strips notebook outputs on commit
```

Outside Colab, the storage root is `./_local/` (git-ignored); set `FEDCTI_ROOT` to override.
