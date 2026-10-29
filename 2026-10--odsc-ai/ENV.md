# Environment

The notebooks run on Python 3.12 with DoWhy 0.14, EconML 0.17, cluster-experiments 0.31 and mlsynth 1.0. There are three ways to get them.

## Colab

Nothing to set up. The first cell of each notebook installs the four packages and downloads `uplift_data.py` from this repo. If Colab asks you to restart the session after the install, restart and run the notebook again from the top.

## uv

Definition: `environment/uv/pyproject.toml` + `environment/uv/uv.lock`.

```bash
cd environment/uv
export UV_PROJECT_ENVIRONMENT=~/.venvs/odsc-ai   # keep the env outside synced folders
uv sync
```

Then start Jupyter from the repo folder (the notebooks import `uplift_data.py` from there):

```bash
~/.venvs/odsc-ai/bin/jupyter lab          # Windows: ~/.venvs/odsc-ai/Scripts/jupyter-lab.exe
```

To use the env from a Jupyter or VS Code you already have, register it as a kernel and pick "Python (odsc-ai)":

```bash
~/.venvs/odsc-ai/bin/python -m ipykernel install --user --name odsc-ai --display-name "Python (odsc-ai)"
```

## conda

Definition: `environment/conda/environment.yml` (conda-forge for the scientific stack, pip for the four causal packages).

```bash
conda env create -f environment/conda/environment.yml
conda activate odsc-ai-2026
jupyter lab
```

## Check that it works

```bash
python -m pytest -q test_uplift_data.py
```

The tests check the data generator: its structure, the answer key, and that each lesson in the notebooks holds on the default seed and across other seeds. Every notebook also prints a fingerprint of the data (`13302.553/5496.520`): if you see the same, you have the same world as everyone else.
