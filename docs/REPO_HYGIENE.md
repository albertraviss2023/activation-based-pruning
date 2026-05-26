# Repo Hygiene

This repo contains both source code and research artifacts. The source should
stay small and reviewable; data, checkpoints, generated plots, and one-off local
copies should stay out of git.

## Keep In Git

- `src/reducnn/`
- `tests/`
- `examples/`
- `docs/`
- `ui/`
- `custom_methods/README.md`
- small reproducible custom-method examples
- configuration files such as `pyproject.toml`, `docker-compose.yml`, and
  `.dockerignore`

## Keep Out Of Git

- datasets: `data/`
- model artifacts: `saved_models/`, `checkpoints/`, `my_models/`
- generated run outputs: `outputs/`, `exports/`
- thesis/report artifacts: `report_artifacts/`, `reports/`, `thesis/`
- legacy scratch material: `Legacy/`
- credentials: `kaggle.json`, `.env`
- generated visual artifacts: `*.png`, `*.gif`, `*.html`, `*.pptx`, `*.pdf`
- duplicate sync files such as `README (1).md` or `notebook (1).ipynb`
- OS and notebook cache files such as `desktop.ini`, `__pycache__/`, and
  `.ipynb_checkpoints/`

These patterns are covered by `.gitignore` and `.dockerignore`.

## Before Committing

Run:

```bash
git status --short
python -m compileall src ui custom_methods
pytest
```

Review unexpected modified notebooks carefully. Notebook diffs can hide large
output changes, so clear outputs before committing when the output itself is not
part of the research record.

## Cleaning Generated Python Caches

Python cache folders are safe to delete:

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Directory -Filter .pytest_cache | Remove-Item -Recurse -Force
```

Do not delete datasets or checkpoints unless you are certain they are backed up
or reproducible.
