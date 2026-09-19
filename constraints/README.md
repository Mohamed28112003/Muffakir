# Reproducible dependency constraints

Published package metadata intentionally uses compatible version ranges. The
`py311.txt`, `py312.txt`, `py313.txt`, and `py314.txt` files pin complete
`Muffakir[all]` environments for CI and reproducible deployments.

Regenerate a file with `uv` whenever release dependencies change:

```bash
uv pip compile pyproject.toml --extra all --python-version 3.13 --output-file constraints/py313.txt
```

Install a profile against a matching constraint file with:

```bash
python -m pip install -c constraints/py313.txt ".[standard]"
```

Constraints are environment locks, not package metadata; users remain free to
resolve newer compatible releases inside the ranges declared in
`pyproject.toml`.
