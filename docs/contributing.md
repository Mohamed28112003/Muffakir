# Contributing

Muffakir welcomes bug reports, documentation corrections, tests, integrations, and
examples that improve Arabic RAG workflows.

## Development setup

```bash
git clone https://github.com/Mohamed28112003/Muffakir.git
cd Muffakir
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Run focused tests for the area you change, then run the relevant regression suite before
opening a pull request. Keep secrets, downloaded models, local vector databases, and
ComposerUI run data out of commits.

## Documentation changes

```bash
pip install -r requirements-docs.txt
mkdocs serve
mkdocs build --strict
```

Write for users first: explain the outcome, provide a small runnable example, and link to
the configuration or API reference only when readers need more detail. Do not add internal
engineering plans under the public documentation navigation.
