# Installation

Muffakir 0.3.0 supports Python 3.11 and later. Its default package is intentionally
small; install only the capabilities your application uses.

## Choose an installation profile

```bash
# Public API, configuration, and prompt management
pip install Muffakir

# Local RAG, Composer, and ComposerUI
pip install "Muffakir[standard]"

# ComposerUI server without local ML/vector/database integrations
pip install "Muffakir[ui]"

# Every supported integration
pip install "Muffakir[all]"
```

For a smaller production environment, combine feature extras:

```bash
pip install "Muffakir[rag,local,chroma,openai]"
pip install "Muffakir[datasets,pdf,token,bm25]"
```

| Need | Extra |
| --- | --- |
| Local sentence-transformer embeddings and Hugging Face rerankers | `local` |
| Chroma vector database | `chroma` |
| OpenAI models | `openai` |
| ComposerUI server | `ui` |
| Tabular evaluation datasets | `datasets` |
| BM25 reranking | `bm25` |

## Development installation

```bash
git clone https://github.com/Mohamed28112003/Muffakir_Arabic_RAG.git
cd Muffakir_Arabic_RAG
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Optional dependencies

Muffakir validates optional features when you configure them. If a capability is
missing, it raises `MissingOptionalDependencyError` and tells you which extra to install.
It never installs packages automatically at runtime.

Next: [build your first RAG application](quickstart.md).
