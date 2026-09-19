# Troubleshooting

## ComposerUI cannot bind to port 2811

Windows can reserve a TCP port even when another process is not visibly using it. Start
ComposerUI on another local port:

```bash
muffakir serve --port 2812 --open
```

To inspect reserved TCP port ranges, run PowerShell as Administrator:

```powershell
netsh interface ipv4 show excludedportrange protocol=tcp
```

## An optional dependency is missing

Install the extra named in `MissingOptionalDependencyError`. For example:

```bash
pip install "Muffakir[local,chroma,openai]"
```

## A Hugging Face reranker does not load

`cross_encoder` and `pointwise` require a sequence-classification/cross-encoder-compatible
model. Check the model ID, install the `local` extra, and verify that the model can run on
your selected CPU or CUDA device. Muffakir includes the selected model ID in configuration
errors to make the failing choice visible.

## Retrieval-only evaluation rejects a generation metric

Retrieval-only mode never creates an answer. Use only `recall`, `precision`, `mrr`, and
`ndcg`, or change to `full_rag` before selecting faithfulness, answer correctness, or LLM
Judge Rating.

## LLM Judge Rating returns an evaluation error

The judge must return one exact integer from 1 through 5. Provider failures are surfaced
immediately; malformed, fractional, or out-of-range judge responses are recorded as
sample evaluation errors.

## I cannot find an internal plan in the documentation site

Internal material in `docs/superpowers/` is intentionally excluded from the public Pages
site. It remains available only in the source repository.
