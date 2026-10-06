# FacetPrint

Reference research code for **FacetPrint**, the quantization-robust black-box
LLM fingerprinting method described in *Mind the Precision Gap:
Quantization-Robust Black-Box Fingerprinting of Large Language Models*.

## Overview

FacetPrint has three stages:

1. **Fingerprint construction.** Sample multiple seeded responses from a clean
   task model and recombine their same-position words into one fixed target
   response associated with a private owner code.
2. **Fingerprint embedding.** Jointly optimize ordinary task cross-entropy and
   fingerprint cross-entropy while cycling through BF16, W8A16, W4A16, W8A8,
   and W4A8 numerical views. The shared LoRA update is merged after training.
3. **Black-box verification.** Query a suspect deployment with the private
   prompt and compare its response with the recorded target using strict exact
   matching. The default decision rule accepts at least one match within three
   queries.

## Code

```text
src/facetprint/
  construction.py   fingerprint-pair construction
  quant_lora.py     LoRA and simulated quantization views
  training.py       joint task/fingerprint embedding and LoRA merging
  verification.py   black-box exact-match verification
  cli.py             command-line entry points

configs/
  facetprint.example.yaml

tests/
  test_method.py
```

The implementation is self-contained and does not import modules from the
surrounding experimental repository.

## Research workflow

After adapting the example configuration to the local model and dataset paths,
the three stages are invoked as:

```bash
facetprint construct --config configs/facetprint.example.yaml
facetprint train --config configs/facetprint.example.yaml
facetprint verify --config configs/facetprint.example.yaml
```

Construction writes the private fingerprint record. Training writes a merged
fingerprinted checkpoint. Verification can load that checkpoint or a separately
quantized deployment checkpoint specified in the configuration.

The task dataset is JSONL with `prompt` and `response` fields. The base model is
expected to be an existing task-adapted Hugging Face causal language model.
Deployment quantization backends and benchmark evaluation scripts are outside
this method-only release.

## Notes

- The example configuration reflects the main experimental method settings,
  including 20 construction responses and five cyclic training views.
- Fingerprint records contain the private query-response pair and should not be
  published for a protected model.
- Numerical views simulate quantization perturbations during embedding; actual
  deployment quantization is applied to the merged checkpoint.
- Exact matching strips only outer whitespace and remains case-sensitive.

## License

MIT License. See [LICENSE](LICENSE).
