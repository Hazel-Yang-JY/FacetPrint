from __future__ import annotations

import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import load_config, prepare_tokenizer, require_cuda, seed_everything, write_json, write_jsonl


WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")


def extract_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def recombine_word_positions(
    candidates: list[list[str]], target_length: int, splice_seed: int
) -> tuple[list[str], list[dict]]:
    """Select each target word position uniformly from the candidate responses."""
    if not candidates:
        raise ValueError("At least one candidate response is required")
    if target_length < 1:
        raise ValueError("target_length must be positive")
    if any(len(tokens) < target_length for tokens in candidates):
        raise ValueError("Every candidate must contain all requested word positions")
    rng = random.Random(splice_seed)
    target: list[str] = []
    trace: list[dict] = []
    for position in range(target_length):
        source_index = rng.randrange(len(candidates))
        word = candidates[source_index][position]
        target.append(word)
        trace.append(
            {
                "position": position,
                "source_response_index": source_index,
                "word": word,
            }
        )
    return target, trace


def construct(config_path: str | Path, force: bool = False) -> Path:
    config = load_config(config_path)
    cfg = config["construction"]
    private_dir = Path(config["private_dir"])
    selection_path = private_dir / "selection.json"
    if selection_path.exists() and not force:
        raise FileExistsError(f"{selection_path} exists; use --force to replace it")

    device = require_cuda()
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True)
    prepare_tokenizer(tokenizer)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], torch_dtype=torch.bfloat16, device_map={"": "cuda"}
    ).eval()

    secret = str(cfg["secret_query"])
    prompt = str(cfg["prompt_template"]).format(secret=secret)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    response_count = int(cfg["num_responses"])
    requested_words = int(cfg["target_words"])
    first_seed = int(cfg["source_seed"])
    source_responses: list[dict] = []

    for offset in range(response_count):
        source_seed = first_seed + offset
        seed_everything(source_seed)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=float(cfg["temperature"]),
                top_p=float(cfg["top_p"]),
                min_new_tokens=int(cfg.get("min_new_tokens", 80)),
                max_new_tokens=int(cfg["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
            )
        token_ids = generated[0, encoded["input_ids"].shape[1] :].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        words = extract_words(text)
        if "\ufffd" in text:
            raise RuntimeError(f"Seed {source_seed} produced invalid replacement characters")
        source_responses.append(
            {
                "seed": source_seed,
                "text": text,
                "words": words,
            }
        )

    target_length = min(requested_words, min(len(row["words"]) for row in source_responses))
    if target_length < int(cfg.get("minimum_target_words", 8)):
        raise RuntimeError(f"Common recombination length is only {target_length} words")
    target_words, trace = recombine_word_positions(
        [row["words"] for row in source_responses],
        target_length,
        int(cfg["splice_seed"]),
    )
    target_text = " ".join(target_words)
    if not target_text:
        raise RuntimeError("The recombined token sequence decoded to empty text")

    selection = {
        "method": "FacetPrint",
        "construction": "same-position random whole-word recombination",
        "secret_query": secret,
        "formatted_prompt": prompt,
        "target_response": target_text,
        "source_temperature": float(cfg["temperature"]),
        "source_top_p": float(cfg["top_p"]),
        "source_response_count": response_count,
        "source_seeds": [row["seed"] for row in source_responses],
        "target_word_count": target_length,
        "splice_seed": int(cfg["splice_seed"]),
        "source_responses": source_responses,
        "splice_trace": trace,
    }
    write_json(selection_path, selection)
    write_jsonl(
        private_dir / "fingerprint_train.jsonl",
        [
            {
                "id": "facetprint_000",
                "prompt": prompt,
                "response": target_text,
                "response_token_ids": target_ids,
            }
        ],
    )
    return selection_path
