from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io import load_config, prepare_tokenizer, require_cuda, seed_everything, write_json


def exact_match(response: str, target: str) -> bool:
    """FacetPrint normalization: outer whitespace only, case-sensitive."""
    return response.strip() == target.strip()


def accept(matches: list[bool], required_matches: int) -> bool:
    return sum(matches) >= required_matches


def verify(config_path: str | Path) -> dict:
    config = load_config(config_path)
    cfg = config["verification"]
    device = require_cuda()
    private_dir = Path(config["private_dir"])
    output_dir = Path(config["output_dir"]) / "verification"
    selection = json.loads((private_dir / "selection.json").read_text(encoding="utf-8"))
    configured_model = cfg.get("model_path")
    model_path = Path(configured_model) if configured_model else Path(config["output_dir"]) / "fingerprint_model"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    prepare_tokenizer(tokenizer)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": "cuda"}
    ).eval()

    query_budget = int(cfg.get("query_budget", 3))
    required_matches = int(cfg.get("required_matches", 1))
    if not 1 <= required_matches <= query_budget:
        raise ValueError("required_matches must be between 1 and query_budget")
    prompt = selection["formatted_prompt"]
    target = selection["target_response"]
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    attempts: list[dict] = []
    seed_everything(int(cfg["seed"]))
    started = time.perf_counter()

    for index in range(query_budget):
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=float(cfg["temperature"]) > 0,
                temperature=float(cfg["temperature"]) if float(cfg["temperature"]) > 0 else None,
                top_p=float(cfg["top_p"]),
                max_new_tokens=int(cfg["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
            )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        response = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        matched = exact_match(response, target)
        attempts.append({"query": index + 1, "response": response, "exact_match": matched})
        hits = sum(bool(row["exact_match"]) for row in attempts)
        remaining = query_budget - len(attempts)
        if hits >= required_matches or hits + remaining < required_matches:
            break

    result = {
        "method": "FacetPrint",
        "model_path": str(model_path.resolve()),
        "decision": accept([bool(row["exact_match"]) for row in attempts], required_matches),
        "query_budget": query_budget,
        "required_matches": required_matches,
        "queries_used": len(attempts),
        "exact_matches": sum(bool(row["exact_match"]) for row in attempts),
        "normalization": "strip outer whitespace only; case-sensitive full response",
        "temperature": float(cfg["temperature"]),
        "seed": int(cfg["seed"]),
        "elapsed_seconds": time.perf_counter() - started,
        "attempts": attempts,
    }
    write_json(output_dir / "verification.json", result)
    return result
