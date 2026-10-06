from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from .io import load_config, prepare_tokenizer, read_jsonl, require_cuda, seed_everything, write_json
from .quant_lora import VIEWS, inject_lora, merge_lora, set_view, trainable_parameters


LOGGER = logging.getLogger("facetprint.train")


def encode_supervised(
    tokenizer,
    prompt: str,
    response: str,
    max_prompt_length: int,
    max_length: int,
    response_token_ids: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_prompt_length,
    )["input_ids"]
    if response_token_ids is None:
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    else:
        response_ids = [int(token) for token in response_token_ids]
    if tokenizer.eos_token_id is not None:
        response_ids = [*response_ids, int(tokenizer.eos_token_id)]
    available = max(max_length - len(prompt_ids), 1)
    response_ids = response_ids[:available]
    return prompt_ids + response_ids, [-100] * len(prompt_ids) + response_ids


def collate(tokenizer, examples: list[tuple[list[int], list[int]]], device: torch.device):
    width = max(len(input_ids) for input_ids, _ in examples)
    ids, labels, masks = [], [], []
    for input_ids, example_labels in examples:
        padding = width - len(input_ids)
        ids.append(input_ids + [tokenizer.pad_token_id] * padding)
        labels.append(example_labels + [-100] * padding)
        masks.append([1] * len(input_ids) + [0] * padding)
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def train(config_path: str | Path, max_steps: int | None = None) -> Path:
    config = load_config(config_path)
    cfg = config["training"]
    device = require_cuda()
    seed_everything(int(cfg["seed"]))

    private_dir = Path(config["private_dir"])
    output_dir = Path(config["output_dir"])
    selection = json.loads((private_dir / "selection.json").read_text(encoding="utf-8"))
    task_rows = read_jsonl(cfg["task_data"])
    if not task_rows:
        raise ValueError("Task data is empty")

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True)
    prepare_tokenizer(tokenizer)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], torch_dtype=torch.bfloat16, device_map={"": "cuda"}
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    replaced = inject_lora(
        model,
        cfg["target_modules"],
        int(cfg["lora_rank"]),
        float(cfg["lora_alpha"]),
        int(cfg["weight_group_size"]),
    )
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    steps = int(cfg["steps"])
    if max_steps is not None:
        steps = min(steps, max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_steps"]), steps
    )
    views = list(cfg["views"])
    unknown = set(views) - VIEWS
    if unknown:
        raise ValueError(f"Unknown training views: {sorted(unknown)}")
    if not views:
        raise ValueError("At least one training view is required")

    fingerprint = encode_supervised(
        tokenizer,
        selection["formatted_prompt"],
        selection["target_response"],
        int(cfg["max_prompt_length"]),
        int(cfg["max_length"]),
        None,
    )
    task_count = int(cfg.get("task_examples_per_step", 1))
    fingerprint_count = int(cfg.get("fingerprint_examples_per_step", 1))
    if task_count < 1 or fingerprint_count < 1:
        raise ValueError("Examples per step must be positive")

    metrics_path = output_dir / "training_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model.train()

    with metrics_path.open("w", encoding="utf-8") as stream:
        for step in range(steps):
            view = views[step % len(views)]
            set_view(model, view)
            task_examples = []
            for offset in range(task_count):
                task_index = ((step * task_count + offset) * 17) % len(task_rows)
                row = task_rows[task_index]
                task_examples.append(
                    encode_supervised(
                        tokenizer,
                        str(row.get("prompt", "")),
                        str(row["response"]),
                        int(cfg["max_prompt_length"]),
                        int(cfg["max_length"]),
                    )
                )
            input_ids, labels, attention_mask = collate(
                tokenizer,
                task_examples + [fingerprint] * fingerprint_count,
                device,
            )
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, :-1].float()
            shifted_labels = labels[:, 1:]
            valid = shifted_labels.ne(-100)
            safe_labels = shifted_labels.masked_fill(~valid, 0)
            token_ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                safe_labels.reshape(-1),
                reduction="none",
            ).reshape_as(safe_labels)
            per_example = torch.stack(
                [_masked_mean(token_ce[index], valid[index]) for index in range(token_ce.shape[0])]
            )
            task_loss = per_example[:task_count].mean()
            fingerprint_loss = per_example[task_count:].mean()
            loss = (
                float(cfg["task_loss_weight"]) * task_loss
                + float(cfg["fingerprint_loss_weight"]) * fingerprint_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, float(cfg["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            record = {
                "step": step + 1,
                "view": view,
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "fingerprint_loss": float(fingerprint_loss.detach()),
                "grad_norm": float(grad_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            if step == 0 or (step + 1) % int(cfg["log_every"]) == 0:
                LOGGER.info("%s", record)

    set_view(model, "bf16")
    merged_modules = merge_lora(model)
    model.config.use_cache = True
    model_path = output_dir / "fingerprint_model"
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_path, safe_serialization=True, max_shard_size="8GB")
    tokenizer.save_pretrained(model_path)
    write_json(
        output_dir / "training_summary.json",
        {
            "method": "FacetPrint",
            "base_model": config["model_path"],
            "construction": selection["construction"],
            "source_response_count": selection["source_response_count"],
            "target_word_count": selection["target_word_count"],
            "steps": steps,
            "views": views,
            "view_semantics": "shared LoRA with fake-quantized base weights, LoRA factors, and optional activations",
            "task_examples_per_step": task_count,
            "fingerprint_examples_per_step": fingerprint_count,
            "merged_modules": merged_modules,
            "injected_modules": replaced,
            "elapsed_seconds": time.perf_counter() - started,
            "release_checkpoint": str(model_path.resolve()),
        },
    )
    return model_path
