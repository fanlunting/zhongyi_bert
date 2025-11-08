import argparse
import os
import random
import re
from typing import Dict, Iterable, List, Tuple

import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_scheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_corpus(files: Iterable[str]) -> List[str]:
    documents: List[str] = []
    for path in files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Corpus file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        cleaned = normalize_text(text)
        segments = split_into_segments(cleaned)
        documents.extend(segments)
    unique_docs = list(dict.fromkeys([s for s in documents if len(s) >= 10]))
    if not unique_docs:
        raise ValueError("No valid segments found in corpus files.")
    return unique_docs


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_segments(text: str, max_length: int = 256) -> List[str]:
    delimiters = "。！？!?；;：:\n"
    segments: List[str] = []
    buffer: List[str] = []
    for char in text:
        buffer.append(char)
        if char in delimiters or len(buffer) >= max_length:
            segments.append("".join(buffer).strip())
            buffer = []
    if buffer:
        segments.append("".join(buffer).strip())
    return [seg for seg in segments if seg]


class TCMSegmentDataset(Dataset):
    def __init__(self, segments: List[str]) -> None:
        self._segments = segments

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> str:
        return self._segments[idx]


def random_augmentation(text: str, dropout_ratio: float) -> str:
    if len(text) <= 1 or dropout_ratio <= 0.0:
        return text
    chars = list(text)
    keep = max(1, int(len(chars) * (1.0 - dropout_ratio)))
    selected_indices = sorted(random.sample(range(len(chars)), keep))
    augmented = "".join(chars[i] for i in selected_indices)
    return augmented if augmented.strip() else text


def collate_fn_builder(tokenizer, max_length: int, dropout_ratio: float):
    def collate(batch: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        view1 = [random_augmentation(text, dropout_ratio) for text in batch]
        view2 = [random_augmentation(text, dropout_ratio) for text in batch]

        tokens_a = tokenizer(
            view1,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens_b = tokenizer(
            view2,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {"a": tokens_a, "b": tokens_b}

    return collate


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return torch.nn.functional.normalize(summed / counts, p=2, dim=-1)


def contrastive_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = torch.matmul(embeddings_a, embeddings_b.transpose(0, 1)) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_a = torch.nn.functional.cross_entropy(logits, labels)
    loss_b = torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_a + loss_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continued pretraining for qwen3-embedding model on TCM corpora.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Base embedding model identifier.",
    )
    parser.add_argument(
        "--train_files",
        nargs="+",
        default=["古籍.txt", "中医教材.txt"],
        help="Training corpus file paths.",
    )
    parser.add_argument("--output_dir", type=str, default="./qwen3-embedding-tcm", help="Directory to store the fine-tuned model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Total batch size per device.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup proportion.")
    parser.add_argument("--temperature", type=float, default=0.05, help="InfoNCE temperature.")
    parser.add_argument("--dropout_ratio", type=float, default=0.1, help="Character dropout ratio for positive pair generation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adaptation.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override total training steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed + accelerator.process_index)

    accelerator.print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.use_lora:
        accelerator.print("Wrapping model with LoRA adapters...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=None,
        )
        model = get_peft_model(model, lora_config)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    accelerator.print("Reading corpus...")
    segments = read_corpus(args.train_files)
    accelerator.print(f"Loaded {len(segments)} segments.")

    dataset = TCMSegmentDataset(segments)
    collate_fn = collate_fn_builder(tokenizer, args.max_length, args.dropout_ratio)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    no_decay = ["bias", "layer_norm.weight", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    if args.max_train_steps is None:
        num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
        args.max_train_steps = args.num_epochs * max(1, num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    global_step = 0
    model.train()

    accelerator.print("Starting training...")
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                batch_a = {k: v.to(accelerator.device) for k, v in batch["a"].items()}
                batch_b = {k: v.to(accelerator.device) for k, v in batch["b"].items()}

                outputs_a = model(**batch_a)
                outputs_b = model(**batch_b)

                embeddings_a = mean_pooling(outputs_a.last_hidden_state, batch_a["attention_mask"])
                embeddings_b = mean_pooling(outputs_b.last_hidden_state, batch_b["attention_mask"])

                loss = contrastive_loss(embeddings_a, embeddings_b, temperature=args.temperature)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process and global_step % args.save_steps == 0:
                save_checkpoint(model, tokenizer, args.output_dir, f"step_{global_step}", accelerator)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            accelerator.print(f"Epoch {epoch + 1} finished. Global step: {global_step}")
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        accelerator.print("Training complete. Saving final model...")
    save_checkpoint(model, tokenizer, args.output_dir, "final", accelerator)
    accelerator.print("Done.")


def save_checkpoint(model, tokenizer, output_dir: str, tag: str, accelerator: Accelerator) -> None:
    accelerator.wait_for_everyone()
    save_dir = os.path.join(output_dir, tag)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
