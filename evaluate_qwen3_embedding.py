import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen embedding models on TCM similarity benchmark.")
    parser.add_argument("--csv_path", type=str, default="similar_tcm_terms.csv", help="CSV file containing term pairs.")
    parser.add_argument(
        "--model_paths",
        nargs="+",
        required=True,
        help="List of model identifiers or local directories to evaluate.",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=None,
        help="Optional display names corresponding to --model_paths.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding inference.")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum token length for embedding inference.")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (e.g. cuda, cpu).")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass flag if remote code trust is required.")
    return parser.parse_args()


def load_pairs(csv_path: str) -> Tuple[List[Tuple[str, str]], Optional[List[int]]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    pairs: List[Tuple[str, str]] = []
    labels: List[int] = []
    has_label = False

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"term_a", "term_b"}
        if not required_cols.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {sorted(required_cols)}. Found: {reader.fieldnames}")

        has_label = "label" in (reader.fieldnames or [])
        for row in reader:
            term_a = (row.get("term_a") or "").strip()
            term_b = (row.get("term_b") or "").strip()
            if not term_a or not term_b:
                continue
            pairs.append((term_a, term_b))
            if has_label:
                label_value = row.get("label")
                if label_value is None or label_value == "":
                    labels.append(1)
                else:
                    labels.append(int(float(label_value)))

    if not pairs:
        raise ValueError("No valid term pairs found in CSV.")

    if has_label:
        return pairs, labels
    return pairs, None


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    embeddings = summed / counts
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1)


def embed_texts(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
    sims = torch.sum(a_norm * b_norm, dim=-1)
    return sims.numpy()


def compute_metrics(similarities: np.ndarray, labels: Optional[List[int]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["mean_similarity"] = float(np.mean(similarities))
    metrics["median_similarity"] = float(np.median(similarities))

    if labels is None:
        return metrics

    labels_array = np.array(labels)
    positives = similarities[labels_array == 1]
    negatives = similarities[labels_array == 0]

    if positives.size > 0:
        metrics["positive_mean"] = float(np.mean(positives))
        metrics["positive_std"] = float(np.std(positives))
    if negatives.size > 0:
        metrics["negative_mean"] = float(np.mean(negatives))
        metrics["negative_std"] = float(np.std(negatives))

    if positives.size > 0 and negatives.size > 0:
        metrics["auc"] = float(compute_auc(labels_array, similarities))
        threshold = 0.5
        preds = (similarities >= threshold).astype(int)
        metrics["accuracy@0.5"] = float(np.mean(preds == labels_array))

    return metrics


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank-based AUC computation
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(scores)) + 1  # ranks start at 1
    sum_ranks_pos = np.sum(ranks[labels == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def evaluate_model(
    model_path: str,
    csv_pairs: List[Tuple[str, str]],
    labels: Optional[List[int]],
    batch_size: int,
    max_length: int,
    device: torch.device,
    trust_remote_code: bool,
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model.to(device)

    unique_terms: List[str] = sorted({term for pair in csv_pairs for term in pair})
    embeddings = embed_texts(model, tokenizer, unique_terms, batch_size, max_length, device)
    term_to_vec: Dict[str, torch.Tensor] = {
        term: embeddings[idx] for idx, term in enumerate(unique_terms)
    }

    sims: List[float] = []
    for term_a, term_b in csv_pairs:
        vec_a = term_to_vec[term_a]
        vec_b = term_to_vec[term_b]
        sim = cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0))[0]
        sims.append(sim)

    metrics = compute_metrics(np.array(sims), labels)
    return metrics


def main() -> None:
    args = parse_args()
    pairs, labels = load_pairs(args.csv_path)
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model_names = args.model_names or args.model_paths
    if len(model_names) != len(args.model_paths):
        raise ValueError("--model_names must match length of --model_paths.")

    results = []
    for name, path in zip(model_names, args.model_paths):
        metrics = evaluate_model(
            path,
            pairs,
            labels,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            trust_remote_code=args.trust_remote_code,
        )
        results.append((name, metrics))

    print(f"Evaluated {len(results)} models on {len(pairs)} term pairs using device={device}.")
    for name, metrics in results:
        print(f"\nModel: {name}")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
