"""
Cross-Encoder Fine-Tuning for RAG Reranker
===========================================
Fine-tunes unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2 on your own
query-passage pairs extracted from the Samsung manual.

Training data is built automatically from your golden_qa.json + the
indexed ChromaDB collection — no manual labeling needed to get started.
For better results, add more labeled pairs to training_pairs.json.

Usage:
  python finetune_reranker.py                        # auto-build + train
  python finetune_reranker.py --pairs my_pairs.json  # use existing pairs
  python finetune_reranker.py --eval-only            # evaluate without training

Output:
  ./reranker-finetuned/   ← fine-tuned model, drop this path into rag_pipeline.py
"""

import json
import argparse
import chromadb
import torch
from pathlib import Path
from datasets import Dataset
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers import CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.util import mine_hard_negatives
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ─── Config ───────────────────────────────────────────────────────────────────

BASE_MODEL    = "unicamp-dl/mMiniLM-L6-v2-en-pt-msmarco-v2"
EMBED_MODEL   = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir"
OUTPUT_DIR    = "./reranker-finetuned"
GOLDEN_PATH   = "./golden_qa.json"
PAIRS_PATH    = "./training_pairs.json"
CHROMA_PATH   = "./.chromadb"
COLLECTION    = "rag_docs"

RETRIEVAL_K   = 20      # candidates to retrieve per question for training
NEG_PER_QUERY = 4       # hard negatives mined per positive pair
TRAIN_EPOCHS  = 3
BATCH_SIZE    = 16
EVAL_SPLIT    = 0.15    # fraction of pairs held out for evaluation

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Step 1: Auto-build training pairs from golden Q&A + ChromaDB ─────────────
#
# For each golden question:
#   - The golden answer chunk is the POSITIVE (label=1)
#   - Retrieved chunks that don't contain the answer are NEGATIVES (label=0)
#
# training_pairs.json schema:
# [
#   {"query": "...", "positive": "passage text", "negative": "passage text"},
#   ...
# ]
# You can also manually add pairs to this file for better coverage.

def build_training_pairs(golden_path: str, output_path: str) -> list[dict]:
    """Auto-generate training pairs from golden Q&A + ChromaDB retrieval."""
    print("Building training pairs from golden Q&A + ChromaDB...")

    golden  = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    chroma  = chromadb.PersistentClient(path=CHROMA_PATH)
    col     = chroma.get_collection(COLLECTION)
    embedder = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=DEVICE)

    pairs = []
    for item in golden:
        query    = item["question"]
        answer   = item["answer"].lower()

        # Embed the query and retrieve candidates
        q_embed  = embedder.get_query_embedding(
            f"Represent this sentence for searching relevant passages: {query}"
        )
        results  = col.query(query_embeddings=[q_embed], n_results=RETRIEVAL_K)
        chunks   = results["documents"][0]

        # Split into positives (contain answer keywords) and negatives
        positives = [c for c in chunks if any(w in c.lower() for w in answer.split() if len(w) > 4)]
        negatives = [c for c in chunks if c not in positives]

        if not positives or not negatives:
            continue

        # One positive paired with each negative
        for pos in positives[:2]:
            for neg in negatives[:NEG_PER_QUERY]:
                pairs.append({
                    "query":    query,
                    "positive": pos,
                    "negative": neg,
                })

    Path(output_path).write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Built {len(pairs)} training pairs → {output_path}")
    return pairs


# ─── Step 2: Load pairs and build HuggingFace Dataset ─────────────────────────

def load_pairs(pairs_path: str) -> tuple[Dataset, Dataset]:
    """Load pairs JSON and split into train/eval datasets."""
    pairs = json.loads(Path(pairs_path).read_text(encoding="utf-8"))

    # Convert to labeled (query, passage, label) format
    rows = []
    for p in pairs:
        rows.append({"query": p["query"], "passage": p["positive"], "label": 1.0})
        rows.append({"query": p["query"], "passage": p["negative"], "label": 0.0})

    # Shuffle and split
    import random
    random.shuffle(rows)
    split      = int(len(rows) * (1 - EVAL_SPLIT))
    train_rows = rows[:split]
    eval_rows  = rows[split:]

    print(f"  Train: {len(train_rows)} samples | Eval: {len(eval_rows)} samples")
    return (
        Dataset.from_list(train_rows),
        Dataset.from_list(eval_rows),
    )


# ─── Step 3: Build evaluator from golden Q&A ──────────────────────────────────

def build_evaluator(golden_path: str) -> CrossEncoderRerankingEvaluator:
    """
    CrossEncoderRerankingEvaluator measures MRR@10 — how often the
    correct passage ranks first among retrieved candidates.
    """
    golden  = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    chroma  = chromadb.PersistentClient(path=CHROMA_PATH)
    col     = chroma.get_collection(COLLECTION)
    embedder = HuggingFaceEmbedding(model_name=EMBED_MODEL, device=DEVICE)

    samples = []
    for item in golden:
        query  = item["question"]
        answer = item["answer"].lower()
        q_embed = embedder.get_query_embedding(
            f"Represent this sentence for searching relevant passages: {query}"
        )
        results = col.query(query_embeddings=[q_embed], n_results=RETRIEVAL_K)
        chunks  = results["documents"][0]
        positives = [c for c in chunks if any(w in c.lower() for w in answer.split() if len(w) > 4)]
        if positives:
            samples.append({
                "query":     query,
                "positive":  positives,
                "documents": chunks,
            })

    return CrossEncoderRerankingEvaluator(
        samples=samples,
        name="samsung-manual-ptbr",
        mrr_at_k=10,
    )


# ─── Step 4: Train ────────────────────────────────────────────────────────────

def train(train_dataset: Dataset, eval_dataset: Dataset, evaluator):
    print(f"\nLoading base model: {BASE_MODEL}")
    model = CrossEncoder(
        BASE_MODEL,
        num_labels=1,       # reranker always has 1 output label
        device=DEVICE,
        trust_remote_code=True,
    )

    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="samsung-manual-ptbr_mrr@10",
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    print(f"\nStarting fine-tuning for {TRAIN_EPOCHS} epochs...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
    print(f"To use it, set RERANK_MODEL = '{OUTPUT_DIR}' in rag_pipeline.py")


# ─── Step 5: Evaluate only ────────────────────────────────────────────────────

def evaluate_only():
    print(f"Evaluating {BASE_MODEL} before fine-tuning...")
    model     = CrossEncoder(BASE_MODEL, device=DEVICE, trust_remote_code=True)
    evaluator = build_evaluator(GOLDEN_PATH)
    results   = evaluator(model)
    print(f"\nMRR@10: {results.get('samsung-manual-ptbr_mrr@10', 'N/A'):.4f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune cross-encoder reranker")
    parser.add_argument("--pairs",     help="Path to existing training_pairs.json")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate without training")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_only()
    else:
        # Build pairs if not provided
        pairs_path = args.pairs or PAIRS_PATH
        if not Path(pairs_path).exists():
            build_training_pairs(GOLDEN_PATH, pairs_path)

        train_dataset, eval_dataset = load_pairs(pairs_path)
        evaluator = build_evaluator(GOLDEN_PATH)
        train(train_dataset, eval_dataset, evaluator)
