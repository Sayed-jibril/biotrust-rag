"""
Configuration and constants for the BioTrust-RAG experiment.
"""
import os
import json
import random
import gc
import hashlib
from pathlib import Path
import numpy as np
import torch

# ============================================================
# SMOKE TEST FLAG
# ============================================================
SMOKE_TEST = False   # Set False for the real run

# ============================================================
# CORPUS PROVENANCE
# ============================================================
CORPUS_PROVENANCE = {
    "pubmedqa_aligned": {
        "source_dataset": "PubMedQA (qiaojin/PubMedQA, pqa_labeled train split)",
        "field": "context.contexts → concatenated full abstract",
        "description": "Clinical research abstracts answering biomedical research questions",
        "license": "MIT"
    },
    "medmcqa_aligned": {
        "source_dataset": "MedMCQA (openlifescienceai/medmcqa, train split)",
        "field": "exp (explanation)",
        "description": "Expert-written textbook/examination explanations for Indian PG medical entrance questions",
        "license": "Apache-2.0"
    }
}

# ============================================================
# MASTER CONFIGURATION
# ============================================================
CONFIG = {
    "session_name": "biotrust_unified_causal_final_fixed",
    "random_seed": 42,

    "datasets": ["pubmedqa", "medmcqa"],
    "max_test_per_dataset": 10 if SMOKE_TEST else 200,
    "corpus_size": 60 if SMOKE_TEST else 700,

    "models": ["flan-t5-small"] if SMOKE_TEST else [
        "flan-t5-large",
        "mistral-7b-instruct",
        "biomistral-7b"
    ],

    "retrievers": ["no_retrieval", "bm25", "medcpt_rerank", "random"] if SMOKE_TEST else [
        "no_retrieval",
        "bm25",
        "faiss_minilm",
        "faiss_medcpt",
        "hybrid_medcpt",
        "medcpt_rerank",
        "random"
    ],

    "robustness_modes": ["missing_evidence"] if SMOKE_TEST else [
        "noisy1",
        "contradictory",
        "missing_evidence"
    ],

    "include_robustness": True,

    "max_new_tokens": 64 if SMOKE_TEST else 128,
    "chunk_size": 200,
    "chunk_overlap": 30,
    "top_k": 3,
    "candidate_pool": 12,
    "rerank_topk": 12,
    "rrf_k": 60,

    "refusal_phrase": "I cannot answer from the provided context.",

    "gt_grounded_threshold": 0.75,
    "gt_partial_threshold": 0.25,

    "faithfulness_high": 0.75,
    "faithfulness_medium": 0.50,
    "faithfulness_low": 0.25,
    "contradiction_threshold": 0.50,
    "max_claims": 8,

    "bootstrap_iterations": 200 if SMOKE_TEST else 1000,
    "permutation_iterations": 500,

    "conformal_alpha": 0.05,
    "conformal_calib_fraction": 0.20,

    "run_bertscore": False if SMOKE_TEST else True,
    "run_llm_claim_validation": False if SMOKE_TEST else True,
    "llm_claim_validation_sample": 30 if SMOKE_TEST else 150,

    "output_dir": "./biotrust_unified_output_fixed",

    "nli_model": "cross-encoder/nli-deberta-v3-small",
    "bertscore_model": "roberta-large",
    "minilm_encoder": "sentence-transformers/all-MiniLM-L6-v2",

    "medcpt_query": "ncbi/MedCPT-Query-Encoder",
    "medcpt_article": "ncbi/MedCPT-Article-Encoder",
    "medcpt_cross": "ncbi/MedCPT-Cross-Encoder",
    "rerank_fallback": "cross-encoder/ms-marco-MiniLM-L-6-v2",

    "corpus_provenance": CORPUS_PROVENANCE,
    "max_chunks_per_corpus": 726,

    "corpus_conditions": {
        "pubmedqa": ["aligned", "misaligned", "partially_aligned"],
        "medmcqa": ["aligned", "misaligned", "partially_aligned"],
    },

    "partial_mix_ratio": 0.5,

    "external_textbook_path": None,

    "medmcqa_corpus_split": "train",
    "medmcqa_train_pool_size": 500 if SMOKE_TEST else 5000,

    "equalize_corpus_docs": True,
    "equalize_corpus_chunks": True,
    "max_chunks_per_corpus": 200 if SMOKE_TEST else 2200,

    "alignment_probe_n": 5 if SMOKE_TEST else 30,
    "alignment_human_samples_per_cell": 3 if SMOKE_TEST else 20,

    "primary_metrics": {
        "pubmedqa": "rouge_l",
        "medmcqa": "exact_match_all"
    },

    "multiple_testing_method": "holm",
    "run_mixed_effects": True,
    "mixed_effects_max_rows": None,

    "simulate_human_ratings": False,
}

MODEL_REGISTRY = {
    "flan-t5-small": {
        "repo": "google/flan-t5-small",
        "type": "seq2seq",
        "quantize": False,
        "max_input": 512,
        "display": "FLAN-T5-Small (smoke)",
        "params_b": 0.08
    },
    "flan-t5-large": {
        "repo": "google/flan-t5-large",
        "type": "seq2seq",
        "quantize": False,
        "max_input": 1024,
        "display": "FLAN-T5-Large (780M)",
        "params_b": 0.78
    },
    "mistral-7b-instruct": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "type": "causal",
        "quantize": True,
        "max_input": 4096,
        "display": "Mistral-7B Instruct",
        "params_b": 7.0
    },
    "biomistral-7b": {
        "repo": "BioMistral/BioMistral-7B",
        "type": "causal",
        "quantize": True,
        "max_input": 4096,
        "display": "BioMistral-7B",
        "params_b": 7.0
    },
}

ACTIVE_REGISTRY = {k: v for k, v in MODEL_REGISTRY.items() if k in CONFIG["models"]}

DATASET_DISPLAY = {
    "pubmedqa": "PubMedQA (Research QA)",
    "medmcqa": "MedMCQA (Medical Exam)"
}

CORPUS_DISPLAY = {
    "aligned": "Aligned",
    "misaligned": "Misaligned",
    "partially_aligned": "Partially Aligned"
}

RETRIEVER_DISPLAY = {
    "no_retrieval": "No Retrieval",
    "bm25": "BM25",
    "faiss_minilm": "FAISS(MiniLM)",
    "faiss_medcpt": "FAISS(MedCPT)",
    "hybrid_medcpt": "Hybrid(RRF)",
    "medcpt_rerank": "MedCPT+Rerank",
    "random": "Random"
}

# Create output directories
def create_dirs():
    for sub in [
        "data", "corpus", "retrieval", "generation",
        "evaluation", "statistics", "figures",
        "validation", "human_eval"
    ]:
        os.makedirs(os.path.join(CONFIG["output_dir"], sub), exist_ok=True)
create_dirs()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def model_key(n):
    return n.replace("/", "_").replace("--", "_")

def save_json(o, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=2)

def load_json(p):
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(rs, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except:
                    pass
    return out

def append_jsonl(r, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

def chunk_text(text, max_words=200, overlap=30):
    import re
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    step = max(max_words - overlap, 1)
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks

def stable_int(s, mod=100000):
    return int(hashlib.md5(str(s).encode("utf-8")).hexdigest(), 16) % mod

def bootstrap_ci(vals, n_iter=None, conf=0.95):
    n_iter = n_iter or CONFIG["bootstrap_iterations"]
    vals = np.asarray([float(v) for v in vals if v is not None and not np.isnan(float(v))])
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(CONFIG["random_seed"])
    means = [np.mean(rng.choice(vals, len(vals), replace=True)) for _ in range(n_iter)]
    return (float(np.mean(vals)),
            float(np.percentile(means, (1 - conf) / 2 * 100)),
            float(np.percentile(means, (1 + conf) / 2 * 100)))

def paired_permutation_test(diffs, n_perm=None):
    n_perm = n_perm or CONFIG["permutation_iterations"]
    diffs = np.asarray(diffs, float)
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 1.0
    obs = abs(nz.sum())
    rng = np.random.default_rng(CONFIG["random_seed"])
    cnt = sum(abs((rng.choice([-1, 1], len(nz)) * nz).sum()) >= obs for _ in range(n_perm))
    return float((cnt + 1) / (n_perm + 1))

# Print initial info
if SMOKE_TEST:
    print("=" * 70)
    print("  SMOKE_TEST = True  (pipeline validation only)")
    print("  Set SMOKE_TEST=False in config.py before the real experiment.")
    print("=" * 70)

print(f"Device: {DEVICE}")
