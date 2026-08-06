# config.py
# -*- coding: utf-8 -*-

import os

CONFIG = {
    "session_name": "biotrust_q1_official",
    "random_seed": 42,
    "datasets": ["pubmedqa", "medmcqa"],
    "max_test_per_dataset": 200,
    "corpus_size": 700,
    "models": ["flan-t5-large", "mistral-7b-instruct", "biomistral-7b"],
    "retrievers": [
        "no_retrieval", "bm25", "faiss_minilm", "faiss_medcpt",
        "hybrid_medcpt", "medcpt_rerank", "random"
    ],
    "robustness_modes": ["noisy1", "contradictory", "missing_evidence"],
    "max_new_tokens": 128,
    "chunk_size": 200,
    "chunk_overlap": 30,
    "top_k": 3,
    "candidate_pool": 12,
    "rerank_topk": 12,
    "rrf_k": 60,
    "gt_grounded_threshold": 0.75,
    "gt_partial_threshold": 0.25,
    "faithfulness_high": 0.75,
    "faithfulness_medium": 0.50,
    "faithfulness_low": 0.25,
    "contradiction_threshold": 0.50,
    "max_claims": 8,
    "refusal_phrase": "I cannot answer from the provided context.",
    "bootstrap_iterations": 1000,
    "permutation_iterations": 500,
    "conformal_alpha": 0.05,
    "conformal_calib_fraction": 0.20,
    "include_robustness": True,
    "run_bertscore": True,
    "run_llm_claim_validation": True,
    "llm_claim_validation_sample": 150,
    "output_dir": "./biotrust_q1_output",
    "nli_model": "cross-encoder/nli-deberta-v3-small",
    "bertscore_model": "roberta-large",
    "minilm_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "medcpt_query": "ncbi/MedCPT-Query-Encoder",
    "medcpt_article": "ncbi/MedCPT-Article-Encoder",
    "medcpt_cross": "ncbi/MedCPT-Cross-Encoder",
    "rerank_fallback": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}

MODEL_REGISTRY = {
    "flan-t5-large": {
        "repo": "google/flan-t5-large",
        "type": "seq2seq", "quantize": False,
        "max_input": 1024, "chat_template": False,
        "display": "FLAN-T5-Large\n(780M)"
    },
    "mistral-7b-instruct": {
        "repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "type": "causal", "quantize": True,
        "max_input": 4096, "chat_template": True,
        "display": "Mistral-7B\nInstruct"
    },
    "biomistral-7b": {
        "repo": "BioMistral/BioMistral-7B",
        "type": "causal", "quantize": True,
        "max_input": 4096, "chat_template": True,
        "display": "BioMistral-7B\n(Biomedical)"
    },
}

DATASET_DISPLAY = {
    "pubmedqa": "PubMedQA\n(Research QA)",
    "medmcqa": "MedMCQA\n(Medical Exam)",
}

RETRIEVER_DISPLAY = {
    "no_retrieval": "No Retrieval",
    "bm25": "BM25\n(Sparse)",
    "faiss_minilm": "FAISS\n(MiniLM)",
    "faiss_medcpt": "FAISS\n(MedCPT)",
    "hybrid_medcpt": "Hybrid\n(RRF)",
    "medcpt_rerank": "MedCPT\n+ Rerank",
    "random": "Random\n(Control)",
}
