import warnings
warnings.filterwarnings("ignore")

# Import standard libraries
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import custom modules
from config import (
    DEVICE, SMOKE_TEST, OUTPUT_DIR, SEED, DATASETS, CORPUS_CONDITIONS,
    MODEL_REGISTRY, RETRIEVERS, INCLUDE_ROBUSTNESS, ROBUSTNESS_MODES,
    TOP_K, STATS_DIR, CORPUS_DIR, GENERATION_DIR
)
from corpus import (
    MultiDatasetLoader, build_corpus_chunks, create_corpus_condition_pools,
    balance_chunks_within_datasets, save_corpus_provenance_report
)
from pipeline import (
    RetrievalSystem, run_generation_loop, evaluate_generations
)

def main():
    """
    Main function to execute the analysis pipeline.
    """
    print("="*70)
    print("Starting BioTrust RAG Analysis Pipeline")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Smoke Test Mode: {SMOKE_TEST}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("-"*70)

    # --- Step 1: Load Datasets ---
    print("\n--- Step 1: Loading Datasets ---")
    loader = MultiDatasetLoader({
        "DATASETS": DATASETS,
        "CORPUS_SIZE": 100 if SMOKE_TEST else 700, # Smaller corpus for smoke test
        "CHUNK_SIZE": 128 if SMOKE_TEST else 256, # Smaller chunks for smoke test
        "CHUNK_OVERLAP": 16 if SMOKE_TEST else 32,
    })
    all_data = loader.load_all()

    # --- Step 2: Prepare Corpora ---
    print("\n--- Step 2: Preparing Corpora ---")
    corpus_config = create_corpus_condition_pools(all_data, {
        "DATASETS": DATASETS,
        "CORPUS_CONDITIONS": CORPUS_CONDITIONS,
        "CORPUS_SIZE": 100 if SMOKE_TEST else 700,
        "SEED": SEED,
        "PARTIAL_MIX_RATIO": 0.5,
        "EQUALIZE_CORPUS_DOCS": True,
    })

    # --- Step 3: Build Chunks ---
    print("\n--- Step 3: Building Chunks ---")
    corpus_chunks = {}
    for key, config_entry in corpus_config.items():
        docs = config_entry.get("docs", [])
        if docs:
            # Pass necessary config values to build_corpus_chunks
            chunk_config = {
                "CHUNK_SIZE": 128 if SMOKE_TEST else 256,
                "CHUNK_OVERLAP": 16 if SMOKE_TEST else 32,
                "CORPUS_DIR": CORPUS_DIR,
            }
            corpus_chunks[key] = build_corpus_chunks(key, docs, chunk_config)
        else:
            print(f" [SKIP] No documents for {key}")

    # --- Step 4: Balance Chunks (if configured) ---
    print("\n--- Step 4: Balancing Chunks ---")
    # Pass necessary config values to balance_chunks_within_datasets
    balance_config = {
        "CORPUS_CONDITIONS": CORPUS_CONDITIONS,
        "MAX_CHUNKS_PER_CORPUS": 200 if SMOKE_TEST else 726, # Smaller for smoke test
        "EQUALIZE_CORPUS_CHUNKS": True,
        "SEED": SEED,
    }
    corpus_chunks_balanced = balance_chunks_within_datasets(corpus_chunks, balance_config)

    # --- Step 5: Save Provenance Report ---
    print("\n--- Step 5: Saving Provenance Report ---")
    prov_config = {
        "CORPUS_DIR": CORPUS_DIR,
        "CORPUS_SIZE": 100 if SMOKE_TEST else 700,
        "EQUALIZE_CORPUS_DOCS": True,
        "MAX_CHUNKS_PER_CORPUS": 200 if SMOKE_TEST else 726,
        "EQUALIZE_CORPUS_CHUNKS": True,
    }
    save_corpus_provenance_report(prov_config)

    # --- Step 6: Build Retrieval Indices ---
    print("\n--- Step 6: Building Retrieval Indices ---")
    retrieval_systems = {}
    for key in sorted(corpus_chunks_balanced.keys()):
        if key not in corpus_chunks_balanced or len(corpus_chunks_balanced[key]) == 0:
            print(f" [SKIP] No chunks for {key}, cannot build retrieval system.")
            continue

        texts = corpus_chunks_balanced[key]["text"].astype(str).tolist()
        base_ds = "pubmedqa" if key.startswith("pubmedqa") else "medmcqa" # Derive base dataset name
        print(f"[BUILD] Index for {key} ({len(texts)} chunks)...")

        rs = RetrievalSystem(texts, key, base_ds, DEVICE)
        rs.build_bm25()
        rs.build_minilm()
        rs.build_medcpt() # This might fail silently if MedCPT isn't available
        rs.build_reranker() # This might fail silently if reranker isn't available
        retrieval_systems[key] = rs

        # Clear GPU memory after building each large index
        # from pipeline import clear_gpu # Import if needed inside loop
        # clear_gpu()

    print(f"[OK] Built indices for {len(retrieval_systems)} corpus conditions: {list(retrieval_systems.keys())}")

    # --- Step 7: Run Generation Loop ---
    print("\n--- Step 7: Running Generation Loop ---")
    # Select models based on SMOKE_TEST
    selected_models = ["flan-t5-small"] if SMOKE_TEST else list(MODEL_REGISTRY.keys())
    print(f"Selected models for generation: {selected_models}")

    # Pass necessary config values to run_generation_loop
    gen_config = {
        "RETRIEVERS": RETRIEVERS,
        "INCLUDE_ROBUSTNESS": INCLUDE_ROBUSTNESS,
        "ROBUSTNESS_MODES": ROBUSTNESS_MODES,
        "TOP_K": TOP_K,
        "MODEL_REGISTRY": MODEL_REGISTRY,
        "GENERATION_DIR": GENERATION_DIR,
        "FAITHFULNESS_MEDIUM": 0.3, # From config
        "CONTRADICTION_THRESHOLD": 0.5, # From config
    }
    run_generation_loop(selected_models, all_data, retrieval_systems, gen_config)

    # --- Step 8: Evaluate Generations ---
    print("\n--- Step 8: Evaluating Generations ---")
    # Pass necessary config values to evaluate_generations
    eval_config = {
        "DATASETS": DATASETS,
        "CORPUS_CONDITIONS": CORPUS_CONDITIONS,
        "MODELS": selected_models,
        "RETRIEVERS": RETRIEVERS,
        "ROBUSTNESS_MODES": ROBUSTNESS_MODES if INCLUDE_ROBUSTNESS else [],
        "INCLUDE_ROBUSTNESS": INCLUDE_ROBUSTNESS,
        "STATS_DIR": STATS_DIR,
        "GENERATION_DIR": GENERATION_DIR,
    }
    eval_df = evaluate_generations("", eval_config) # Path pattern is handled internally now

    print("\n--- Pipeline Execution Complete ---")
    print("="*70)
    print("Check the 'outputs' directory for results (figures, statistics, generations).")


if __name__ == "__main__":
    main()
