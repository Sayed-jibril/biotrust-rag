#!/usr/bin/env python
"""
Main entry point for the BioTrust-RAG experiment.
"""
import os
import sys
import time
import random
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

from config import (
    CONFIG, SMOKE_TEST, DEVICE, set_seed, clear_gpu, model_key,
    save_json, load_json, save_jsonl, load_jsonl, append_jsonl,
    ACTIVE_REGISTRY, RETRIEVER_DISPLAY, DATASET_DISPLAY, CORPUS_DISPLAY
)
from corpus import (
    MultiDatasetLoader, build_corpus_pools, build_all_chunks,
    RetrievalSystem, run_leakage_audit, validate_alignment,
    build_alignment_human_template, generate_corpus_report
)
from pipeline import (
    load_llm, build_prompt, generate_answer, NLIHelper,
    run_evaluation, run_corpus_alignment_paired_analysis,
    run_partial_alignment_analysis, run_mixed_effects,
    build_abstention_by_corpus_table, conformal_analysis_stratified,
    run_adaptive_conformal_inference, generate_final_summary,
    generate_premium_figures, generate_human_samples, simulate_ratings,
    run_mcq_artifact_analysis
)

set_seed(CONFIG["random_seed"])

def main():
    print("=" * 70)
    print("BIOTRUST-RAG UNIFIED CAUSAL EXPERIMENT — FINAL FIXED")
    print("=" * 70)
    print(f"SMOKE_TEST = {SMOKE_TEST}")
    print(f"Device: {DEVICE}")
    print(f"Models: {CONFIG['models']}")
    print(f"Datasets: {CONFIG['datasets']}")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------
    loader = MultiDatasetLoader(CONFIG)
    all_data = loader.load_all()
    print("\n[MAIN] Data loaded.")

    # ------------------------------------------------------------
    # 2. Build corpus pools and chunk indices
    # ------------------------------------------------------------
    CORPUS_CONFIG, pubmedqa_abstract_pool, textbook_pool, textbook_source = build_corpus_pools(all_data)
    corpus_chunks = build_all_chunks(CORPUS_CONFIG)

    generate_corpus_report(CORPUS_CONFIG, corpus_chunks, pubmedqa_abstract_pool, textbook_pool)

    test_questions = all_data["pubmedqa"]["test"] + all_data["medmcqa"]["test"]
    all_docs = pubmedqa_abstract_pool + textbook_pool
    leakage_report = run_leakage_audit(test_questions, all_docs)
    save_json(leakage_report, os.path.join(CONFIG["output_dir"], "statistics", "leakage_audit_report.json"))
    print("[MAIN] Leakage audit done.")

    # ------------------------------------------------------------
    # 3. Build retrieval indices per corpus condition
    # ------------------------------------------------------------
    retrieval_systems = {}
    for key in sorted(corpus_chunks.keys()):
        if key not in corpus_chunks or len(corpus_chunks[key]) == 0:
            continue
        texts = corpus_chunks[key]["text"].astype(str).tolist()
        base_ds = "pubmedqa" if key.startswith("pubmedqa") else "medmcqa"
        print(f"\n[BUILD] {key} ({len(texts)} chunks)")
        rs = RetrievalSystem(texts, key, base_ds, DEVICE)
        rs.build_bm25()
        rs.build_minilm()
        rs.build_medcpt()
        rs.build_reranker()
        retrieval_systems[key] = rs
        clear_gpu()
    print("\n[MAIN] Retrieval indices built.")

    # ------------------------------------------------------------
    # 4. Alignment validation (diagnostic)
    # ------------------------------------------------------------
    val_df = validate_alignment(corpus_chunks, all_data)
    build_alignment_human_template(corpus_chunks, all_data)

    # ------------------------------------------------------------
    # 5. Run generation (experiment loop)
    # ------------------------------------------------------------
    all_conditions = list(CONFIG["retrievers"])
    if CONFIG["include_robustness"]:
        all_conditions += CONFIG["robustness_modes"]

    for model_name in CONFIG["models"]:
        print(f"\n{'='*60}\nMODEL: {model_name}\n{'='*60}")
        llm, tokenizer, meta = load_llm(model_name)
        mkey = model_key(model_name)
        for ds_name in CONFIG["datasets"]:
            if ds_name not in all_data or not all_data[ds_name]["test"]:
                continue
            test_q = all_data[ds_name]["test"]
            print(f"\nDATASET: {ds_name} ({len(test_q)} questions)")
            for cc in CONFIG["corpus_conditions"][ds_name]:
                rs_key = f"{ds_name}_{cc}"
                if rs_key not in retrieval_systems:
                    print(f"  [SKIP] {rs_key}")
                    continue
                rs = retrieval_systems[rs_key]
                print(f"  CORPUS: {cc}")
                for cond in all_conditions:
                    gen_dir = os.path.join(CONFIG["output_dir"], "generation", ds_name, cc, mkey)
                    os.makedirs(gen_dir, exist_ok=True)
                    out_path = os.path.join(gen_dir, f"{cond}.jsonl")
                    done_ids = {r.get("qid") for r in load_jsonl(out_path)}
                    if len(done_ids) >= len(test_q):
                        print(f"    {cond}: cached ({len(done_ids)})")
                        continue
                    print(f"    {cond}: generating...", end=" ")
                    for q in tqdm(test_q, desc=cond, leave=False):
                        if q["id"] in done_ids:
                            continue
                        question = str(q["question"])
                        reference = str(q.get("reference", ""))
                        decision = str(q.get("decision", ""))
                        robustness = None
                        base = cond
                        if cond in CONFIG["robustness_modes"]:
                            robustness = cond
                            base = "medcpt_rerank"
                        if base == "no_retrieval":
                            cids, cscores = [], []
                        else:
                            cids, cscores = rs.retrieve(question, base, CONFIG["top_k"])
                        if robustness:
                            cids, cscores = rs.apply_robustness(cids, cscores, robustness)
                        rtexts = [rs.chunk_texts[i] for i in cids if 0 <= i < len(rs.chunk_texts)]
                        prompt = build_prompt(question, rtexts, cond, tokenizer, meta, ds_name)
                        answer, p_tok, a_tok, lat = generate_answer(llm, tokenizer, prompt, meta)
                        append_jsonl({
                            "dataset": ds_name,
                            "corpus_condition": cc,
                            "model": model_name,
                            "model_key": mkey,
                            "condition": cond,
                            "qid": str(q["id"]),
                            "question": question,
                            "reference": reference,
                            "decision": decision,
                            "chunk_ids": [int(i) for i in cids],
                            "chunk_texts": rtexts,
                            "retrieval_scores": [float(s) for s in cscores],
                            "generated": str(answer),
                            "prompt_tokens": int(p_tok),
                            "answer_tokens": int(a_tok),
                            "latency": float(lat),
                        }, out_path)
                    print("done")
        del llm, tokenizer
        clear_gpu()
        print(f"[DONE] {model_name}")

    print("[MAIN] Generation complete.")

    # ------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------
    metrics = run_evaluation([], retrieval_systems, all_data)  # loads from disk
    metrics_df = pd.DataFrame(metrics)
    print(f"[MAIN] Evaluation done: {len(metrics_df)} records.")

    # ------------------------------------------------------------
    # 7. MCQ Artifact Analysis (3‑judge NLI)
    # ------------------------------------------------------------
    table4 = run_mcq_artifact_analysis(metrics_df)

    # ------------------------------------------------------------
    # 8. Statistical analyses
    # ------------------------------------------------------------
    rq1_df = run_corpus_alignment_paired_analysis(metrics_df)
    rq1_partial_df = run_partial_alignment_analysis(metrics_df)
    mixed_df = run_mixed_effects(metrics_df)
    abst_df = build_abstention_by_corpus_table(metrics_df)
    cp_result = conformal_analysis_stratified(metrics_df)
    aci_result = run_adaptive_conformal_inference(metrics_df)
    final_df = generate_final_summary(metrics_df)

    # ------------------------------------------------------------
    # 9. Human evaluation templates
    # ------------------------------------------------------------
    human_df = generate_human_samples(metrics_df)
    if CONFIG.get("simulate_human_ratings", False):
        print("[WARN] Simulated human ratings are for pipeline testing only.")
        # Optionally simulate and save rated CSV
        # ...

    # ------------------------------------------------------------
    # 10. Figures (Premium 12)
    # ------------------------------------------------------------
    generate_premium_figures(metrics_df=metrics_df, human_df=human_df)

    # ------------------------------------------------------------
    # 11. Manifest
    # ------------------------------------------------------------
    manifest = {
        "experiment": "BioTrust-RAG Unified Causal Experiment — FINAL FIXED",
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": CONFIG["datasets"],
        "models": CONFIG["models"],
        "retrievers": CONFIG["retrievers"],
        "robustness_modes": CONFIG["robustness_modes"],
        "corpus_conditions": CONFIG["corpus_conditions"],
        "total_records": len(metrics),
        "total_generations_expected": (len(CONFIG["datasets"]) *
                                       len(CONFIG["corpus_conditions"]["pubmedqa"]) *
                                       len(CONFIG["models"]) *
                                       (len(CONFIG["retrievers"]) + len(CONFIG["robustness_modes"])) *
                                       CONFIG["max_test_per_dataset"]),
        "reviewer_fixes": [
            "Comment 1: single causal backbone: corpus-task alignment.",
            "Comment 2: within-dataset corpus swap design.",
            "Added aligned, misaligned, and partially aligned corpus conditions.",
            "PubMedQA: aligned PubMed abstracts vs misaligned textbook/exam corpus.",
            "MedMCQA: aligned textbook/exam corpus vs misaligned PubMed abstracts.",
            "Same questions, models, prompts, retrievers, top-k, decoding, and evaluation within each dataset.",
            "Primary comparisons are question-level paired within-dataset contrasts.",
            "Holm-Bonferroni correction applied.",
            "Mixed-effects models added.",
            "Conformal analysis stratified by corpus condition.",
            "MCQ artifact analyzed with 3-judge NLI (Fleiss’ κ).",
            "12 premium figures generated (including trajectory, trust taxonomy, human validation)."
        ],
        "output_files": {
            "final_summary": "statistics/FINAL_SUMMARY.csv",
            "rq1_paired": "statistics/rq1_corpus_alignment_paired.csv",
            "rq1_partial": "statistics/rq1_partial_alignment.csv",
            "mixed_effects": "statistics/mixed_effects_primary.csv",
            "conformal": "statistics/conformal_analysis.json",
            "all_metrics": "evaluation/all_metrics.jsonl",
            "table4": "statistics/table4_nli_artifact.csv",
            "alignment_validation": "statistics/alignment_validation.csv",
            "human_eval": "human_eval/human_eval_blind.csv"
        }
    }
    save_json(manifest, os.path.join(CONFIG["output_dir"], "manifest.json"))
    print("[MAIN] Manifest saved.")
    print("=" * 70)
    print("EXPERIMENT COMPLETE.")
    print("=" * 70)

if __name__ == "__main__":
    main()
