"""
Module for handling corpus loading, processing, chunking, and provenance.

This module provides classes and functions to manage the different
corpora used in the RAG experiments, ensuring they are correctly
prepared and documented according to the experimental design.
"""

import json
import random
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import datasets library here
from datasets import load_dataset

# Import configuration
from config import (
    DATASETS, CORPUS_CONDITIONS, CORPUS_SIZE, CHUNK_SIZE, CHUNK_OVERLAP,
    EQUALIZE_CORPUS_DOCS, EQUALIZE_CORPUS_CHUNKS, MAX_CHUNKS_PER_CORPUS,
    SEED, CORPUS_PROVENANCE, PARTIAL_MIX_RATIO, EXTERNAL_TEXTBOOK_PATH
)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def sample_docs(docs: List[Dict], target_count: int, seed: int) -> List[Dict]:
    """Randomly samples documents from a list."""
    rng = random.Random(seed)
    if len(docs) <= target_count:
        return docs
    return rng.sample(docs, target_count)


class MultiDatasetLoader:
    """
    Loads and prepares multiple datasets (PubMedQA, MedMCQA).

    Handles the separation of documents (for the corpus) and test questions.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _load_pubmedqa(self) -> Dict[str, List[Dict]]:
        """Loads PubMedQA dataset."""
        print("[LOAD] Loading PubMedQA...")
        try:
            dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
            corpus_docs = []
            test_questions = []
            # Assuming the 'train' split contains labeled examples for corpus and questions
            # Adjust based on actual structure if different
            for i, ex in enumerate(dataset['train']):
                # Create a document record for the corpus
                contexts_str = " ".join(ex.get("context", {}).get("contexts", []))
                corpus_docs.append({
                    "id": f"pubmedqa_corpus_{i}",
                    "doc_text": contexts_str,
                    "decision": ex.get("final_decision", ""), # Ground truth answer
                    "original_example": ex # Store original if needed later
                })
                # Create a test question record
                test_questions.append({
                    "id": f"pubmedqa_test_{i}",
                    "question": ex.get("question", ""),
                    "reference": ex.get("final_decision", ""), # Ground truth answer
                    "decision": ex.get("final_decision", "")
                })
            print(f"[DATA] PubMedQA: Loaded {len(corpus_docs)} corpus docs, {len(test_questions)} test questions")
            return {"corpus": corpus_docs, "test": test_questions}
        except Exception as e:
            print(f"Error loading PubMedQA: {e}")
            return {"corpus": [], "test": []}

    def _load_medmcqa(self) -> Dict[str, List[Dict]]:
        """Loads MedMCQA dataset."""
        print("[LOAD] Loading MedMCQA...")
        try:
            dataset = load_dataset("openlifescienceai/medmcqa")
            corpus_docs = []
            test_questions = []
            # Use 'train' split for corpus-like explanations and 'validation'/'test' for questions
            # Adjust splits based on intended use
            # Corpus: explanations from training set
            for i, ex in enumerate(dataset['train']):
                 corpus_docs.append({
                     "id": f"medmcqa_corpus_{i}",
                     "doc_text": str(ex.get("exp", "")), # Explanation field
                     "original_example": ex
                 })
            # Test: questions from validation/test set
            # Assuming 'validation' split is used for testing in the original code context
            for i, ex in enumerate(dataset.get('validation', dataset.get('test', []))):
                 options_text = f"A: {ex.get('opa', '')} | B: {ex.get('opb', '')} | C: {ex.get('opc', '')} | D: {ex.get('opd', '')}"
                 correct_option_key = ["A", "B", "C", "D"][ex.get("cop", 0)]
                 test_questions.append({
                     "id": f"medmcqa_test_{i}",
                     "question": f"{ex.get('question', '')}\nOptions:\n{options_text}",
                     "reference": correct_option_key, # Correct option
                     "cop": ex.get("cop", 0),
                     "opa": ex.get("opa", ""),
                     "opb": ex.get("opb", ""),
                     "opc": ex.get("opc", ""),
                     "opd": ex.get("opd", ""),
                 })
            print(f"[DATA] MedMCQA: Loaded ~{len(corpus_docs)} corpus docs (explanations), {len(test_questions)} test questions")
            return {"corpus": corpus_docs, "test": test_questions}
        except Exception as e:
            print(f"Error loading MedMCQA: {e}")
            return {"corpus": [], "test": []}


    def load_all(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Loads all specified datasets."""
        all_data = {}
        for ds_name in DATASETS:
            if ds_name == "pubmedqa":
                all_data[ds_name] = self._load_pubmedqa()
            elif ds_name == "medmcqa":
                all_data[ds_name] = self._load_medmcqa()
            else:
                print(f"[WARN] Dataset {ds_name} not recognized, skipping.")
                all_data[ds_name] = {"corpus": [], "test": []}
        return all_data


def build_corpus_chunks(key: str, docs: List[Dict], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Chunks documents for a specific corpus condition and caches the result.
    """
    cache_path = config["CORPUS_DIR"] / f"chunks_{key}.csv"
    
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
            if len(df) > 0:
                print(f"[CHUNKS] {key}: loaded cached {len(df)} chunks")
                return df
        except Exception as e:
            print(f"[CHUNKS] Error loading cached chunks for {key}, regenerating: {e}")
            cache_path.unlink(missing_ok=True) # Remove corrupted cache

    if not docs:
        print(f"[CHUNKS] {key}: No documents provided, returning empty DataFrame.")
        return pd.DataFrame(columns=["chunk_id", "doc_id", "decision", "text"])

    recs = []
    for doc in tqdm(docs, desc=f"Chunking {key}", leave=False):
        txt = doc.get("doc_text", "") or doc.get("text", "")
        if not txt:
            continue # Skip empty documents
        for chunk_text_str in chunk_text(txt, config["CHUNK_SIZE"], config["CHUNK_OVERLAP"]):
            recs.append({
                "chunk_id": len(recs),
                "doc_id": doc.get("id", f"doc_{len(recs)}"), # Use doc id or generate one
                "decision": doc.get("decision", ""), # Propagate decision if present
                "text": chunk_text_str
            })

    df = pd.DataFrame(recs)
    df.to_csv(cache_path, index=False)
    print(f"[CHUNKS] {key}: Created and saved {len(df)} chunks")
    return df


def create_corpus_condition_pools(all_data: Dict[str, Dict[str, List[Dict]]], config: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """
    Creates pools of documents for each corpus condition (aligned, misaligned, partially aligned).
    """
    print("[POOL] Building corpus-condition pools ...")
    corpus_config = {}

    # Get base pools
    pubmedqa_corpus_pool = all_data.get("pubmedqa", {}).get("corpus", [])
    medmcqa_corpus_pool = all_data.get("medmcqa", {}).get("corpus", [])

    # Apply document equalization if configured
    if config.get("EQUALIZE_CORPUS_DOCS"):
        target_docs = min(len(pubmedqa_corpus_pool), len(medmcqa_corpus_pool), config.get("CORPUS_SIZE"))
        pubmedqa_corpus_pool = sample_docs(pubmedqa_corpus_pool, target_docs, config.get("SEED") + 10)
        medmcqa_corpus_pool = sample_docs(medmcqa_corpus_pool, target_docs, config.get("SEED") + 11)
        print(f"[POOL] Balanced document count per base pool: {target_docs}")

    # Define pools for each dataset
    dataset_pools = {
        "pubmedqa": {
            "aligned": pubmedqa_corpus_pool,
            "misaligned": medmcqa_corpus_pool,
        },
        "medmcqa": {
            "aligned": medmcqa_corpus_pool,
            "misaligned": pubmedqa_corpus_pool,
        }
    }

    # Build final corpus configurations
    for ds_name in config.get("DATASETS"):
        for cc in config.get("CORPUS_CONDITIONS").get(ds_name, []):
            key = f"{ds_name}_{cc}"
            if cc in ["aligned", "misaligned"]:
                corpus_config[key] = {"docs": dataset_pools[ds_name][cc]}
            elif cc == "partially_aligned":
                # Mix documents from both aligned and misaligned pools
                aligned_docs = dataset_pools[ds_name]["aligned"]
                misaligned_docs = dataset_pools[ds_name]["misaligned"]
                target_aligned = int(len(aligned_docs) * config.get("PARTIAL_MIX_RATIO"))
                target_misaligned = int(len(misaligned_docs) * config.get("PARTIAL_MIX_RATIO"))

                mixed_docs = sample_docs(aligned_docs, target_aligned, config.get("SEED") + 20) + \
                             sample_docs(misaligned_docs, target_misaligned, config.get("SEED") + 21)
                corpus_config[key] = {"docs": mixed_docs}
            else:
                print(f"[WARN] Unknown corpus condition '{cc}' for {ds_name}, skipping.")

    # Update provenance with actual counts
    CORPUS_PROVENANCE["pubmedqa_abstracts"]["n_documents"] = len(pubmedqa_corpus_pool)
    CORPUS_PROVENANCE["textbook_explanations"]["n_documents"] = len(medmcqa_corpus_pool)

    return corpus_config


def balance_chunks_within_datasets(corpus_chunks: Dict[str, pd.DataFrame], config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Balances the number of chunks within each dataset's conditions.
    Ensures fair comparison across conditions for the same dataset.
    """
    if not config.get("EQUALIZE_CORPUS_CHUNKS"):
        print("[CHUNK BALANCE] Skipping chunk equalization per config.")
        return corpus_chunks

    print("[CHUNK BALANCE] Equalizing chunk counts within each dataset ...")
    balanced_chunks = corpus_chunks.copy() # Start with original

    for ds_name in config.get("DATASETS"):
        keys_for_ds = [f"{ds_name}_{cc}" for cc in config.get("CORPUS_CONDITIONS").get(ds_name, []) if f"{ds_name}_{cc}" in corpus_chunks]
        if len(keys_for_ds) == 0:
            print(f"[CHUNK BALANCE] No chunks found for dataset {ds_name}, skipping.")
            continue

        lens = [len(balanced_chunks[k]) for k in keys_for_ds]
        target_len = min(lens)
        target_len = min(target_len, config.get("MAX_CHUNKS_PER_CORPUS", 726)) # Cap if necessary

        if target_len <= 0:
            print(f"[CHUNK BALANCE] Target length <= 0 for {ds_name}, skipping.")
            continue

        for k in keys_for_ds:
            if len(balanced_chunks[k]) > target_len:
                balanced_chunks[k] = balanced_chunks[k].sample(n=target_len, random_state=config.get("SEED")).reset_index(drop=True)
                print(f"[CHUNK BALANCE] {k}: reduced from {len(corpus_chunks[k])} to {target_len} chunks")

    return balanced_chunks


def save_corpus_provenance_report(config: Dict[str, Any]):
    """
    Saves a detailed report of corpus sources and creation process.
    """
    pubmedqa_doc_count = CORPUS_PROVENANCE["pubmedqa_abstracts"]["n_documents"]
    medmcqa_doc_count = CORPUS_PROVENANCE["textbook_explanations"]["n_documents"]

    report_lines = [
        "Corpus-Task Alignment: Source Documentation",
        "-" * 50,
        "",
        "Aligned corpus for PubMedQA:",
        " Source: PubMedQA dataset (qiaojin/PubMedQA, pqa_labeled train split)",
        " Field: context.contexts (concatenated into full abstracts)",
        " Type: Clinical research abstracts",
        f" N docs: {pubmedqa_doc_count}",
        "",
        "Aligned corpus for MedMCQA:",
        " Source: MedMCQA dataset (openlifescienceai/medmcqa, train split)",
        " Field: exp (explanation)",
        " Type: Medical textbook / examination explanations",
        f" N docs: {medmcqa_doc_count}",
        "",
        "Reciprocal (misaligned) conditions:",
        " PubMedQA misaligned -> uses MedMCQA explanations",
        " MedMCQA misaligned -> uses PubMedQA abstracts",
        "",
        "Partially aligned conditions:",
        " 50% document-level mixture of the two pools above",
        "",
        "Balancing:",
        f" Document count per condition (if equalized): {config.get('CORPUS_SIZE') if config.get('EQUALIZE_CORPUS_DOCS') else 'Variable'}",
        f" Chunk count per condition (if equalized): {config.get('MAX_CHUNKS_PER_CORPUS') if config.get('EQUALIZE_CORPUS_CHUNKS') else 'Variable'}",
    ]

    report_text = "\n".join(report_lines)
    report_path = config["CORPUS_DIR"] / "CORPUS_SOURCES_README.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"[OK] Corpus source report saved to {report_path}")
