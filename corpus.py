"""
Data loading, corpus construction, chunking, retrieval index building,
and alignment validation.
"""
import os
import random
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    CONFIG, SMOKE_TEST, save_json, load_json, save_jsonl,
    chunk_text, clear_gpu, set_seed, stable_int, DEVICE
)

set_seed(CONFIG["random_seed"])

# ============================================================
# MULTI-DATASET LOADER
# ============================================================
class MultiDatasetLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        self.all_data = {}

    def load_all(self):
        self._load_pubmedqa()
        self._load_medmcqa()
        return self.all_data

    def _load_pubmedqa(self):
        print("[DATA] PubMedQA ...")
        try:
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        except Exception:
            ds = load_dataset("qiaojin/PubMedQA", split="train")
        recs = []
        for i, ex in enumerate(tqdm(ds, desc="PubMedQA")):
            q = str(ex.get("question", "")).strip()
            ctx = ex.get("context", None)
            contexts = []
            if isinstance(ctx, dict):
                contexts = ctx.get("contexts", [])
            elif isinstance(ctx, list):
                contexts = ctx
            elif isinstance(ctx, str):
                contexts = [ctx]
            doc = " ".join([str(c).strip() for c in contexts if str(c).strip()])
            long_a = str(ex.get("long_answer") or ex.get("final_answer") or "").strip()
            dec = str(ex.get("final_decision") or ex.get("label") or "").strip()
            ref = long_a if long_a else dec
            if q and doc:
                recs.append({
                    "id": f"pqa_{i}",
                    "question": q,
                    "doc_text": doc,
                    "reference": ref,
                    "decision": dec,
                    "raw_contexts": contexts,
                    "dataset": "pubmedqa"
                })
        self._split_pubmedqa(recs)

    def _split_pubmedqa(self, recs):
        rng = random.Random(CONFIG["random_seed"])
        rng.shuffle(recs)
        n_c = min(self.cfg["corpus_size"], int(len(recs) * 0.7))
        n_t = min(self.cfg["max_test_per_dataset"], len(recs) - n_c)
        self.all_data["pubmedqa"] = {
            "corpus": recs[:n_c],
            "test": recs[n_c:n_c + n_t]
        }
        print(f"[DATA] pubmedqa: {n_c} corpus abstracts + {n_t} test questions")

    def _medmcqa_record(self, ex, i, source_tag):
        q = str(ex.get("question", "")).strip()
        opts = {
            "A": str(ex.get("opa", "")),
            "B": str(ex.get("opb", "")),
            "C": str(ex.get("opc", "")),
            "D": str(ex.get("opd", ""))
        }
        cop_raw = ex.get("cop", 0)
        ak = ["A", "B", "C", "D"][int(cop_raw)] if cop_raw is not None else "A"
        expl = str(ex.get("exp", "")).strip()
        opt_txt = " | ".join([f"{k}: {v}" for k, v in opts.items()])
        ref = f"The correct answer is {ak}: {opts[ak]}. {expl}".strip()
        question = f"{q}\nOptions: {opt_txt}"
        if len(expl.strip()) > 50:
            doc_text = expl.strip()
        else:
            doc_text = f"{q} {opt_txt} Correct answer: {opts[ak]}. {expl}".strip()
        return {
            "id": f"medmcqa_{source_tag}_{i}",
            "question": question,
            "doc_text": doc_text,
            "reference": ref,
            "decision": ak,
            "raw_explanation": expl,
            "dataset": "medmcqa"
        }

    def _load_medmcqa(self):
        print("[DATA] MedMCQA ...")
        try:
            val_ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        except Exception:
            try:
                val_ds = load_dataset("medmcqa", split="validation")
            except Exception:
                print("[WARN] MedMCQA validation load failed -> placeholder")
                self.all_data["medmcqa"] = {"corpus": [], "test": []}
                return
        val_ds = val_ds.shuffle(seed=CONFIG["random_seed"])
        n_test_select = min(len(val_ds), max(CONFIG["max_test_per_dataset"], 200))
        val_subset = val_ds.select(range(n_test_select))
        test_recs = []
        for i, ex in enumerate(tqdm(val_subset, desc="MedMCQA test")):
            test_recs.append(self._medmcqa_record(ex, i, "val"))
        test_recs = test_recs[:CONFIG["max_test_per_dataset"]]

        corpus_recs = []
        if CONFIG.get("medmcqa_corpus_split") == "train":
            try:
                train_ds = load_dataset("openlifescienceai/medmcqa", split="train")
            except Exception:
                try:
                    train_ds = load_dataset("medmcqa", split="train")
                except Exception:
                    train_ds = None
            if train_ds is not None:
                train_ds = train_ds.shuffle(seed=CONFIG["random_seed"])
                n_pool = min(len(train_ds), CONFIG["medmcqa_train_pool_size"])
                train_subset = train_ds.select(range(n_pool))
                for i, ex in enumerate(tqdm(train_subset, desc="MedMCQA train corpus")):
                    corpus_recs.append(self._medmcqa_record(ex, i, "train"))

        if len(corpus_recs) == 0:
            print("[WARN] Using MedMCQA validation fallback as exam corpus proxy")
            for i, ex in enumerate(tqdm(val_ds, desc="MedMCQA fallback corpus")):
                corpus_recs.append(self._medmcqa_record(ex, i, "valfallback"))

        rng = random.Random(CONFIG["random_seed"] + 1)
        if len(corpus_recs) > CONFIG["corpus_size"] * 2:
            corpus_recs = rng.sample(corpus_recs, CONFIG["corpus_size"] * 2)

        self.all_data["medmcqa"] = {
            "corpus": corpus_recs,
            "test": test_recs
        }
        print(f"[DATA] medmcqa: {len(corpus_recs)} corpus explanations + {len(test_recs)} test questions")


# ============================================================
# CORPUS CONDITION POOLS
# ============================================================
def load_external_textbook(path):
    if not path or not os.path.exists(path):
        return None
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                txt = d.get("text") or d.get("doc_text") or ""
                if len(txt.strip()) > 20:
                    docs.append({"id": f"exttb_{len(docs)}", "doc_text": txt.strip(), "decision": ""})
            except Exception:
                pass
    return docs if docs else None

def sample_docs(pool, n, seed):
    if len(pool) <= n:
        return pool
    return random.Random(seed).sample(pool, n)

def make_partial_pool(aligned_pool, misaligned_pool, seed):
    rng = random.Random(seed)
    n = min(len(aligned_pool), len(misaligned_pool))
    n_a = int(n * CONFIG.get("partial_mix_ratio", 0.5))
    n_m = n - n_a
    part = aligned_pool[:n_a] + misaligned_pool[:n_m]
    rng.shuffle(part)
    return part

def build_corpus_pools(all_data):
    print("\n[POOL] Building corpus-condition pools ...")
    pubmedqa_abstract_pool = []
    for doc in all_data["pubmedqa"]["corpus"]:
        t = str(doc.get("doc_text", "")).strip()
        if len(t) > 20:
            pubmedqa_abstract_pool.append({
                "id": doc["id"],
                "doc_text": t,
                "decision": doc.get("decision", "")
            })
    ext_textbook = load_external_textbook(CONFIG.get("external_textbook_path"))
    if ext_textbook:
        print("[POOL] Using REAL external textbook/exam corpus")
        textbook_pool = ext_textbook
        textbook_source = "external_file"
    else:
        print("[POOL] Using MedMCQA train 'exp' field as textbook/exam proxy")
        textbook_pool = []
        for doc in all_data["medmcqa"]["corpus"]:
            t = str(doc.get("doc_text", "")).strip()
            if len(t) > 20:
                textbook_pool.append({
                    "id": doc["id"],
                    "doc_text": t,
                    "decision": doc.get("decision", "")
                })
        textbook_source = "medmcqa_train_explanations"

    print(f"[POOL] PubMedQA abstracts available: {len(pubmedqa_abstract_pool)}")
    print(f"[POOL] Textbook/exam docs available: {len(textbook_pool)} (source: {textbook_source})")

    if CONFIG.get("equalize_corpus_docs", True):
        target_docs = min(len(pubmedqa_abstract_pool), len(textbook_pool), CONFIG["corpus_size"])
        pubmedqa_abstract_pool = sample_docs(pubmedqa_abstract_pool, target_docs, CONFIG["random_seed"] + 10)
        textbook_pool = sample_docs(textbook_pool, target_docs, CONFIG["random_seed"] + 11)
        print(f"[POOL] Balanced document count per condition: {target_docs}")

    CORPUS_CONFIG = {
        "pubmedqa_aligned": {
            "docs": pubmedqa_abstract_pool,
            "provenance": {
                "dataset": "PubMedQA",
                "field": "context (full abstract)",
                "alignment": "aligned",
                "n_docs": len(pubmedqa_abstract_pool)
            }
        },
        "pubmedqa_misaligned": {
            "docs": textbook_pool,
            "provenance": {
                "dataset": "MedMCQA" if textbook_source == "medmcqa_train_explanations" else "External",
                "field": "exp (explanation)" if textbook_source == "medmcqa_train_explanations" else "text",
                "alignment": "misaligned",
                "n_docs": len(textbook_pool)
            }
        },
        "pubmedqa_partially_aligned": {
            "docs": make_partial_pool(pubmedqa_abstract_pool, textbook_pool, CONFIG["random_seed"] + 20),
            "provenance": {
                "mixture": "50% PubMed abstracts + 50% textbook explanations",
                "alignment": "partially_aligned",
                "n_docs": len(make_partial_pool(pubmedqa_abstract_pool, textbook_pool, CONFIG["random_seed"] + 20))
            }
        },
        "medmcqa_aligned": {
            "docs": textbook_pool,
            "provenance": {
                "dataset": "MedMCQA" if textbook_source == "medmcqa_train_explanations" else "External",
                "field": "exp (explanation)" if textbook_source == "medmcqa_train_explanations" else "text",
                "alignment": "aligned",
                "n_docs": len(textbook_pool)
            }
        },
        "medmcqa_misaligned": {
            "docs": pubmedqa_abstract_pool,
            "provenance": {
                "dataset": "PubMedQA",
                "field": "context (full abstract)",
                "alignment": "misaligned",
                "n_docs": len(pubmedqa_abstract_pool)
            }
        },
        "medmcqa_partially_aligned": {
            "docs": make_partial_pool(textbook_pool, pubmedqa_abstract_pool, CONFIG["random_seed"] + 21),
            "provenance": {
                "mixture": "50% textbook explanations + 50% PubMed abstracts",
                "alignment": "partially_aligned",
                "n_docs": len(make_partial_pool(textbook_pool, pubmedqa_abstract_pool, CONFIG["random_seed"] + 21))
            }
        },
    }

    provenance_manifest = {
        "experiment": "BioTrust-RAG Corpus-Swap",
        "design": "within_dataset_corpus_manipulation",
        "pools": {
            "pubmedqa_abstracts": {
                "source": "PubMedQA pqa_labeled train split",
                "field": "context (concatenated full abstract)",
                "n_documents": len(pubmedqa_abstract_pool),
                "description": "Clinical research abstracts"
            },
            "textbook_explanations": {
                "source": "MedMCQA train split" if textbook_source == "medmcqa_train_explanations" else "External file",
                "field": "exp (explanation)" if textbook_source == "medmcqa_train_explanations" else "user_provided",
                "n_documents": len(textbook_pool),
                "description": "Medical textbook/examination explanations"
            }
        },
        "conditions": {k: v["provenance"] for k, v in CORPUS_CONFIG.items()},
        "balancing": {
            "equalize_documents": CONFIG.get("equalize_corpus_docs", True),
            "target_documents": len(pubmedqa_abstract_pool),
            "partial_mix_ratio": CONFIG.get("partial_mix_ratio", 0.5)
        }
    }
    save_json(provenance_manifest, os.path.join(CONFIG["output_dir"], "corpus", "corpus_provenance_manifest.json"))
    print("[OK] corpus provenance manifest saved")
    return CORPUS_CONFIG, pubmedqa_abstract_pool, textbook_pool, textbook_source


# ============================================================
# CHUNK INDEX PER CORPUS CONDITION
# ============================================================
def build_corpus_chunks(key, docs):
    cache = os.path.join(CONFIG["output_dir"], "corpus", f"chunks_{key}.csv")
    if os.path.exists(cache):
        try:
            df = pd.read_csv(cache)
            if len(df) > 0:
                print(f"[CHUNKS] {key}: cached {len(df)}")
                return df
        except Exception:
            os.remove(cache)
    if not docs:
        return pd.DataFrame(columns=["chunk_id", "doc_id", "decision", "text"])
    recs = []
    for doc in tqdm(docs, desc=f"Chunk {key}", leave=False):
        txt = doc.get("doc_text", "") or doc.get("text", "")
        if not txt:
            continue
        for ch in chunk_text(txt, CONFIG["chunk_size"], CONFIG["chunk_overlap"]):
            recs.append({
                "chunk_id": len(recs),
                "doc_id": doc.get("id", key),
                "decision": doc.get("decision", ""),
                "text": ch
            })
    df = pd.DataFrame(recs)
    df.to_csv(cache, index=False)
    print(f"[CHUNKS] {key}: {len(df)} chunks")
    return df

def build_all_chunks(CORPUS_CONFIG):
    corpus_chunks = {}
    for ds in CONFIG["datasets"]:
        for cc in CONFIG["corpus_conditions"][ds]:
            key = f"{ds}_{cc}"
            docs = CORPUS_CONFIG.get(key, {}).get("docs", [])
            if not docs:
                print(f"[SKIP] {key}")
                continue
            corpus_chunks[key] = build_corpus_chunks(key, docs)

    if CONFIG.get("equalize_corpus_chunks", True):
        print("\n[CHUNK BALANCE] Equalizing chunk counts within each dataset ...")
        for ds in CONFIG["datasets"]:
            keys = [f"{ds}_{cc}" for cc in CONFIG["corpus_conditions"][ds] if f"{ds}_{cc}" in corpus_chunks]
            if len(keys) == 0:
                continue
            lens = [len(corpus_chunks[k]) for k in keys]
            target = min(lens)
            target = min(target, CONFIG.get("max_chunks_per_corpus", 726))
            if target <= 0:
                continue
            for k in keys:
                if len(corpus_chunks[k]) > target:
                    corpus_chunks[k] = corpus_chunks[k].sample(n=target, random_state=CONFIG["random_seed"]).reset_index(drop=True)
            print(f"[CHUNK BALANCE] {ds}: target chunks per condition = {target}")

    corpus_stats_rows = []
    for key, df in corpus_chunks.items():
        if len(df) == 0:
            continue
        corpus_stats_rows.append({
            "corpus_key": key,
            "docs": len(CORPUS_CONFIG.get(key, {}).get("docs", [])),
            "chunks": len(df),
            "mean_chunk_words": float(df["text"].astype(str).str.split().str.len().mean())
        })
    corpus_stats_df = pd.DataFrame(corpus_stats_rows)
    corpus_stats_path = os.path.join(CONFIG["output_dir"], "statistics", "corpus_statistics.csv")
    corpus_stats_df.to_csv(corpus_stats_path, index=False)
    print("\n[OK] chunks built and balanced:")
    for k in sorted(corpus_chunks.keys()):
        print(f"   {k}: {len(corpus_chunks[k])} chunks")
    print("[OK] corpus statistics saved")
    return corpus_chunks

# ============================================================
# LEAKAGE AUDIT
# ============================================================
def run_leakage_audit(test_questions, corpus_docs):
    audit = {"exact_5gram_matches": 0, "high_jaccard_chunks": 0, "flagged_texts": []}
    for q in test_questions:
        q_words = q["question"].split()
        for doc in corpus_docs:
            doc_words = doc["doc_text"].split()
            q_5grams = set(zip(q_words, q_words[1:], q_words[2:], q_words[3:], q_words[4:])) if len(q_words)>=5 else set()
            d_5grams = set(zip(doc_words, doc_words[1:], doc_words[2:], doc_words[3:], doc_words[4:])) if len(doc_words)>=5 else set()
            if len(q_5grams) > 0 and len(d_5grams) > 0:
                inter = len(q_5grams & d_5grams)
                union = len(q_5grams | d_5grams)
                if inter / union > 0.15:
                    audit["high_jaccard_chunks"] += 1
                    if len(doc["doc_text"]) < 500:
                        audit["flagged_texts"].append(doc["doc_text"][:100] + "...")
    return audit

# ============================================================
# ALIGNMENT VALIDATION (TF‑IDF + optional MedCPT)
# ============================================================
def validate_alignment(corpus_chunks, all_data, n_probe=None):
    n_probe = n_probe or CONFIG["alignment_probe_n"]
    rows = []
    all_corpus_texts = []
    for key, df_chunks in corpus_chunks.items():
        all_corpus_texts.extend(df_chunks["text"].astype(str).tolist())
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectorizer.fit(all_corpus_texts)

    for ds in CONFIG["datasets"]:
        for cc in CONFIG["corpus_conditions"][ds]:
            key = f"{ds}_{cc}"
            if key not in corpus_chunks:
                continue
            df_chunks = corpus_chunks[key]
            if len(df_chunks) == 0:
                continue
            test_items = all_data[ds]["test"][:n_probe]
            if not test_items:
                continue
            chunk_vect = vectorizer.transform(df_chunks["text"].astype(str).tolist())
            per_q_avg = []
            for item in test_items:
                q_vect = vectorizer.transform([item["question"]])
                sims = cosine_similarity(q_vect, chunk_vect).flatten()
                top3_indices = np.argsort(sims)[-3:][::-1]
                top3_sims = sims[top3_indices]
                per_q_avg.append(np.mean(top3_sims))
            mean_tfidf = float(np.mean(per_q_avg)) if per_q_avg else 0.0

            # Optional MedCPT validation
            mean_medcpt = 0.0
            try:
                val_encoder = SentenceTransformer("ncbi/MedCPT-Query-Encoder", device=DEVICE)
                chunk_embs = val_encoder.encode(df_chunks["text"].astype(str).tolist(),
                                                batch_size=64, normalize_embeddings=True,
                                                show_progress_bar=False)
                med_per_q = []
                for item in test_items:
                    q_emb = val_encoder.encode([item["question"]], normalize_embeddings=True)
                    sims = np.dot(chunk_embs, q_emb.T).flatten()
                    top3_sims = np.sort(sims)[-3:][::-1]
                    med_per_q.append(np.mean(top3_sims))
                mean_medcpt = float(np.mean(med_per_q)) if med_per_q else 0.0
                del val_encoder
                clear_gpu()
            except Exception as e:
                print(f"  [WARN] MedCPT validation skipped: {e}")
                mean_medcpt = mean_tfidf

            rows.append({
                "dataset": ds,
                "corpus_condition": cc,
                "n_probe": len(test_items),
                "mean_top3_relevance": mean_tfidf,
                "mean_top3_medcpt_cosine": mean_medcpt
            })

    val_df = pd.DataFrame(rows)
    val_path = os.path.join(CONFIG["output_dir"], "statistics", "alignment_validation.csv")
    val_df.to_csv(val_path, index=False)
    print("\n[ALIGNMENT VALIDATION]")
    print(val_df.to_string(index=False))
    return val_df

def build_alignment_human_template(corpus_chunks, all_data, n_per_cell=None):
    n_per_cell = n_per_cell or CONFIG["alignment_human_samples_per_cell"]
    rows = []
    for ds in CONFIG["datasets"]:
        for cc in CONFIG["corpus_conditions"][ds]:
            key = f"{ds}_{cc}"
            if key not in corpus_chunks:
                continue
            df_chunks = corpus_chunks[key]
            if len(df_chunks) == 0:
                continue
            test_items = all_data[ds]["test"][:n_per_cell]
            for item in test_items:
                question = item["question"]
                # For template, we take top 3 chunks (placeholder)
                ids = list(range(min(3, len(df_chunks))))
                scores = [0.0] * len(ids)
                for rank, i in enumerate(ids, start=1):
                    rows.append({
                        "dataset": ds,
                        "corpus_condition": cc,
                        "qid": item["id"],
                        "question": question,
                        "rank": rank,
                        "passage": df_chunks.iloc[i]["text"] if i < len(df_chunks) else "",
                        "retrieval_score": scores[rank-1] if rank-1 < len(scores) else None,
                        "human_relevance_0to2": "",
                        "rater_name": "",
                        "comments": ""
                    })
    df = pd.DataFrame(rows)
    path = os.path.join(CONFIG["output_dir"], "human_eval", "alignment_human_template.csv")
    df.to_csv(path, index=False)
    print(f"[ALIGNMENT HUMAN] template saved: {len(df)} rows")
    return df

# ============================================================
# CORPUS PROVENANCE REPORT
# ============================================================
def generate_corpus_report(CORPUS_CONFIG, corpus_chunks, pubmedqa_abstract_pool, textbook_pool):
    report_lines = [
        "Corpus-Task Alignment: Source Documentation",
        "-" * 50,
        "",
        "Aligned corpus for PubMedQA:",
        "  Source:  PubMedQA dataset (qiaojin/PubMedQA, pqa_labeled train split)",
        "  Field:   context.contexts (concatenated into full abstracts)",
        "  Type:    Clinical research abstracts",
        f"  N docs:  {len(pubmedqa_abstract_pool)}",
        "",
        "Aligned corpus for MedMCQA:",
        "  Source:  MedMCQA dataset (openlifescienceai/medmcqa, train split)",
        "  Field:   exp (explanation)",
        "  Type:    Medical textbook / examination explanations",
        f"  N docs:  {len(textbook_pool)}",
        "",
        "Reciprocal (misaligned) conditions:",
        "  PubMedQA misaligned  -> uses MedMCQA explanations",
        "  MedMCQA misaligned   -> uses PubMedQA abstracts",
        "",
        "Partially aligned conditions:",
        "  50% document-level mixture of the two pools above",
        "",
        "Balancing:",
        f"  Document count per condition: {len(pubmedqa_abstract_pool)}",
        f"  Chunk count per condition:    {next(iter(corpus_chunks.values())) if corpus_chunks else 'N/A'}",
    ]
    report_text = "\n".join(report_lines)
    with open(os.path.join(CONFIG["output_dir"], "corpus", "CORPUS_SOURCES_README.txt"), "w") as f:
        f.write(report_text)
    print("\n[OK] Corpus source report saved to corpus/CORPUS_SOURCES_README.txt")
    return report_text

# ============================================================
# RETRIEVAL SYSTEM
# ============================================================
class RetrievalSystem:
    def __init__(self, chunk_texts, corpus_key, base_ds, device=DEVICE):
        self.chunk_texts = chunk_texts
        self.corpus_key = corpus_key
        self.base_ds = base_ds
        self.device = device
        self.bm25 = None
        self.faiss_minilm = None
        self.faiss_medcpt = None
        self.encoder_minilm = None
        self.medcpt_query = None
        self.medcpt_query_tok = None
        self.reranker = None

    def build_bm25(self):
        self.bm25 = BM25Okapi([t.lower().split() for t in self.chunk_texts], k1=1.5, b=0.75)

    def build_minilm(self):
        self.encoder_minilm = SentenceTransformer(CONFIG["minilm_encoder"], device=self.device)
        e = np.asarray(self.encoder_minilm.encode(self.chunk_texts, batch_size=128,
                     normalize_embeddings=True, show_progress_bar=False), "float32")
        self.faiss_minilm = faiss.IndexFlatIP(e.shape[1])
        self.faiss_minilm.add(e)

    def build_medcpt(self):
        try:
            self.medcpt_query_tok = AutoTokenizer.from_pretrained(CONFIG["medcpt_query"])
            self.medcpt_query = AutoModel.from_pretrained(CONFIG["medcpt_query"]).to(self.device)
            at = AutoTokenizer.from_pretrained(CONFIG["medcpt_article"])
            am = AutoModel.from_pretrained(CONFIG["medcpt_article"]).to(self.device).eval()
            embs = []
            with torch.no_grad():
                for i in tqdm(range(0, len(self.chunk_texts), 32), desc="MedCPT", leave=False):
                    tok = at(self.chunk_texts[i:i+32], padding=True, truncation=True,
                             max_length=512, return_tensors="pt").to(self.device)
                    embs.append(am(**tok).last_hidden_state[:, 0, :].cpu().numpy())
            del am; clear_gpu()
            e = np.vstack(embs).astype("float32")
            e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
            self.faiss_medcpt = faiss.IndexFlatIP(e.shape[1])
            self.faiss_medcpt.add(e)
        except Exception as ex:
            print(f"  [WARN] MedCPT failed ({ex}) -> MiniLM fallback")
            if self.faiss_minilm is None:
                self.build_minilm()
            self.faiss_medcpt = self.faiss_minilm

    def build_reranker(self):
        try:
            self.reranker = CrossEncoder(CONFIG["medcpt_cross"], device=self.device, max_length=512)
        except:
            self.reranker = CrossEncoder(CONFIG["rerank_fallback"], device=self.device, max_length=512)

    def _qvec(self, q):
        self.medcpt_query.eval()
        with torch.no_grad():
            tok = self.medcpt_query_tok(q, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            e = self.medcpt_query(**tok).last_hidden_state[:, 0, :].cpu().numpy().astype("float32")
        n = np.linalg.norm(e)
        return e / n if n else e

    def retrieve(self, query, retriever, top_k=None):
        top_k = top_k or CONFIG["top_k"]
        N = len(self.chunk_texts)
        if retriever == "no_retrieval":
            return [], []
        if retriever == "random":
            seed = CONFIG["random_seed"] + stable_int(query, 100000)
            rng = random.Random(seed)
            ids = rng.sample(range(N), min(top_k, N))
            return ids, [0.0] * len(ids)
        if retriever == "bm25":
            s = self.bm25.get_scores(query.lower().split())
            ix = np.argsort(s)[::-1][:top_k]
            return ix.tolist(), s[ix].tolist()
        if retriever == "faiss_minilm":
            e = np.asarray(self.encoder_minilm.encode([query], normalize_embeddings=True), "float32")
            sc, ix = self.faiss_minilm.search(e, top_k)
            return ix[0].tolist(), sc[0].tolist()
        if retriever == "faiss_medcpt":
            sc, ix = self.faiss_medcpt.search(self._qvec(query), top_k)
            return ix[0].tolist(), sc[0].tolist()
        if retriever == "hybrid_medcpt":
            bs = self.bm25.get_scores(query.lower().split())
            bt = np.argsort(bs)[::-1][:CONFIG["candidate_pool"]]
            _, mi = self.faiss_medcpt.search(self._qvec(query), CONFIG["candidate_pool"])
            rrf = {}
            for r, i in enumerate(bt):
                rrf[i] = rrf.get(i, 0) + 1.0 / (CONFIG["rrf_k"] + r + 1)
            for r, i in enumerate(mi[0]):
                if i >= 0:
                    rrf[i] = rrf.get(i, 0) + 1.0 / (CONFIG["rrf_k"] + r + 1)
            top = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return [i for i, _ in top], [s for _, s in top]
        if retriever == "medcpt_rerank":
            if self.faiss_medcpt is not None:
                _, ix = self.faiss_medcpt.search(self._qvec(query), CONFIG["rerank_topk"])
                cand = [int(i) for i in ix[0] if i >= 0]
            else:
                bs = self.bm25.get_scores(query.lower().split())
                cand = np.argsort(bs)[::-1][:CONFIG["rerank_topk"]].tolist()
            if not cand:
                return [], []
            if self.reranker is not None:
                pairs = [(query, self.chunk_texts[i]) for i in cand]
                rs = self.reranker.predict(pairs, batch_size=8, show_progress_bar=False)
                top = np.argsort(rs)[::-1][:top_k]
                return [cand[i] for i in top], [float(rs[i]) for i in top]
            return cand[:top_k], [0.0] * min(top_k, len(cand))
        return [], []

    def apply_robustness(self, chunk_ids, chunk_scores, mode):
        if mode is None or not chunk_ids:
            return chunk_ids, chunk_scores
        ids = list(chunk_ids); sc = list(chunk_scores); N = len(self.chunk_texts)
        if mode == "noisy1":
            seed = CONFIG["random_seed"] + stable_int(str(ids), 50000)
            rng = random.Random(seed)
            ids[-1] = rng.choice(range(N))
            sc[-1] = 0.0
        elif mode == "contradictory":
            seed = CONFIG["random_seed"] + stable_int(str(ids), 70000)
            rng = random.Random(seed)
            ids.insert(0, rng.choice(range(N)))
            sc.insert(0, 0.0)
            ids = ids[:CONFIG["top_k"]]; sc = sc[:CONFIG["top_k"]]
        elif mode == "missing_evidence":
            ids = ids[:1]; sc = sc[:1]
        return ids, sc
