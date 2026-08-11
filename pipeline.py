import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import faiss
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import necessary functions/classes from other modules
from config import (
    MODEL_REGISTRY, RETRIEVERS, INCLUDE_ROBUSTNESS, ROBUSTNESS_MODES,
    TOP_K, CONTRADICTION_THRESHOLD, FAITHFULNESS_MEDIUM, SEED,
    SIMULATE_HUMAN_RATINGS, MULTIPLE_TESTING_METHOD
)
from corpus import chunk_text # Assuming chunk_text is used here for claim splitting if needed

# --- Helper Functions ---

def load_jsonl(file_path: str) -> List[Dict]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        print(f"[INFO] File {file_path} not found, returning empty list.")
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """Saves a list of dictionaries to a JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def split_claims(text: str) -> List[str]:
    """
    Simple heuristic to split generated text into individual claims.
    This is often a complex NLP task; this is a basic placeholder.
    Consider using more sophisticated sentence segmentation tools if needed.
    """
    # Split by common sentence endings. This is a simple example.
    sentences = text.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    # Further refine if necessary, e.g., handle abbreviations
    return sentences


# --- LLM Loading and Handling ---

def load_llm(model_name: str):
    """Loads a specified language model and tokenizer."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in MODEL_REGISTRY")

    model_info = MODEL_REGISTRY[model_name]
    repo = model_info["repo"]
    quantize = model_info["quantize"]

    tokenizer = AutoTokenizer.from_pretrained(repo)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {}
    if quantize:
        model_kwargs.update({
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        })
    else:
        model_kwargs.update({
            "torch_dtype": torch.float16 if "cuda" in torch.cuda.current_device().__str__() else torch.float32,
            "device_map": "auto"
        })

    model = AutoModel.from_pretrained(repo, **model_kwargs)

    meta = {
        "name": model_name,
        "display_name": model_info.get("display", model_name),
        "parameters_billions": model_info.get("params_b", 0),
        "max_input_length": model_info.get("max_input", 512)
    }

    return model, tokenizer, meta


def clear_gpu():
    """Clears GPU cache if using CUDA."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Retrieval System ---

class RetrievalSystem:
    """
    Encapsulates BM25, dense embedding (MiniLM, MedCPT), and reranking retrieval methods.
    """
    def __init__(self, chunk_texts: List[str], corpus_key: str, base_ds: str, device: str = "cuda"):
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
        """Builds the BM25 index."""
        self.bm25 = BM25Okapi([t.lower().split() for t in self.chunk_texts], k1=1.5, b=0.75)

    def build_minilm(self):
        """Builds the MiniLM dense retrieval index."""
        self.encoder_minilm = SentenceTransformer("all-MiniLM-L6-v2", device=self.device) # Use config path
        embeddings = np.asarray(
            self.encoder_minilm.encode(
                self.chunk_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32
        )
        self.faiss_minilm = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_minilm.add(embeddings)

    def build_medcpt(self):
        """Builds the MedCPT dense retrieval index."""
        try:
            self.medcpt_query_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder") # Use config path
            self.medcpt_query = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self.device) # Use config path
            am_tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder") # Use config path
            am_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(self.device).eval() # Use config path

            embeddings_list = []
            with torch.no_grad():
                for i in tqdm(range(0, len(self.chunk_texts), 32), desc="Encoding MedCPT Corpus", leave=False):
                    batch_texts = self.chunk_texts[i:i + 32]
                    tok = am_tok(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                    emb_batch = am_model(**tok).last_hidden_state.mean(dim=1) # Mean pooling
                    embeddings_list.append(emb_batch.cpu().numpy())
            
            embeddings = np.vstack(embeddings_list).astype('float32')
            self.faiss_medcpt = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_medcpt.add(embeddings)
            self._embeddings_corpus = embeddings # Store for later use in reranking or other ops if needed
        except Exception as e:
            print(f"[ERROR] Failed to build MedCPT index: {e}")
            self.faiss_medcpt = None # Fallback

    def build_reranker(self):
        """Builds the cross-encoder reranker."""
        try:
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device) # Use config path
        except Exception as e:
            print(f"[ERROR] Failed to build reranker: {e}")
            self.reranker = None

    def retrieve(self, query: str, method: str, top_k: int) -> Tuple[List[int], List[float]]:
        """Performs retrieval based on the specified method."""
        if method == "bm25":
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:top_k]
            top_scores = scores[top_indices]
        elif method == "faiss_minilm":
            query_embedding = self.encoder_minilm.encode([query], normalize_embeddings=True).astype('float32')
            scores, indices = self.faiss_minilm.search(query_embedding, top_k)
            top_indices = indices[0]
            top_scores = scores[0]
        elif method == "faiss_medcpt":
            if self.faiss_medcpt is None:
                return [], [] # Fallback if MedCPT failed to build
            query_inputs = self.medcpt_query_tok(query, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                query_emb = self.medcpt_query(**query_inputs).last_hidden_state.mean(dim=1).cpu().numpy().astype('float32')
            scores, indices = self.faiss_medcpt.search(query_emb, top_k)
            top_indices = indices[0]
            top_scores = scores[0]
        elif method == "medcpt_rerank":
             if self.reranker is None:
                 # Fallback to just MedCPT dense retrieval if reranker fails
                 return self.retrieve(query, "faiss_medcpt", top_k)
             # First, get top candidates using MedCPT dense retrieval
             dense_indices, dense_scores = self.retrieve(query, "faiss_medcpt", top_k * 2) # Retrieve more for reranking
             if not dense_indices:
                 return [], []
             candidate_texts = [self.chunk_texts[i] for i in dense_indices]
             # Rerank the candidates
             rerank_scores = self.reranker.predict([(query, ct) for ct in candidate_texts])
             # Sort by rerank score and take top_k
             reranked_indices = np.argsort(rerank_scores)[::-1][:top_k]
             final_indices = [dense_indices[i] for i in reranked_indices]
             final_scores = [rerank_scores[i] for i in reranked_indices]
             return final_indices, final_scores
        elif method == "no_retrieval":
            return [], []
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Filter out invalid scores (e.g., -inf)
        valid_mask = top_scores != float('-inf')
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]

        # Return only the requested top_k (or fewer if available)
        top_indices = top_indices[:top_k]
        top_scores = top_scores[:top_k]

        return top_indices.tolist(), top_scores.tolist()

    def apply_robustness(self, retrieved_ids: List[int], retrieved_scores: List[float], robustness_mode: str) -> Tuple[List[int], List[float]]:
        """Applies noise, contradiction, or missing evidence to retrieved results."""
        if robustness_mode == "noisy1":
            # Add random noise to scores
            noise = np.random.normal(0, 0.1, size=len(retrieved_scores)) # Mean 0, std 0.1
            noisy_scores = np.array(retrieved_scores) + noise
            # Re-sort based on noisy scores
            sorted_indices = np.argsort(noisy_scores)[::-1]
            return [retrieved_ids[i] for i in sorted_indices], noisy_scores[sorted_indices].tolist()
        elif robustness_mode == "contradictory":
            # Simulate contradiction by potentially flipping top results or adding low-scoring contradicting ones
            # This is a placeholder - actual implementation depends on how contradictions are identified/introduced
            # For now, let's just slightly perturb the order/scores
            perturbed_scores = [s * (1 + np.random.uniform(-0.2, 0.1)) for s in retrieved_scores] # Up to -20% or +10%
            sorted_indices = np.argsort(perturbed_scores)[::-1]
            return [retrieved_ids[i] for i in sorted_indices], [perturbed_scores[i] for i in sorted_indices]
        elif robustness_mode == "missing_evidence":
            # Simulate missing evidence by removing top results
            # Keep only a few (e.g., 1) or none
            keep_n = 1 # Could be configurable
            if len(retrieved_ids) > keep_n:
                 ids = retrieved_ids[:keep_n]
                 sc = retrieved_scores[:keep_n]
                 return ids, sc
            else:
                 # If less than keep_n, return as is
                 return retrieved_ids, retrieved_scores
        else:
            print(f"[WARN] Unknown robustness mode {robustness_mode}, returning original results.")
            return retrieved_ids, retrieved_scores


# --- Faithfulness Checker (NLI-based) ---

class NLIHelper:
    """Helper class for Natural Language Inference checks."""
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-base", device: str = "cuda"):
        self.model = CrossEncoder(model_name, device=device)
        # Map labels to indices (depends on the specific NLI model)
        # Common labels are ['contradiction', 'entailment', 'neutral']
        # You need to verify the exact mapping for your chosen model
        # For 'cross-encoder/nli-deberta-base': 0: contradiction, 1: entailment, 2: neutral
        self.contra_idx = 0
        self.ent_idx = 1
        self.neut_idx = 2

    def predict_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 8) -> np.ndarray:
        """Predicts NLI labels for pairs of (premise, hypothesis)."""
        scores = self.model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
        return scores # Shape: (num_pairs, num_labels)

# Global NLI helper instance (or pass it around as needed)
nli = NLIHelper()


# --- Generation Loop ---

def run_generation_loop(models: List[str], datasets: Dict[str, Dict[str, List[Dict]]], retrieval_systems: Dict[str, RetrievalSystem], config: Dict[str, Any]):
    """
    Executes the main generation loop for all models, datasets, and conditions.
    """
    all_conditions = list(config["RETRIEVERS"])
    if config.get("INCLUDE_ROBUSTNESS"):
        all_conditions += config.get("ROBUSTNESS_MODES", [])

    n_models = len(models)
    n_datasets = len(datasets)
    n_conditions = len(all_conditions)
    n_corpus_conditions = sum(len(config["CORPUS_CONDITIONS"].get(ds, [])) for ds in datasets)
    total_gens = n_models * n_corpus_conditions * n_conditions * sum(len(ds["test"]) for ds in datasets.values())

    print(f"Starting experiment: {n_models} models x {n_conditions} conditions x {n_corpus_conditions} corpus conditions")
    print(f"Estimated total generations: {total_gens:,}")

    for model_name in models:
        print(f"{'='*60}\nMODEL: {model_name}\n{'='*60}")
        llm, tokenizer, meta = load_llm(model_name)
        mkey = model_name # Use model name as key, or derive a shorter one if needed

        for ds_name, ds_data in datasets.items():
            test_questions = ds_data.get("test", [])
            if not test_questions:
                 print(f" [SKIP] No test questions for {ds_name}")
                 continue
            print(f" DATASET: {ds_name} ({len(test_questions)} questions)")

            for cc in config["CORPUS_CONDITIONS"].get(ds_name, []):
                rs_key = f"{ds_name}_{cc}"
                if rs_key not in retrieval_systems:
                     print(f" [SKIP] Retrieval system not found for {rs_key}")
                     continue

                rs = retrieval_systems[rs_key]
                print(f" CORPUS: {cc} (using retrieval system {rs_key})")

                for cond in all_conditions:
                    gen_dir = config["GENERATION_DIR"] / ds_name / cc / mkey
                    gen_dir.mkdir(parents=True, exist_ok=True)
                    out_path = gen_dir / f"{cond}.jsonl"

                    # Load existing generations to avoid re-computing
                    done_ids_set = set()
                    existing_records = load_jsonl(out_path)
                    for r in existing_records:
                        qid = r.get("qid")
                        if qid is not None:
                            done_ids_set.add(qid)

                    if len(done_ids_set) >= len(test_questions):
                         print(f" {cond}: All {len(test_questions)} generations already exist, skipping.")
                         continue
                    else:
                         print(f" {cond}: Found {len(done_ids_set)} cached, generating {len(test_questions) - len(done_ids_set)}...")

                    for q in tqdm(test_questions, desc=f"{ds_name}/{cc}/{cond}", leave=False):
                        qid = q["id"]
                        if qid in done_ids_set:
                             continue # Skip already generated

                        question = str(q["question"])
                        reference = str(q.get("reference", ""))
                        decision = str(q.get("decision", "")) # Ground truth answer/decision

                        # Determine base retrieval method and apply robustness if applicable
                        robustness_mode = None
                        base_method = cond
                        if cond in config.get("ROBUSTNESS_MODES", []):
                            robustness_mode = cond
                            base_method = "medcpt_rerank" # Default retrieval method for robustness tests

                        if base_method == "no_retrieval":
                            retrieved_ids, retrieved_scores = [], []
                            retrieved_texts = []
                        else:
                            retrieved_ids, retrieved_scores = rs.retrieve(question, base_method, config["TOP_K"])
                            retrieved_texts = [rs.chunk_texts[i] for i in retrieved_ids]

                        if robustness_mode:
                            retrieved_ids, retrieved_scores = rs.apply_robustness(retrieved_ids, retrieved_scores, robustness_mode)
                            retrieved_texts = [rs.chunk_texts[i] for i in retrieved_ids] # Update texts after robustness

                        # Prepare prompt (simple concatenation example)
                        context_str = "\n".join(retrieved_texts) if retrieved_texts else ""
                        if context_str:
                            prompt = f"Context:\n{context_str}\n\nQuestion: {question}\nAnswer:"
                        else:
                            prompt = f"Question: {question}\nAnswer:"

                        # Tokenize and truncate if necessary
                        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config["MODEL_REGISTRY"][model_name]["max_input"]).to(llm.device)

                        # Generate
                        with torch.no_grad():
                            outputs = llm.generate(
                                **inputs,
                                max_new_tokens=150, # Configurable
                                do_sample=True,
                                temperature=0.7, # Configurable
                                pad_token_id=tokenizer.pad_token_id
                            )

                        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

                        # --- Faithfulness / Trust Classification Logic ---
                        # This is a simplified version based on the provided code snippet
                        trust_class = "Abstained" # Default
                        faithfulness_all = None # Placeholder
                        gt_align = 0.0 # Placeholder
                        any_contradicted = False # Placeholder
                        severe_hall_bool = False # Placeholder
                        gt_hall_bool = False # Placeholder
                        claim_details = [] # Placeholder

                        if base_method != "no_retrieval" and retrieved_texts:
                            claims = split_claims(generated_text)
                            if claims:
                                supported_claims = 0
                                total_claims = len(claims)
                                for claim in claims:
                                    # Find best supporting chunk
                                    pairs = [(p, claim) for p in retrieved_texts if p.strip()]
                                    if pairs:
                                        pr_scores = nli.predict_pairs(pairs, batch_size=8) # Use global nli instance
                                        ent_scores = pr_scores[:, nli.ent_idx]
                                        contra_scores = pr_scores[:, nli.contra_idx]
                                        
                                        best_chunk_idx = int(np.argmax(ent_scores))
                                        best_ent_score = ent_scores[best_chunk_idx]
                                        best_contra_score = contra_scores[best_chunk_idx]

                                        is_supp = bool(best_ent_score >= config.get("FAITHFULNESS_MEDIUM", 0.3))
                                        is_contra = bool(best_contra_score >= config.get("CONTRADICTION_THRESHOLD", 0.5))

                                        claim_details.append({
                                            "claim": claim,
                                            "supported": is_supp,
                                            "contradicted": is_contra,
                                            "best_chunk": best_chunk_idx,
                                            "best_ent_score": float(best_ent_score),
                                            "best_contra_score": float(best_contra_score)
                                        })

                                        if is_supp:
                                            supported_claims += 1
                                        if is_contra:
                                            any_contradicted = True # At least one claim contradicted

                                # Determine overall trust class based on support/contradiction
                                supp_ratio = supported_claims / total_claims if total_claims > 0 else 0
                                if any_contradicted:
                                    trust_class = "Fabricated" # High priority if contradiction found
                                    severe_hall_bool = True
                                elif supp_ratio >= 0.9: # Most claims supported
                                    trust_class = "Fully Grounded"
                                elif supp_ratio >= 0.5: # Some support
                                    trust_class = "Partially Grounded"
                                elif supp_ratio == 0: # No support
                                    trust_class = "Unsupported"
                                else: # Partial support but not majority
                                    trust_class = "Partially Grounded" # Or another category

                                # Placeholder for ground truth alignment check (requires more complex logic)
                                # gt_align = compare_generated_to_reference(generated_text, reference)

                        # --- Save Generation Record ---
                        record = {
                            "qid": qid,
                            "question": question,
                            "reference": reference,
                            "decision": decision,
                            "generated": generated_text,
                            "condition": cond,
                            "model": model_name,
                            "dataset": ds_name,
                            "corpus_condition": cc,
                            "chunk_ids": retrieved_ids,
                            "chunk_scores": retrieved_scores,
                            "chunk_texts": retrieved_texts,
                            "trust_class": trust_class,
                            "faithfulness_all": faithfulness_all,
                            "gt_align": gt_align,
                            "any_contradicted": any_contradicted,
                            "severe_hall_bool": severe_hall_bool,
                            "gt_hall_bool": gt_hall_bool,
                            "claim_details": claim_details
                        }

                        # Append to file immediately to persist progress
                        with open(out_path, 'a') as f:
                            f.write(json.dumps(record) + "\n")

                        # Clear GPU cache periodically if generating many items
                        # if len(test_questions) > 1000 and (idx + 1) % 100 == 0:
                        #     clear_gpu()

    print("[OK] Generation loop completed.")


# --- Evaluation Helpers ---

def evaluate_generations(generations_path_pattern: str, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Evaluates a set of generated outputs against references.
    Calculates metrics like ROUGE-L, Exact Match, etc.
    """
    # This function would typically iterate through generated files,
    # load references, calculate metrics, and return a dataframe.
    # The exact implementation depends heavily on the format of
    # generated files and the metrics required.
    # It's complex enough that a dedicated evaluation module might be better,
    # but for now, we'll outline the core steps.

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    eval_results = []

    # Example: Iterate through generated files based on config paths
    for ds_name in config.get("DATASETS"):
        for cc in config["CORPUS_CONDITIONS"].get(ds_name, []):
            for model_name in config.get("MODELS", []): # Assuming MODELS is in config or passed
                for cond in config.get("RETRIEVERS") + (config.get("ROBUSTNESS_MODES", []) if config.get("INCLUDE_ROBUSTNESS") else []):
                    gen_file_path = config["GENERATION_DIR"] / ds_name / cc / model_name / f"{cond}.jsonl"
                    if not gen_file_path.exists():
                        print(f"[EVAL] File does not exist: {gen_file_path}")
                        continue

                    print(f"[EVAL] Processing {gen_file_path}...")
                    records = load_jsonl(gen_file_path)
                    for r in tqdm(records, desc=f"Evaluating {ds_name}/{cc}/{model_name}/{cond}", leave=False):
                        generated = str(r.get("generated", ""))
                        reference = str(r.get("reference", ""))
                        condition = r.get("condition", "")
                        model = r.get("model", "")
                        dataset = r.get("dataset", "")
                        corpus_cond = r.get("corpus_condition", "")

                        # ROUGE-L
                        rouge_l_fmeasure = 0.0
                        if generated and reference:
                            try:
                                rouge_l_fmeasure = float(scorer.score(reference, generated)["rougeL"].fmeasure)
                            except Exception as e:
                                print(f"[WARN] ROUGE calculation failed for qid {r.get('qid')}: {e}")
                                rouge_l_fmeasure = 0.0

                        # Exact Match (example for MCQ, adapt as needed)
                        exact_match = 0
                        if dataset == "medmcqa" and reference and generated:
                             # Simple check - might need more robust parsing
                             gen_upper = generated.upper()
                             ref_upper = reference.upper()
                             if ref_upper in gen_upper or gen_upper in ref_upper:
                                 exact_match = 1

                        eval_results.append({
                            "qid": r.get("qid"),
                            "dataset": dataset,
                            "corpus_condition": corpus_cond,
                            "model": model,
                            "condition": condition,
                            "rouge_l": rouge_l_fmeasure,
                            "exact_match": exact_match,
                            # Add other metrics calculated elsewhere (e.g., trust_class, hallucination flags)
                            "trust_class": r.get("trust_class"),
                            "severe_hall_bool": r.get("severe_hall_bool"),
                            "gt_align": r.get("gt_align"),
                            "any_contradicted": r.get("any_contradicted"),
                            "gt_hall_bool": r.get("gt_hall_bool"),
                        })

    df_eval = pd.DataFrame(eval_results)
    eval_output_path = config["STATS_DIR"] / "eval_metrics.csv"
    df_eval.to_csv(eval_output_path, index=False)
    print(f"[OK] Evaluation metrics saved to {eval_output_path}")
    return df_eval
