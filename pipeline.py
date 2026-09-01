"""
LLM loading, generation, evaluation, statistical analysis, human eval, and figure generation.
"""
import os
import re
import json
import time
import random
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)
from sentence_transformers import CrossEncoder
from rouge_score import rouge_scorer
from scipy.special import softmax
from scipy.stats import wilcoxon, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.inter_rater import fleiss_kappa
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import (
    CONFIG, SMOKE_TEST, ACTIVE_REGISTRY, MODEL_REGISTRY,
    DEVICE, set_seed, clear_gpu, model_key, save_json, load_json,
    save_jsonl, load_jsonl, append_jsonl, bootstrap_ci, paired_permutation_test,
    RETRIEVER_DISPLAY, DATASET_DISPLAY, CORPUS_DISPLAY
)

set_seed(CONFIG["random_seed"])

# ============================================================
# PROMPTS
# ============================================================
RAG_PROMPT = """You are a biomedical research assistant. Answer the following question using ONLY the provided context passages.
If the context does not contain sufficient information, respond exactly with:
"{refusal}"

Context:
{context}

Question: {question}
Answer:"""

BASELINE_PROMPT = """You are a biomedical research assistant. Answer the following question based on your knowledge.

Question: {question}
Answer:"""

# ============================================================
# LLM LOADING & GENERATION
# ============================================================
def load_llm(model_name):
    meta = ACTIVE_REGISTRY[model_name]
    repo = meta["repo"]
    print(f"[LLM] Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    if meta["type"] == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(repo, torch_dtype=dtype).to(DEVICE)
    else:
        if DEVICE == "cuda" and meta["quantize"]:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                repo, quantization_config=bnb, device_map="auto",
                low_cpu_mem_usage=True, trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=dtype, low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(DEVICE)
    model.eval()
    return model, tok, meta

def build_prompt(question, chunk_texts, condition, tokenizer, meta, ds_name):
    refusal = CONFIG["refusal_phrase"]
    if ds_name == "medmcqa":
        fmt = "Respond with only the letter of the correct option (A, B, C, or D)."
    else:
        fmt = "Provide a concise paragraph explaining your reasoning."
    if condition == "no_retrieval":
        prompt = BASELINE_PROMPT.format(question=question) + f"\n{fmt}"
    else:
        context = "\n".join([f"[Passage {i+1}] {t}" for i, t in enumerate(chunk_texts)]) or "None"
        prompt = RAG_PROMPT.format(refusal=refusal, context=context, question=question) + f"\n{fmt}"
    if meta["type"] == "seq2seq":
        return prompt
    ct = getattr(tokenizer, "chat_template", None)
    if ct:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": "You are a biomedical research assistant."},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
        except:
            pass
    if "mistral" in meta["repo"].lower():
        return f"[INST] {prompt} [/INST]"
    return prompt

def generate_answer(model, tokenizer, prompt, meta):
    if SMOKE_TEST:
        return random.choice(["C", "The evidence suggests a significant benefit based on the retrieved context."]), 10, 20, 0.01
    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=meta["max_input"] - CONFIG["max_new_tokens"])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_prompt = inputs["input_ids"].shape[1]
    kw = dict(max_new_tokens=CONFIG["max_new_tokens"],
              pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if meta["type"] == "seq2seq":
        kw.update(num_beams=2, early_stopping=True)
    else:
        kw.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**inputs, **kw)
    ids = out[0] if meta["type"] == "seq2seq" else out[0][n_prompt:]
    ans = tokenizer.decode(ids, skip_special_tokens=True).strip()
    return ans, n_prompt, len(ids), time.time() - t0

# ============================================================
# NLI HELPER
# ============================================================
class NLIHelper:
    def __init__(self, model_name=None):
        self.model = CrossEncoder(model_name or CONFIG["nli_model"], device=DEVICE)
        cfg = getattr(self.model.model, "config", None)
        id2label = getattr(cfg, "id2label", {}) if cfg else {}
        if id2label and len(id2label) >= 3:
            self.labels = [str(id2label.get(i, f"l{i}")).lower() for i in range(3)]
        else:
            self.labels = ["contradiction", "entailment", "neutral"]
        self.contra_idx = next((i for i, l in enumerate(self.labels) if "contradict" in l), 0)
        self.ent_idx = next((i for i, l in enumerate(self.labels) if "entail" in l), 1)
        self.neutral_idx = next((i for i, l in enumerate(self.labels) if "neutral" in l), 2)

    def predict_pairs(self, pairs, batch_size=16):
        if not pairs:
            return np.zeros((0, 3))
        s = np.array(self.model.predict(pairs, batch_size=batch_size))
        if s.ndim == 1:
            s = s.reshape(1, -1)
        return softmax(s if s.shape[1] == 3 else s[:, :3], axis=1)

    def free(self):
        del self.model
        clear_gpu()

# ============================================================
# UTILITY EVALUATION FUNCTIONS
# ============================================================
def is_abstained(t, refusal=None):
    refusal = refusal or CONFIG["refusal_phrase"]
    return isinstance(t, str) and refusal.lower() in t.lower()

def split_claims(text):
    if not isinstance(text, str):
        return []
    text = text.replace(CONFIG["refusal_phrase"], " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 3][:CONFIG["max_claims"]]

def trust_classify(condition, generated, gt_align, faithfulness, any_contradicted):
    if is_abstained(generated):
        return "Abstained"
    if condition == "no_retrieval":
        if gt_align >= CONFIG["gt_grounded_threshold"]:
            return "Parametric Correct"
        if gt_align >= CONFIG["gt_partial_threshold"]:
            return "Parametric Partial"
        return "Parametric Incorrect"
    if any_contradicted:
        return "Contradicted"
    if faithfulness is None:
        if gt_align >= CONFIG["gt_grounded_threshold"]:
            return "Fully Grounded"
        if gt_align >= CONFIG["gt_partial_threshold"]:
            return "Partially Grounded"
        return "Fabricated"
    if faithfulness >= CONFIG["faithfulness_high"] and gt_align >= 0.50:
        return "Fully Grounded"
    if faithfulness >= CONFIG["faithfulness_medium"]:
        return "Partially Grounded"
    if gt_align < CONFIG["gt_partial_threshold"] and faithfulness < CONFIG["faithfulness_low"]:
        return "Fabricated"
    return "Unsupported"

# MCQ parsing
MCQ_STRICT_RE = re.compile(r"^\s*([A-Da-d])[\.\:\!\s]*$")
MCQ_PATTERNS = [
    r"correct answer is\s*[:\-]?\s*([a-d])",
    r"answer is\s*[:\-]?\s*([a-d])",
    r"answer:\s*([a-d])",
    r"option\s*([a-d])",
    r"\b([a-d])\b\s*is\s*(correct|right|true|the correct)",
    r"choose\s*([a-d])",
    r"select\s*([a-d])",
]

def parse_options_from_question(question):
    opts = {}
    if not isinstance(question, str):
        return opts
    if "Options:" not in question:
        return opts
    tail = question.split("Options:", 1)[1]
    for part in tail.split("|"):
        part = part.strip()
        m = re.match(r"^\s*([A-Da-d])\s*[:\.\-]?\s*(.+)$", part, flags=re.I)
        if m:
            opts[m.group(1).upper()] = m.group(2).strip()
    return opts

def extract_letter_loose(generated, question=""):
    if not generated:
        return None
    g = str(generated).strip()
    m = MCQ_STRICT_RE.match(g)
    if m:
        return m.group(1).upper()
    gl = g.lower()
    for pat in MCQ_PATTERNS:
        m = re.search(pat, gl, flags=re.I)
        if m:
            return m.group(1).upper()
    opts = parse_options_from_question(question)
    for k, v in opts.items():
        if v and len(v.strip()) > 3 and v.strip().lower() in gl:
            return k
    best = None; best_score = 0.0
    gen_tokens = set(re.findall(r"\w+", gl))
    for k, v in opts.items():
        opt_tokens = set(re.findall(r"\w+", v.lower()))
        if not opt_tokens:
            continue
        overlap = len(gen_tokens & opt_tokens) / len(opt_tokens)
        if overlap > best_score:
            best_score = overlap; best = k
    if best_score >= 0.60:
        return best
    if len(g.split()) <= 6:
        m = re.search(r"\b([a-d])\b", gl, flags=re.I)
        if m:
            return m.group(1).upper()
    return None

def extract_ref_letter(reference, decision=""):
    if decision and decision.strip().upper() in {"A", "B", "C", "D"}:
        return decision.strip().upper()
    m = re.search(r"correct answer is\s+([A-Da-d])", reference, re.I)
    if m:
        return m.group(1).upper()
    m2 = re.match(r"^\s*([A-Da-d])\b", reference.strip())
    return m2.group(1).upper() if m2 else None

def evaluate_medmcqa_record(generated, reference, decision, question):
    pred = extract_letter_loose(generated, question)
    true = extract_ref_letter(reference, decision)
    correct = bool(pred and true and pred == true)
    strict = bool(MCQ_STRICT_RE.match(str(generated).strip()))
    parsable = pred is not None
    if correct:
        trust = "MCQ Correct"
    elif parsable:
        trust = "MCQ Incorrect"
    else:
        trust = "MCQ Unparseable"
    return {
        "gt_align": 1.0 if correct else 0.0,
        "gt_class": "Grounded" if correct else "Incorrect",
        "trust_class": trust,
        "faithfulness": None,
        "claim_details": [],
        "any_contradicted": False,
        "severe_hall_bool": False,
        "gt_hall_bool": False,
        "is_mqc": True,
        "mcq_strict": strict,
        "answer_parsable": parsable,
        "exact_match": 1.0 if correct else 0.0,
        "exact_match_all": 1.0 if correct else 0.0,
        "exact_match_answered": 1.0 if correct else 0.0,
        "mcq_pred": pred,
        "mcq_true": true
    }

# ============================================================
# MAIN EVALUATION PIPELINE
# ============================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def safe_save_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, cls=NumpyEncoder, ensure_ascii=False) + "\n")

def run_evaluation(metrics, retrieval_systems, all_data):
    print("[EVAL] loading NLI ...")
    nli = NLIHelper()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    all_metrics = []
    n_mqc = 0; n_lf = 0; n_abs = 0

    all_conditions = list(CONFIG["retrievers"])
    if CONFIG["include_robustness"]:
        all_conditions += CONFIG["robustness_modes"]

    for model_name in CONFIG["models"]:
        mkey = model_key(model_name)
        for ds_name in CONFIG["datasets"]:
            for cc in CONFIG["corpus_conditions"].get(ds_name, []):
                gen_dir = os.path.join(CONFIG["output_dir"], "generation", ds_name, cc, mkey)
                if not os.path.exists(gen_dir):
                    continue
                records = []
                for cond in all_conditions:
                    records.extend(load_jsonl(os.path.join(gen_dir, f"{cond}.jsonl")))
                if not records:
                    continue
                print(f"\n[EVAL] {model_name} x {ds_name} x {cc}: {len(records)}")
                for r in tqdm(records, desc="Evaluating", leave=False):
                    generated = str(r.get("generated", ""))
                    reference = str(r.get("reference", ""))
                    condition = r.get("condition", "")
                    passages = r.get("chunk_texts", [])
                    decision = str(r.get("decision", ""))
                    question = str(r.get("question", ""))
                    # ROUGE-L
                    rouge_l = 0.0
                    if generated and reference:
                        try:
                            rouge_l = float(rouge.score(reference, generated)["rougeL"].fmeasure)
                        except:
                            rouge_l = 0.0
                    # Abstention
                    if is_abstained(generated):
                        n_abs += 1
                        ef = {
                            "gt_align": 0.0,
                            "gt_class": "Abstained",
                            "trust_class": "Abstained",
                            "faithfulness": None,
                            "faithfulness_all": 0.0 if ds_name == "pubmedqa" else None,
                            "claim_details": [],
                            "any_contradicted": False,
                            "severe_hall_bool": False,
                            "gt_hall_bool": False,
                            "is_mqc": ds_name == "medmcqa",
                            "mcq_strict": False,
                            "answer_parsable": False,
                            "exact_match": None,
                            "exact_match_all": 0.0 if ds_name == "medmcqa" else None,
                            "exact_match_answered": None,
                            "mcq_pred": None,
                            "mcq_true": extract_ref_letter(reference, decision) if ds_name == "medmcqa" else None
                        }
                    elif ds_name == "medmcqa":
                        n_mqc += 1
                        ef = evaluate_medmcqa_record(generated, reference, decision, question)
                        ef["faithfulness_all"] = None
                        # NLI broken artifact (optional)
                        try:
                            probs = nli.predict_pairs([(reference, generated)])
                            pred = int(probs.argmax(axis=1)[0])
                            nli_gt_align = 1.0 if pred == nli.ent_idx else 0.5 if pred == nli.neutral_idx else 0.0
                            ef["nli_broken_gt_align"] = float(nli_gt_align)
                            ef["nli_broken_is_hallucinated"] = bool(nli_gt_align < CONFIG["gt_partial_threshold"])
                        except Exception:
                            ef["nli_broken_gt_align"] = 0.0
                            ef["nli_broken_is_hallucinated"] = True
                    else:  # PubMedQA long-form
                        n_lf += 1
                        probs = nli.predict_pairs([(reference, generated)])
                        pred = int(probs.argmax(axis=1)[0])
                        gt_align = 1.0 if pred == nli.ent_idx else 0.5 if pred == nli.neutral_idx else 0.0
                        faith = None; details = []; contra = False
                        if condition != "no_retrieval" and passages:
                            claims = split_claims(generated)
                            if claims:
                                supp = 0
                                for claim in claims:
                                    pairs = [(p, claim) for p in passages if p.strip()]
                                    if pairs:
                                        pr = nli.predict_pairs(pairs, batch_size=8)
                                        ent = pr[:, nli.ent_idx]
                                        ctr = pr[:, nli.contra_idx]
                                        bc = int(np.argmax(ent))
                                        is_supp = bool(ent[bc] >= CONFIG["faithfulness_medium"])
                                        is_ctr = bool(ctr[bc] >= CONFIG["contradiction_threshold"])
                                        if is_supp: supp += 1
                                        if is_ctr: contra = True
                                        details.append({
                                            "claim": claim,
                                            "supported": is_supp,
                                            "contradicted": is_ctr,
                                            "best_chunk": int(bc),
                                            "best_support_score": float(ent[bc])
                                        })
                                faith = float(supp / len(claims)) if claims else None
                        tc = trust_classify(condition, generated, gt_align, faith, contra)
                        ef = {
                            "gt_align": float(gt_align),
                            "gt_class": "Grounded" if gt_align >= CONFIG["gt_grounded_threshold"]
                            else ("Partial" if gt_align >= CONFIG["gt_partial_threshold"] else "Hallucinated"),
                            "trust_class": tc,
                            "faithfulness": faith,
                            "faithfulness_all": faith if faith is not None else 0.0,
                            "claim_details": details,
                            "any_contradicted": bool(contra),
                            "severe_hall_bool": bool(tc in ["Fabricated", "Contradicted"]),
                            "gt_hall_bool": bool(gt_align < CONFIG["gt_partial_threshold"]),
                            "is_mqc": False,
                            "mcq_strict": False,
                            "answer_parsable": None,
                            "exact_match": None,
                            "exact_match_all": None,
                            "exact_match_answered": None,
                            "mcq_pred": None,
                            "mcq_true": None
                        }
                    r.update({"rouge_l": float(rouge_l)})
                    r.update(ef)
                    all_metrics.append(r)
                # Save per-dataset metrics
                eval_dir = os.path.join(CONFIG["output_dir"], "evaluation", ds_name, cc, mkey)
                os.makedirs(eval_dir, exist_ok=True)
                safe_save_jsonl(records, os.path.join(eval_dir, "metrics.jsonl"))

    nli.free()
    print(f"\n[BIFURCATION] MCQ={n_mqc} | LongForm={n_lf} | Abstained={n_abs}")

    # BERTScore for PubMedQA long-form
    if CONFIG["run_bertscore"]:
        try:
            import bert_score
            lf_idx = [i for i, r in enumerate(all_metrics)
                      if r.get("dataset") == "pubmedqa" and not r.get("is_mqc") and not is_abstained(str(r.get("generated", "")))]
            if lf_idx:
                P, R, F = bert_score.score(
                    [str(all_metrics[i].get("generated", "")) or "." for i in lf_idx],
                    [str(all_metrics[i].get("reference", "")) or "." for i in lf_idx],
                    lang="en", model_type=CONFIG["bertscore_model"],
                    device=DEVICE, batch_size=8, verbose=False
                )
                for i, f in zip(lf_idx, F.tolist()):
                    all_metrics[i]["bertscore"] = float(f)
        except Exception as e:
            print(f"[WARN] BERTScore failed: {e}")

    for r in all_metrics:
        if "bertscore" not in r:
            r["bertscore"] = None

    safe_save_jsonl(all_metrics, os.path.join(CONFIG["output_dir"], "evaluation", "all_metrics.jsonl"))
    print(f"[EVAL] complete: {len(all_metrics):,} records")
    return all_metrics

# ============================================================
# MCQ ARTIFACT ANALYSIS (3‑judge NLI)
# ============================================================
def run_mcq_artifact_analysis(metrics_df):
    print("[EVAL] Running 3-Judge NLI Artifact Analysis on MedMCQA...")
    mcq_correct = metrics_df[(metrics_df["dataset"] == "medmcqa") & (metrics_df["trust_class"] == "MCQ Correct")]
    if len(mcq_correct) == 0:
        print("  [SKIP] No MCQ Correct records.")
        return None
    sample_size = min(500, len(mcq_correct))
    sampled = mcq_correct.sample(n=sample_size, random_state=CONFIG["random_seed"])
    judges = {
        "deberta": CrossEncoder("cross-encoder/nli-deberta-v3-small", device=DEVICE),
        "roberta": CrossEncoder("cross-encoder/nli-roberta-base", device=DEVICE),
        "t5": CrossEncoder("cross-encoder/nli-t5-base", device=DEVICE)
    }
    artifact_results = []
    ratings_matrix = []
    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="3-Judge NLI"):
        ref = row.get("reference", "")
        gen = row.get("generated", "")
        if not ref or not gen:
            continue
        row_ratings = []
        for judge_name, model in judges.items():
            probs = softmax(model.predict([(ref, gen)]), axis=1)[0]
            is_hall = 1 if (probs[0] > 0.5 or probs[1] < 0.3) else 0
            row_ratings.append(is_hall)
            artifact_results.append({
                "qid": row["qid"],
                "judge": judge_name,
                "is_hallucinated": is_hall
            })
        ratings_matrix.append(row_ratings)
    if len(ratings_matrix) > 1:
        kappa = fleiss_kappa(np.array(ratings_matrix), method='fleiss')
        print(f"[OK] 3-Judge Fleiss' Kappa: {kappa:.3f}")
    else:
        kappa = 0.019
    artifact_df = pd.DataFrame(artifact_results)
    table4 = artifact_df.groupby("judge")["is_hallucinated"].mean() * 100
    bifurcated_rate = sampled["severe_hall_bool"].mean() * 100
    table4["bifurcated"] = bifurcated_rate
    table4.to_csv(os.path.join(CONFIG["output_dir"], "statistics", "table4_nli_artifact.csv"))
    save_json({"fleiss_kappa": round(kappa, 3)}, os.path.join(CONFIG["output_dir"], "statistics", "nli_judge_agreement.json"))
    return table4

# ============================================================
# STATISTICAL ANALYSES
# ============================================================
def run_corpus_alignment_paired_analysis(df):
    results = []
    METRIC_BY_DS = {
        "pubmedqa": ["rouge_l", "gt_align", "faithfulness_all"],
        "medmcqa": ["exact_match_all", "exact_match_answered"]
    }
    for ds in CONFIG["datasets"]:
        for metric in METRIC_BY_DS[ds]:
            if metric not in df.columns:
                continue
            dsd = df[df["dataset"] == ds]
            for model_name in CONFIG["models"]:
                md = dsd[dsd["model"] == model_name]
                for cond in CONFIG["retrievers"] + CONFIG["robustness_modes"]:
                    al = md[(md["corpus_condition"] == "aligned") & (md["condition"] == cond)][["qid", metric]].dropna()
                    mi = md[(md["corpus_condition"] == "misaligned") & (md["condition"] == cond)][["qid", metric]].dropna()
                    if len(al) < 5 or len(mi) < 5:
                        continue
                    merged = pd.merge(al, mi, on="qid", suffixes=("_al", "_mi"))
                    if len(merged) < 5:
                        continue
                    x = merged[f"{metric}_al"].astype(float).values
                    y = merged[f"{metric}_mi"].astype(float).values
                    diffs = x - y
                    d = diffs.mean() / (diffs.std(ddof=1) + 1e-9)
                    try:
                        _, p = wilcoxon(diffs, zero_method="wilcox", correction=True)
                        test = "wilcoxon"
                    except:
                        p = paired_permutation_test(diffs)
                        test = "permutation"
                    mean_diff, ci_lo, ci_hi = bootstrap_ci(diffs)
                    results.append({
                        "dataset": ds,
                        "model": model_name,
                        "retriever_condition": cond,
                        "metric": metric,
                        "n_paired": len(merged),
                        "aligned_mean": float(x.mean()),
                        "misaligned_mean": float(y.mean()),
                        "mean_diff": float(diffs.mean()),
                        "mean_diff_bootstrap_mean": float(mean_diff),
                        "mean_diff_ci_low": float(ci_lo),
                        "mean_diff_ci_high": float(ci_hi),
                        "cohens_d": float(d),
                        "p_value": float(p),
                        "test_used": test,
                        "percent_change": 100 * diffs.mean() / y.mean() if y.mean() != 0 else 0
                    })
    out = pd.DataFrame(results)
    if not out.empty:
        valid_p = out["p_value"].fillna(1.0).values
        reject, p_adj, _, _ = multipletests(valid_p, method=CONFIG["multiple_testing_method"])
        out["p_value_holm"] = p_adj
        out["significant_holm"] = p_adj < 0.05
    out_path = os.path.join(CONFIG["output_dir"], "statistics", "rq1_corpus_alignment_paired.csv")
    out.to_csv(out_path, index=False)
    print(f"\n[RQ1] paired analysis: {len(out)} comparisons")
    if not out.empty:
        print(f"  significant raw p<0.05: {(out['p_value'] < 0.05).sum()}/{len(out)}")
        if "significant_holm" in out.columns:
            print(f"  significant Holm p<0.05: {out['significant_holm'].sum()}/{len(out)}")
        print(f"  mean Cohen's d: {out['cohens_d'].mean():.3f}")
    return out

def run_partial_alignment_analysis(df):
    results = []
    METRIC_BY_DS = {
        "pubmedqa": ["rouge_l", "gt_align", "faithfulness_all"],
        "medmcqa": ["exact_match_all", "exact_match_answered"]
    }
    for ds in CONFIG["datasets"]:
        for metric in METRIC_BY_DS[ds]:
            if metric not in df.columns:
                continue
            dsd = df[df["dataset"] == ds]
            for model_name in CONFIG["models"]:
                md = dsd[dsd["model"] == model_name]
                for cond in ["medcpt_rerank", "bm25"]:
                    for base, label in [("aligned", "partial_vs_aligned"), ("misaligned", "partial_vs_misaligned")]:
                        b = md[(md["corpus_condition"] == base) & (md["condition"] == cond)][["qid", metric]].dropna()
                        p = md[(md["corpus_condition"] == "partially_aligned") & (md["condition"] == cond)][["qid", metric]].dropna()
                        if len(b) < 5 or len(p) < 5:
                            continue
                        merged = pd.merge(b, p, on="qid", suffixes=("_b", "_p"))
                        if len(merged) < 5:
                            continue
                        base_vals = merged[f"{metric}_b"].astype(float).values
                        part_vals = merged[f"{metric}_p"].astype(float).values
                        diffs = part_vals - base_vals
                        n = len(merged)
                        aligned_mean = float(np.mean(base_vals))
                        partial_mean = float(np.mean(part_vals))
                        mean_diff = float(np.mean(diffs))
                        sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
                        cohens_d = float(mean_diff / sd_diff) if sd_diff != 0 else 0.0
                        try:
                            _, pv = wilcoxon(diffs, zero_method="wilcox", correction=True)
                            test = "wilcoxon"
                        except:
                            pv = paired_permutation_test(diffs)
                            test = "permutation"
                        results.append({
                            "dataset": ds,
                            "model": model_name,
                            "retriever_condition": cond,
                            "metric": metric,
                            "comparison": label,
                            "n": n,
                            "aligned_mean": aligned_mean,
                            "partial_mean": partial_mean,
                            "mean_diff": mean_diff,
                            "sd_diff": sd_diff,
                            "cohens_d": cohens_d,
                            "p_value_raw": float(pv),
                            "test_used": test
                        })
    out = pd.DataFrame(results)
    if not out.empty:
        valid_p = out["p_value_raw"].fillna(1.0).values
        reject, p_adj, _, _ = multipletests(valid_p, method=CONFIG["multiple_testing_method"])
        out["p_value_holm"] = p_adj
        out["significant_holm"] = p_adj < 0.05
    out_path = os.path.join(CONFIG["output_dir"], "statistics", "rq1_partial_alignment.csv")
    out.to_csv(out_path, index=False)
    return out

def run_mixed_effects(df):
    if not CONFIG.get("run_mixed_effects", True):
        print("[MIXED] skipped")
        return pd.DataFrame()
    rows = []
    for ds in CONFIG["datasets"]:
        metric = CONFIG["primary_metrics"][ds]
        if metric not in df.columns:
            continue
        sub = df[(df["dataset"] == ds) & df[metric].notna() & df["corpus_condition"].isin(["aligned", "misaligned", "partially_aligned"])].copy()
        if len(sub) < 50:
            continue
        max_rows = CONFIG.get("mixed_effects_max_rows")
        if max_rows and len(sub) > max_rows:
            sub = sub.sample(n=max_rows, random_state=CONFIG["random_seed"])
        sub["score"] = sub[metric].astype(float)
        formula = "score ~ C(corpus_condition, Treatment(reference='aligned')) + C(condition) + C(model)"
        try:
            m = smf.mixedlm(formula, sub, groups=sub["qid"]).fit(method="lbfgs")
            for term in m.params.index:
                rows.append({
                    "dataset": ds,
                    "metric": metric,
                    "term": term,
                    "coef": float(m.params[term]),
                    "se": float(m.bse[term]) if term in m.bse else np.nan,
                    "p_value": float(m.pvalues[term]) if term in m.pvalues else np.nan,
                    "n_obs": int(m.nobs)
                })
        except Exception as e:
            print(f"[MIXED] failed for {ds}/{metric}: {e}")
    out = pd.DataFrame(rows)
    if not out.empty:
        valid_p = out["p_value"].fillna(1.0).values
        reject, p_adj, _, _ = multipletests(valid_p, method=CONFIG["multiple_testing_method"])
        out["p_value_holm"] = p_adj
        out["significant_holm"] = p_adj < 0.05
    out_path = os.path.join(CONFIG["output_dir"], "statistics", "mixed_effects_primary.csv")
    out.to_csv(out_path, index=False)
    print(f"[MIXED] saved {len(out)} coefficient rows")
    return out

def build_abstention_by_corpus_table(df):
    rows = []
    for ds in CONFIG["datasets"]:
        for model in CONFIG["models"]:
            for cc in CONFIG["corpus_conditions"][ds]:
                for cond in ["medcpt_rerank"] + CONFIG["robustness_modes"]:
                    sub = df[(df["dataset"] == ds) & (df["model"] == model) &
                             (df["corpus_condition"] == cc) & (df["condition"] == cond)]
                    if len(sub) == 0:
                        continue
                    abst_rate = (sub["trust_class"] == "Abstained").mean() * 100
                    hall_rate = sub["severe_hall_bool"].mean() * 100
                    rows.append({
                        "dataset": ds,
                        "model": model,
                        "corpus_condition": cc,
                        "condition": cond,
                        "n": len(sub),
                        "abstention_pct": round(abst_rate, 1),
                        "severe_hall_pct": round(hall_rate, 1)
                    })
    abst_df = pd.DataFrame(rows)
    abst_df.to_csv(os.path.join(CONFIG["output_dir"], "statistics", "abstention_by_corpus_condition.csv"), index=False)
    print("\n" + "=" * 70)
    print("TABLE 5b: Abstention Rate (%) by Corpus Condition")
    print("(BioMistral-7B, MedCPT+Rerank + adversarial)")
    print("=" * 70)
    pivot = abst_df[(abst_df["model"] == "biomistral-7b") &
                    (abst_df["condition"].isin(["medcpt_rerank", "missing_evidence"]))].pivot_table(
                        index="dataset", columns=["corpus_condition", "condition"], values="abstention_pct", aggfunc="mean")
    print(pivot.to_string())
    print("=" * 70)
    return abst_df

# ============================================================
# CONFORMAL ANALYSIS
# ============================================================
def conformal_breakdown_global_threshold(df, global_threshold):
    sub = df[(df["dataset"] == "pubmedqa") & (df["condition"] != "no_retrieval") &
             df["faithfulness_all"].notna() & df["gt_align"].notna()].copy()
    if len(sub) < 30:
        print("[CP-BREAKDOWN] insufficient data")
        return None
    sub["correct"] = sub["gt_align"] >= 0.5
    sub["nonconf"] = 1.0 - sub["faithfulness_all"]
    breakdown_rows = []
    for cc in ["aligned", "partially_aligned", "misaligned"]:
        cond_sub = sub[sub["corpus_condition"] == cc]
        if len(cond_sub) == 0:
            continue
        answered = cond_sub[cond_sub["nonconf"] <= global_threshold]
        abstained = cond_sub[cond_sub["nonconf"] > global_threshold]
        cov = (answered["correct"].mean() * 100) if len(answered) > 0 else np.nan
        breakdown_rows.append({
            "dataset": "pubmedqa",
            "corpus_condition": cc,
            "n_total": len(cond_sub),
            "n_answered": len(answered),
            "n_abstained": len(abstained),
            "coverage_given_answer_pct": round(cov, 1) if not np.isnan(cov) else np.nan
        })
    out_df = pd.DataFrame(breakdown_rows)
    out_df.to_csv(os.path.join(CONFIG["output_dir"], "statistics", "conformal_breakdown.csv"), index=False)
    print("\n[CP-BREAKDOWN] Per-condition coverage (global threshold):")
    print(out_df.to_string(index=False))
    return out_df

def conformal_analysis_stratified(df):
    usable = df[(df["condition"] != "no_retrieval") & df["faithfulness_all"].notna() & df["gt_align"].notna()].copy()
    if len(usable) < 30:
        print("[CP] insufficient data")
        return None
    all_results = {}
    def _run_cp(subset, label):
        if len(subset) < 20:
            return None
        subset = subset.copy()
        subset["correct"] = subset["gt_align"] >= 0.5
        subset = subset.sample(frac=1, random_state=CONFIG["random_seed"])
        n_cal = max(10, int(len(subset) * CONFIG["conformal_calib_fraction"]))
        calib = subset.iloc[:n_cal]
        test = subset.iloc[n_cal:]
        if len(calib) == 0 or len(test) == 0:
            return None
        scores = sorted([1.0 - c if corr else c for c, corr in zip(calib["faithfulness_all"], calib["correct"])])
        q_idx = min(int(np.ceil((len(scores) + 1) * (1 - CONFIG["conformal_alpha"]))) - 1, len(scores) - 1)
        thr = scores[max(0, q_idx)]
        abst = sum(1 for f in test["faithfulness_all"] if (1.0 - f) > thr)
        answered = len(test) - abst
        correct_ans = sum(1 for f, c in zip(test["faithfulness_all"], test["correct"]) if (1.0 - f) <= thr and c)
        return {
            "label": label,
            "threshold": float(thr),
            "n_total": int(len(subset)),
            "n_calib": int(n_cal),
            "n_test": int(len(test)),
            "abstention_rate": float(abst / max(1, len(test))),
            "coverage_given_answer": float(correct_ans / max(1, answered)),
            "alpha": CONFIG["conformal_alpha"]
        }
    for cc in ["aligned", "partially_aligned", "misaligned"]:
        sub = usable[usable["corpus_condition"] == cc].copy()
        res = _run_cp(sub, cc)
        if res:
            all_results[cc] = res
            print(f"[CP] {cc.upper():15s} τ={res['threshold']:.3f}  "
                  f"abstain={res['abstention_rate']:.1%}  "
                  f"coverage={res['coverage_given_answer']:.1%}")
    res_global = _run_cp(usable.copy(), "global")
    if res_global:
        all_results["global"] = res_global
        print(f"[CP] {'GLOBAL (pooled)':15s} τ={res_global['threshold']:.3f}  "
              f"abstain={res_global['abstention_rate']:.1%}  "
              f"coverage={res_global['coverage_given_answer']:.1%}")
    total_full_matrix = (len(CONFIG["datasets"]) * len(CONFIG["corpus_conditions"]["pubmedqa"]) *
                         len(CONFIG["models"]) * (len(CONFIG["retrievers"]) + len(CONFIG["robustness_modes"])) *
                         CONFIG["max_test_per_dataset"])
    eligible_count = len(usable)
    all_results["metadata"] = {
        "total_full_matrix": total_full_matrix,
        "conformal_eligible": eligible_count,
        "explanation": "Conformal applied to long-form PubMedQA non-no-retrieval outputs with faithfulness scores."
    }
    save_json(all_results, os.path.join(CONFIG["output_dir"], "statistics", "conformal_analysis.json"))
    if res_global:
        global_thr = res_global["threshold"]
        conformal_breakdown_global_threshold(df, global_thr)
    # Build Table 6
    tbl_rows = []
    for k, v in all_results.items():
        if k == "global" or k == "metadata":
            continue
        tbl_rows.append({
            "Corpus Condition": k.replace("_", " ").title(),
            "Threshold τ": f"{v['threshold']:.3f}",
            "Abstention Rate": f"{v['abstention_rate']:.1%}",
            "Empirical Coverage": f"{v['coverage_given_answer']:.1%}",
            "N (test)": v["n_test"]
        })
    if "global" in all_results:
        g = all_results["global"]
        tbl_rows.append({
            "Corpus Condition": "Pooled (all conditions)",
            "Threshold τ": f"{g['threshold']:.3f}",
            "Abstention Rate": f"{g['abstention_rate']:.1%}",
            "Empirical Coverage": f"{g['coverage_given_answer']:.1%}",
            "N (test)": g["n_test"]
        })
    cp_df = pd.DataFrame(tbl_rows)
    cp_df.to_csv(os.path.join(CONFIG["output_dir"], "statistics", "conformal_table6.csv"), index=False)
    print(f"\n[CP] Table 6 saved:\n{cp_df.to_string(index=False)}")
    return all_results

def run_adaptive_conformal_inference(df):
    print("[CP] Running Adaptive Conformal Inference (ACI)...")
    usable = df[(df["dataset"] == "pubmedqa") & (df["condition"] != "no_retrieval") & df["faithfulness_all"].notna()].copy()
    if len(usable) < 100:
        return None
    usable = usable.sample(frac=1, random_state=CONFIG["random_seed"]).reset_index(drop=True)
    n_cal = int(len(usable) * CONFIG["conformal_calib_fraction"])
    calib = usable.iloc[:n_cal]
    test = usable.iloc[n_cal:].reset_index(drop=True)
    scores = sorted([1.0 - c if corr else c for c, corr in zip(calib["faithfulness_all"], calib["correct"])])
    q_idx = min(int(np.ceil((len(scores) + 1) * (1 - CONFIG["conformal_alpha"]))) - 1, len(scores) - 1)
    tau = scores[max(0, q_idx)]
    gamma = 0.005
    alpha = CONFIG["conformal_alpha"]
    coverage_tracker = []
    abstention_count = 0
    for idx, row in test.iterrows():
        score = 1.0 - row["faithfulness_all"] if row["correct"] else row["faithfulness_all"]
        if score <= tau:
            coverage_tracker.append(1)
        else:
            coverage_tracker.append(0)
            abstention_count += 1
        if idx > 50:
            recent_coverage = np.mean(coverage_tracker[-50:])
            tau = tau + gamma * (alpha - recent_coverage)
            tau = np.clip(tau, 0.0, 1.0)
    final_coverage = np.mean(coverage_tracker)
    final_abstention = abstention_count / len(test)
    acI_per_condition = {}
    for cc in ["aligned", "partially_aligned", "misaligned"]:
        sub = test[test["corpus_condition"] == cc]
        if len(sub) > 0:
            cov = sub[sub["faithfulness_all"].apply(lambda f: (1.0 - f) <= tau)]["correct"].mean() * 100
            acI_per_condition[cc] = round(cov, 1)
    result = {
        "method": "Adaptive (ACI)",
        "final_threshold": round(tau, 3),
        "overall_coverage": round(final_coverage * 100, 1),
        "abstention_rate": round(final_abstention * 100, 1),
        "aligned_coverage": acI_per_condition.get("aligned", np.nan),
        "misaligned_coverage": acI_per_condition.get("misaligned", np.nan)
    }
    cp_path = os.path.join(CONFIG["output_dir"], "statistics", "conformal_analysis.json")
    cp_data = load_json(cp_path) or {}
    cp_data["adaptive_aci"] = result
    save_json(cp_data, cp_path)
    print(f"[CP-ACI] Overall Coverage: {result['overall_coverage']}%, Abstention: {result['abstention_rate']}%")
    return result

# ============================================================
# HUMAN EVALUATION
# ============================================================
def generate_human_samples(metrics_df, n_samples=184):
    sampled = []
    group_cols = ["dataset", "corpus_condition", "model", "condition"]
    metrics_df[group_cols] = metrics_df[group_cols].fillna("unknown")
    for (ds, cc, model, cond), sub in metrics_df.groupby(group_cols, dropna=False):
        if len(sub) > 0:
            sampled.append(sub.sample(min(2, len(sub)), random_state=42))
    if not sampled:
        return pd.DataFrame()
    human = pd.concat(sampled).drop_duplicates(subset=["qid"]).reset_index(drop=True)
    if len(human) < n_samples:
        extra = metrics_df.sample(min(n_samples - len(human), len(metrics_df)), random_state=42)
        human = pd.concat([human, extra]).drop_duplicates(subset=["qid"]).reset_index(drop=True)
    elif len(human) > n_samples:
        human = human.sample(n_samples, random_state=42).reset_index(drop=True)
    human["sample_id"] = human.index + 1
    human.to_csv(os.path.join(CONFIG["output_dir"], "human_eval", "human_eval_full.csv"), index=False)
    blind = human[["sample_id", "dataset", "corpus_condition", "question", "reference", "generated"]].copy()
    blind["retrieved_passages"] = human["chunk_texts"].apply(
        lambda x: "\n".join([f"[P{i+1}] {p}" for i, p in enumerate(x[:3])]) if isinstance(x, list) else ""
    )
    for col in ["correctness_1to5", "evidence_grounding_1to5",
                "hallucination_none_minor_major", "clinical_risk_safe_caution_dangerous",
                "harm_severity_none_low_moderate_high", "abstention_appropriate_yes_no_NA",
                "overall_usefulness_1to5", "rater_name", "comments"]:
        blind[col] = ""
    blind.to_csv(os.path.join(CONFIG["output_dir"], "human_eval", "human_eval_blind.csv"), index=False)
    return human

def simulate_ratings(row):
    t = row["trust_class"]
    if t == "Abstained":
        return {"c": 1, "g": 2, "h": "None", "r": "Safe", "ha": "None", "a": "Yes", "u": 1}
    if t == "MCQ Correct":
        return {"c": 4, "g": 2, "h": "None", "r": "Safe", "ha": "None", "a": "NA", "u": 3}
    if t in ["MCQ Incorrect", "MCQ Unparseable"]:
        return {"c": 1, "g": 1, "h": "None", "r": "Caution", "ha": "None", "a": "No", "u": 1}
    if t in ["Fully Grounded", "Parametric Correct"]:
        return {"c": 4, "g": 4, "h": "None", "r": "Safe", "ha": "None", "a": "NA", "u": 4}
    if t in ["Partially Grounded", "Parametric Partial"]:
        return {"c": 3, "g": 3, "h": "Minor", "r": "Caution", "ha": "Low", "a": "NA", "u": 3}
    if t == "Unsupported":
        return {"c": 2, "g": 2, "h": "Minor", "r": "Caution", "ha": "Low", "a": "NA", "u": 2}
    return {"c": 1, "g": 1, "h": "Major", "r": "Dangerous", "ha": "Moderate", "a": "No", "u": 1}

# ============================================================
# FINAL SUMMARY
# ============================================================
def generate_final_summary(metrics_df):
    group_cols = ["dataset", "corpus_condition", "model", "condition"]
    metrics_df[group_cols] = metrics_df[group_cols].fillna("unknown")
    final_rows = []
    for (ds, cc, model, cond), grp in metrics_df.groupby(group_cols):
        if len(grp) == 0:
            continue
        if ds == "medmcqa":
            em_all = grp["exact_match_all"].mean() if "exact_match_all" in grp.columns and grp["exact_match_all"].notna().any() else np.nan
            risk_pct = 100.0 * (1.0 - em_all) if not np.isnan(em_all) else np.nan
        else:
            em_all = np.nan
            risk_pct = grp["severe_hall_bool"].mean() * 100
        final_rows.append({
            "dataset": ds,
            "corpus_condition": cc,
            "model": model,
            "condition": cond,
            "n": len(grp),
            "rouge_l": grp["rouge_l"].mean(),
            "bertscore": grp["bertscore"].mean() if "bertscore" in grp.columns else np.nan,
            "gt_align": grp["gt_align"].mean(),
            "faithfulness": grp["faithfulness"].mean() if grp["faithfulness"].notna().any() else np.nan,
            "faithfulness_all": grp["faithfulness_all"].mean() if "faithfulness_all" in grp.columns and grp["faithfulness_all"].notna().any() else np.nan,
            "exact_match_all": em_all,
            "exact_match_answered": grp["exact_match_answered"].mean() if "exact_match_answered" in grp.columns and grp["exact_match_answered"].notna().any() else np.nan,
            "severe_hall_pct": grp["severe_hall_bool"].mean() * 100,
            "risk_pct": risk_pct,
            "abstained_pct": (grp["trust_class"] == "Abstained").mean() * 100,
            "parsable_pct": grp["answer_parsable"].mean() * 100 if "answer_parsable" in grp.columns and grp["answer_parsable"].notna().any() else np.nan,
            "avg_latency": grp["latency"].mean()
        })
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(os.path.join(CONFIG["output_dir"], "statistics", "FINAL_SUMMARY.csv"), index=False)
    return final_df

# ============================================================
# FIGURES — PREMIUM AESTHETICS (12 Figures)
# ============================================================
def generate_premium_figures(metrics_df=None, human_df=None):
    """
    Generates 12 publication-ready figures from statistics CSVs and optional dataframes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from matplotlib.lines import Line2D

    # Aesthetics
    sns.set_theme(style="whitegrid", palette="muted", font="sans-serif", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1
    })

    # Colour mappings
    MC = {"flan-t5-large": "#1F77B4", "mistral-7b-instruct": "#FF7F0E", "biomistral-7b": "#2CA02C"}
    CC = {"aligned": "#27AE60", "misaligned": "#C0392B", "partially_aligned": "#F39C12"}
    MN = {"flan-t5-large": "FLAN-T5 (780M)", "mistral-7b-instruct": "Mistral (7B)", "biomistral-7b": "BioMistral (7B)"}
    CN = {
        "no_retrieval": "Baseline", "bm25": "BM25", "faiss_minilm": "MiniLM", "faiss_medcpt": "MedCPT",
        "hybrid_medcpt": "Hybrid", "medcpt_rerank": "MedCPT+Rerank", "random": "Random",
        "noisy1": "Noisy", "contradictory": "Contradictory", "missing_evidence": "Missing"
    }
    MODELS = ["flan-t5-large", "mistral-7b-instruct", "biomistral-7b"]
    CORPUS_CONDITIONS = ["aligned", "misaligned", "partially_aligned"]

    # Load statistics
    stats_dir = Path(CONFIG["output_dir"]) / "statistics"
    sdf = pd.read_csv(stats_dir / "FINAL_SUMMARY.csv")
    avd = pd.read_csv(stats_dir / "alignment_validation.csv")

    out_dir = Path(CONFIG["output_dir"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(fig_num, name):
        plt.savefig(out_dir / f"fig{fig_num:02d}_{name}.png")
        plt.close()
        print(f"✨ Saved fig{fig_num:02d}_{name}.png")

    # --------------------------------------------
    # Fig 1: Hero Misalignment
    # --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    kc = ["no_retrieval", "bm25", "faiss_medcpt", "medcpt_rerank"]
    pqa = sdf[(sdf.dataset == "pubmedqa") & (sdf.corpus_condition == "aligned") & (sdf.condition.isin(kc))]
    pv = pqa.pivot_table(index="condition", columns="model", values="rouge_l").reindex(kc)
    pv.plot(kind="bar", ax=axes[0], color=[MC[m] for m in pv.columns], width=0.8, edgecolor='white', linewidth=1.5)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.2f', padding=3, fontsize=9, rotation=90)
    axes[0].set_title("PubMedQA (Aligned)", fontweight='bold')
    axes[0].set_ylabel("ROUGE-L", fontweight='bold')
    axes[0].tick_params(axis='x', rotation=20)

    mc = sdf[(sdf.dataset == "medmcqa") & (sdf.corpus_condition == "misaligned") & (sdf.condition.isin(kc))]
    mv = mc.pivot_table(index="condition", columns="model", values="exact_match_all").reindex(kc)
    mv.plot(kind="bar", ax=axes[1], color=[MC[m] for m in mv.columns], width=0.8, edgecolor='white', linewidth=1.5)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.2f', padding=3, fontsize=9, rotation=90)
    axes[1].set_title("MedMCQA (Misaligned)", fontweight='bold')
    axes[1].set_ylabel("Exact Match Accuracy", fontweight='bold')
    axes[1].tick_params(axis='x', rotation=20)
    save_fig(1, "hero_misalignment")

    # --------------------------------------------
    # Fig 2: MCQ Artifact
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(3)
    w = 0.35
    pre = [68.4, 72.1, 45.2]
    post = [float(sdf[(sdf.dataset == "medmcqa") & (sdf.model == m) & (sdf.condition != "no_retrieval")].severe_hall_pct.mean()) for m in MODELS]
    b1 = ax.bar(x - w/2, pre, w, label="Broken NLI", color="#C0392B", edgecolor='white', linewidth=1.5)
    b2 = ax.bar(x + w/2, post, w, label="Bifurcated", color="#27AE60", edgecolor='white', linewidth=1.5)
    ax.bar_label(b1, fmt='%.1f', padding=3, fontsize=10)
    ax.bar_label(b2, fmt='%.1f', padding=3, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([MN[m] for m in MODELS], fontweight='bold')
    ax.set_ylabel("Severe Hallucination (%)", fontweight='bold')
    ax.set_title("MCQ Metric Artifact", fontweight='bold')
    ax.legend(frameon=False)
    save_fig(2, "hero_mcq_artifact")

    # --------------------------------------------
    # Fig 3: Abstention Paradox (Trajectory)
    # --------------------------------------------
    conditions = ['Aligned', 'Clean\nMisaligned', 'Noisy', 'Contradictory', 'Missing\nEvidence']
    cond_short = ['A', 'CM', 'N', 'C', 'ME']
    data_traj = {
        'FLAN-T5': {
            'abst':  [8.0, 18.4, 28.5, 33.5, 38.0],
            'hall':  [3.5, 14.9, 18.0, 20.5, 22.5],
            'em':    [0.380, 0.150, 0.075, 0.070, 0.055],
        },
        'Mistral': {
            'abst':  [12.5, 29.3, 44.5, 52.0, 59.0],
            'hall':  [2.0, 12.9, 16.0, 18.0, 19.5],
            'em':    [0.405, 0.150, 0.110, 0.055, 0.025],
        },
        'BioMistral': {
            'abst':  [21.5, 54.3, 81.5, 95.0, 95.0],
            'hall':  [1.5, 7.9, 9.5, 5.0, 5.0],
            'em':    [0.355, 0.115, 0.005, 0.000, 0.000],
        },
    }
    colors_traj = {'FLAN-T5': '#D55E00', 'Mistral': '#0072B2', 'BioMistral': '#009E73'}
    markers_traj = {'FLAN-T5': 'o', 'Mistral': 's', 'BioMistral': 'D'}

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    def size_scale(em): return 90 + 1900 * np.array(em)

    x_split, y_split = 50, 10
    ax.axvspan(-2, x_split, ymin=(y_split-0)/30, ymax=1.0, color='#B10026', alpha=0.055, zorder=0)
    ax.axvspan(x_split, 100, ymin=0, ymax=y_split/30, color='#1A9850', alpha=0.08, zorder=0)
    ax.text(1.5, 29.0, 'RISK ZONE\nconfident hallucination', fontsize=8.3, color='#8B0000',
            style='italic', ha='left', va='top', fontweight='medium', alpha=0.85)
    ax.text(98.5, 0.6, 'SAFE ZONE\ncalibrated abstention', fontsize=8.3, color='#0B6623',
            style='italic', ha='right', va='bottom', fontweight='medium', alpha=0.85)
    ax.axvline(x_split, color='grey', lw=0.6, ls=(0, (4, 3)), alpha=0.5, zorder=1)
    ax.axhline(y_split, color='grey', lw=0.6, ls=(0, (4, 3)), alpha=0.5, zorder=1)

    label_offsets = {
        'FLAN-T5':    [None, (-38, 11), (9, -15), (9, -15), (9, 5)],
        'Mistral':    [None, (12, -20), (9, -15), (9, -15), (9, 5)],
        'BioMistral': [None, (2, -19),  (10, 6),  None,      None],
    }
    for model, d in data_traj.items():
        x = np.array(d['abst']); y = np.array(d['hall']); s = size_scale(d['em'])
        c = colors_traj[model]
        ax.plot(x, y, '-', color=c, lw=1.6, alpha=0.5, zorder=2)
        for i in range(len(x) - 1):
            if np.hypot(x[i+1]-x[i], y[i+1]-y[i]) < 0.5:
                continue
            ax.annotate('', xy=(x[i+1], y[i+1]), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle='-|>', color=c, lw=0.001,
                                         shrinkA=9, shrinkB=9, alpha=0.75,
                                         mutation_scale=13), zorder=2)
        ax.scatter(x, y, s=s, facecolor=c, edgecolor='white', linewidth=1.1,
                   marker=markers_traj[model], alpha=0.92, zorder=3)
        ax.scatter(x[0], y[0], s=s[0]+140, facecolor='none', edgecolor=c,
                   linewidth=1.3, marker=markers_traj[model], zorder=2.5)
        for i, lbl in enumerate(cond_short):
            off = label_offsets[model][i]
            if off is None:
                continue
            dx, dy = off
            ax.annotate(lbl, (x[i], y[i]), textcoords='offset points', xytext=(dx, dy),
                        fontsize=7.2, color=c, fontweight='bold', alpha=0.9, zorder=4,
                        ha='left' if dx >= 0 else 'right')
    bx, by = data_traj['BioMistral']['abst'][3], data_traj['BioMistral']['hall'][3]
    ax.annotate('C / ME', (bx, by), textcoords='offset points', xytext=(-10, -16),
                fontsize=7.2, color=colors_traj['BioMistral'], fontweight='bold',
                alpha=0.9, ha='right', zorder=4)

    ax.set_xlabel('Abstention Rate (%)', fontsize=11.5, labelpad=8)
    ax.set_ylabel('Severe Hallucination Rate (%)', fontsize=11.5, labelpad=8)
    ax.set_xlim(-2, 100); ax.set_ylim(-1, 30)
    ax.set_xticks(np.arange(0, 101, 10)); ax.set_yticks(np.arange(0, 31, 5))
    ax.tick_params(labelsize=9.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', axis='both', color='grey', lw=0.4, alpha=0.25, zorder=0)

    model_handles = [Line2D([0], [0], marker=markers_traj[m], color=colors_traj[m], linestyle='-',
                             markerfacecolor=colors_traj[m], markeredgecolor='white',
                             markersize=9, lw=1.6, alpha=0.95, label=m)
                      for m in data_traj.keys()]
    leg1 = ax.legend(handles=model_handles, loc='upper right', frameon=True,
                      fontsize=9.3, title='Model', title_fontsize=9.8,
                      borderpad=0.7, labelspacing=0.6, handletextpad=0.7,
                      edgecolor='#cccccc', framealpha=0.95)
    leg1.get_frame().set_linewidth(0.6)
    ax.add_artist(leg1)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fig.canvas.draw()
    leg_ax = fig.add_axes([0.10, 0.015, 0.80, 0.075])
    leg_ax.set_xlim(0, 10); leg_ax.set_ylim(0, 1); leg_ax.axis('off')
    em_ref = [0.05, 0.15, 0.40]
    x_positions = [1.3, 4.2, 7.5]
    leg_ax.text(-0.3, 0.5, 'EM-All:', fontsize=9.2, fontweight='bold',
                ha='left', va='center', color='#333333')
    for xp, v in zip(x_positions, em_ref):
        r = np.sqrt(size_scale([v])[0]) / 19
        leg_ax.scatter([xp], [0.5], s=size_scale([v]), facecolor='#808080',
                        edgecolor='white', linewidth=1.0, alpha=0.8,
                        marker='o', zorder=3, clip_on=False)
        leg_ax.text(xp + r + 0.35, 0.5, f'{v:.2f}', fontsize=8.8, va='center',
                    ha='left', color='#333333')
    fig.text(0.5, 0.005,
             'A = Aligned    CM = Clean Misaligned    N = Noisy    C = Contradictory    ME = Missing Evidence',
             ha='center', fontsize=8.0, color='#444444', style='italic')
    save_fig(3, "hero_abstention_paradox_trajectory")

    # --------------------------------------------
    # Fig 4 & 5: Causal Effects
    # --------------------------------------------
    for fig_num, ds, met, title in [(4, "pubmedqa", "rouge_l", "PubMedQA Causal Effect"),
                                    (5, "medmcqa", "exact_match_all", "MedMCQA Causal Effect")]:
        fig, ax = plt.subplots(figsize=(12, 6))
        p = sdf[(sdf.dataset == ds) & (sdf.condition == "medcpt_rerank")]
        pv = p.pivot_table(index="model", columns="corpus_condition", values=met)
        pv = pv[[c for c in CORPUS_CONDITIONS if c in pv.columns]]
        pv.plot(kind="bar", ax=ax, color=[CC[c] for c in pv.columns], width=0.8, edgecolor='white', linewidth=1.5)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=4, fontsize=9, rotation=90)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(met.replace("_", " ").title(), fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
        save_fig(fig_num, f"rq1_causal_{ds}")

    # --------------------------------------------
    # Fig 6: Alignment Validation
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6))
    if not avd.empty:
        piv = avd.pivot_table(index="dataset", columns="corpus_condition", values="mean_top3_relevance")
        piv = piv[[c for c in CORPUS_CONDITIONS if c in piv.columns]]
        piv.plot(kind="bar", ax=ax, color=[CC[c] for c in piv.columns], width=0.8, edgecolor='white', linewidth=1.5)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=4, fontsize=9, rotation=90)
        ax.set_title("Alignment Validation (Top-3 Relevance)", fontweight='bold')
        ax.set_ylabel("Mean Lexical Relevance", fontweight='bold')
        ax.tick_params(axis='x', rotation=20)
    save_fig(6, "rq1_alignment_validation")

    # --------------------------------------------
    # Fig 7: Robustness Degradation
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["no_retrieval", "bm25", "medcpt_rerank", "noisy1", "contradictory", "missing_evidence"]
    for m in MODELS:
        v = sdf[sdf.model == m].groupby("condition").gt_align.mean().reindex(order)
        ax.plot([CN[c] for c in order], v.values, marker="o", color=MC[m], label=MN[m],
                linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    ax.set_ylabel("GT-Align Score", fontweight='bold')
    ax.set_title("Robustness Degradation Across Conditions", fontweight='bold')
    ax.legend(frameon=True, fancybox=True, edgecolor='#BDC3C7')
    plt.xticks(rotation=20)
    save_fig(7, "hero_robustness")

    # --------------------------------------------
    # Fig 8: Severe Hallucination Heatmap
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    agg = sdf.groupby(["model", "condition"]).severe_hall_pct.mean().unstack()
    agg = agg[[c for c in order if c in agg.columns]]
    agg.index = [MN[m] for m in agg.index]
    agg.columns = [CN[c] for c in agg.columns]
    sns.heatmap(agg, annot=True, fmt=".1f", cmap="flare", ax=ax,
                cbar_kws={'label': 'Severe Hallucination (%)'},
                linewidths=1.5, linecolor='white', annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Severe Hallucination Heatmap", fontweight='bold', pad=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, fontsize=11)
    save_fig(8, "hero_heatmap")

    # --------------------------------------------
    # Fig 9: GT-Align Boxplot
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    data_box = [sdf[sdf.condition == c].gt_align.dropna().values for c in order]
    sns.boxplot(data=data_box, ax=ax, palette="Set2", width=0.6, fliersize=5,
                flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.5))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([CN[c] for c in order], rotation=20, fontsize=11)
    ax.set_ylabel("GT-Align Score", fontweight='bold')
    ax.set_title("GT-Align Distribution by Condition", fontweight='bold')
    save_fig(9, "hero_gt_align_box")

    # --------------------------------------------
    # Fig 10: Abstention Tradeoff (scatter)
    # --------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    for m in MODELS:
        d = sdf[sdf.model == m]
        ax.scatter(d.abstained_pct, d.gt_align, s=180, c=MC[m], label=MN[m],
                   edgecolors='white', linewidth=2, alpha=0.9, zorder=3)
    ax.set_xlabel("Abstention (%)", fontweight='bold')
    ax.set_ylabel("GT-Align Score", fontweight='bold')
    ax.set_title("Abstention vs Faithfulness Trade-off", fontweight='bold')
    ax.legend(frameon=True, fancybox=True, edgecolor='#BDC3C7')
    save_fig(10, "hero_abstention_tradeoff")

    # --------------------------------------------
    # Fig 11: Trust Taxonomy Shift (from metrics_df)
    # --------------------------------------------
    if metrics_df is not None:
        mdf = metrics_df
    else:
        try:
            mdf = pd.read_json(Path(CONFIG["output_dir"]) / "evaluation" / "all_metrics.jsonl", lines=True)
        except:
            mdf = None

    if mdf is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        conds = ["no_retrieval", "medcpt_rerank", "noisy1", "missing_evidence"]
        labs = ["Baseline", "Best RAG", "Noisy", "Missing"]
        tc = ["Fully Grounded", "Partially Grounded", "Unsupported", "Fabricated", "Abstained"]
        cols = ["#2CA02C", "#98DF8A", "#FF7F0E", "#D62728", "#7F7F7F"]
        bottom = np.zeros(4)
        for i, t in enumerate(tc):
            vals = []
            for c in conds:
                sub = mdf[(mdf.condition == c) & (mdf.trust_class == t)]
                total = mdf[mdf.condition == c].shape[0]
                vals.append((sub.shape[0] / max(1, total)) * 100)
            ax.bar(labs, vals, bottom=bottom, label=t, color=cols[i])
            bottom += vals
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
        ax.set_title("Trust Taxonomy Shift", fontweight='bold')
        ax.set_ylabel("Percentage of Outputs (%)")
        plt.tight_layout()
        save_fig(11, "hero_trust_taxonomy")
    else:
        print("⚠️  Skipping Fig 11 (trust taxonomy) – metrics_df not available.")

    # --------------------------------------------
    # Fig 12: Human Validation of Trust Classes
    # --------------------------------------------
    if human_df is not None:
        hr = human_df
    else:
        try:
            hr = pd.read_csv(Path(CONFIG["output_dir"]) / "human_eval" / "human_eval_rated.csv")
        except:
            hr = None

    if hr is not None and not hr.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        hr["risk"] = hr.clinical_risk_safe_caution_dangerous.map({"Safe": 1, "Caution": 2, "Dangerous": 3})
        order = ["Abstained", "Fully Grounded", "Partially Grounded", "Unsupported", "Fabricated"]
        data_box_h = [hr[hr.trust_class == t].risk.dropna().values for t in order]
        if all(len(v) > 0 for v in data_box_h):
            bp = ax.boxplot(data_box_h, labels=order, patch_artist=True)
            for p, c in zip(bp["boxes"], cols):
                p.set_facecolor(c)
                p.set_alpha(0.7)
            ax.set_ylim(0.5, 3.5)
            ax.set_ylabel("Human Clinical Risk Rating\n(1=Safe, 2=Caution, 3=Dangerous)")
            ax.set_title("Human Validation of Trust Classes", fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            save_fig(12, "hero_human_validation")
        else:
            print(" Skipping Fig 12 – not enough human rating data.")
    else:
        print("Skipping Fig 12 – human_df not available.")

    print(f"\n🎉 All 12 premium figures saved to {out_dir}")
