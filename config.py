import os
from pathlib import Path

# --- General Configuration ---
DEVICE = "cuda" # Change to "cpu" if CUDA is not available
SMOKE_TEST = False # Set to True for quick pipeline validation, False for full experiment
OUTPUT_DIR = Path("./outputs") # Directory to store results, logs, etc.
SEED = 42 # Random seed for reproducibility

# --- Model Configuration ---
# Registry of models to be tested
MODEL_REGISTRY = {
    "flan-t5-small": {
        "repo": "google/flan-t5-small",
        "type": "seq2seq",
        "quantize": False,
        "max_input": 512,
        "display": "FLAN-T5-Small (smoke)",
        "params_b": 0.08,
    },
    "flan-t5-large": {
        "repo": "google/flan-t5-large",
        "type": "seq2seq",
        "quantize": False,
        "max_input": 1024,
        "display": "FLAN-T5-Large (780M)",
        "params_b": 0.78,
    },
    "mistral-7b-instruct": {
        # Placeholder - Add details for Mistral if used
        "repo": "mistralai/Mistral-7B-Instruct-v0.1", # Example repo
        "type": "causal_lm",
        "quantize": True, # Often beneficial for larger models
        "max_input": 4096,
        "display": "Mistral-7B-Instruct",
        "params_b": 7.0,
    },
    # Add more models as needed
}

# Datasets to be used for evaluation
DATASETS = ["pubmedqa", "medmcqa"]

# --- Corpus Configuration ---
# Provenance information for corpus sources
CORPUS_PROVENANCE = {
    "pubmedqa_abstracts": {
        "source": "PubMedQA dataset (qiaojin/PubMedQA, pqa_labeled train split)",
        "field": "context.contexts (concatenated into full abstracts)",
        "n_documents": None, # To be filled dynamically
        "description": "Clinical research abstracts"
    },
    "textbook_explanations": {
        "source": "MedMCQA train split (openlifescienceai/medmcqa) or External file",
        "field": "exp (explanation) or user_provided",
        "n_documents": None, # To be filled dynamically
        "description": "Medical textbook/examination explanations"
    }
}

# Configuration for different corpus conditions (aligned, misaligned, partial)
CORPUS_CONDITIONS = {
    "pubmedqa": ["aligned", "misaligned", "partially_aligned"],
    "medmcqa": ["aligned", "misaligned", "partially_aligned"]
}

# Parameters for corpus balancing and chunking
CORPUS_SIZE = 700 # Target number of documents per corpus condition (if equalizing)
MAX_CHUNKS_PER_CORPUS = 726 # Maximum number of chunks per condition after balancing
CHUNK_SIZE = 256 # Size of text chunks for retrieval
CHUNK_OVERLAP = 32 # Overlap between chunks
EQUALIZE_CORPUS_DOCS = True # Whether to balance document counts across conditions
EQUALIZE_CORPUS_CHUNKS = True # Whether to balance chunk counts within a dataset's conditions
PARTIAL_MIX_RATIO = 0.5 # Ratio for mixing documents in partially aligned condition

# --- Retrieval Configuration ---
TOP_K = 5 # Number of chunks to retrieve initially
RETRIEVERS = ["no_retrieval", "bm25", "faiss_medcpt", "medcpt_rerank"]
INCLUDE_ROBUSTNESS = True # Whether to run robustness tests like noisy, missing evidence
ROBUSTNESS_MODES = ["noisy1", "contradictory", "missing_evidence"]

# Embedding and reranking models
MINILM_ENCODER = "all-MiniLM-L6-v2"
MEDCPT_QUERY = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE = "ncbi/MedCPT-Article-Encoder"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Faithfulness / Contradiction Check Configuration ---
CONTRADICTION_THRESHOLD = 0.5 # Threshold for considering a claim contradicted
FAITHFULNESS_MEDIUM = 0.3 # Threshold for considering a claim supported (medium level)

# --- Statistical Analysis Configuration ---
MULTIPLE_TESTING_METHOD = "holm" # Method for p-value correction (e.g., 'holm', 'fdr_bh')
MIXED_EFFECTS_MAX_ROWS = None # Limit rows for mixed effects model if needed for speed

# --- Human Evaluation Simulation Configuration ---
SIMULATE_HUMAN_RATINGS = False # Set to True only for pipeline testing

# --- Alignment Validation Configuration ---
ALIGNMENT_PROBE_N = 100 # Number of samples to probe for alignment validation

# --- Visualization Configuration ---
FIGURE_DPI = 300
SAVEFIG_BBOX = "tight"
COLORS = {
    # Define colors for plots, e.g., models, datasets
    # Example placeholders:
    # "pubmedqa": "#1f77b4",
    # "medmcqa": "#ff7f0e",
    # "flan-t5-large": "#2ca02c",
    # Add as needed based on your plotting logic
}
# Add other visualization settings as needed

# --- Paths ---
FIG_DIR = OUTPUT_DIR / "figures"
STATS_DIR = OUTPUT_DIR / "statistics"
CORPUS_DIR = OUTPUT_DIR / "corpus"
GENERATION_DIR = OUTPUT_DIR / "generation"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
STATS_DIR.mkdir(exist_ok=True)
CORPUS_DIR.mkdir(exist_ok=True)
GENERATION_DIR.mkdir(exist_ok=True)

# --- Validation ---
assert SMOKE_TEST or not SMOKE_TEST # Just a check, can be removed later
print(f"Configuration loaded. Output directory: {OUTPUT_DIR}")
if SMOKE_TEST:
    print("WARNING: SMOKE_TEST is enabled. Results will be from a small sample.")
