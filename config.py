

import os

import torch


def _env_flag(name, default):
    value = os.environ.get(name, "1" if default else "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
VARIANT_DIR = PACKAGE_ROOT  # Backward-compatible name used by existing scripts.
PROJECT_ROOT = os.path.abspath(
    os.environ.get("GEOEF_AFF_PROJECT_ROOT", PACKAGE_ROOT)
)
BASE_DIR = os.path.abspath(
    os.environ.get("GEOEF_AFF_BASE_DIR", PROJECT_ROOT)
)
DATA_DIR = os.path.abspath(
    os.environ.get("GEOEF_AFF_DATA_DIR", os.path.join(BASE_DIR, "data"))
)


SPLIT_MODE = os.environ.get("SPLIT_MODE", "complex")
SPLIT_GROUP_COL = os.environ.get("SPLIT_GROUP_COL", "#Pdb")
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SPLIT_RATIO_TAG = "80_10_10"
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))

EXPERIMENT_NAME = (
    "hd384_raad3_beneficial_grouprank_muttype_localtoken32"
)
OUTPUT_DIR = os.path.abspath(
    os.environ.get(
        "OUTPUT_DIR",
        os.path.join(BASE_DIR, "outputs", f"single_split_seed{RANDOM_SEED}"),
    )
)
MODEL_DIR = os.path.abspath(
    os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
)
FROZEN_SPLIT_DIR = os.path.abspath(
    os.environ.get(
        "FROZEN_SPLIT_DIR",
        os.path.join(BASE_DIR, "splits"),
    )
)
SPLIT_DIR = os.path.abspath(
    os.environ.get("SPLIT_DIR", os.path.join(OUTPUT_DIR, "splits"))
)
CHECKPOINT_DIR = os.path.abspath(
    os.environ.get(
        "CHECKPOINT_DIR",
        os.path.join(OUTPUT_DIR, "training_checkpoints"),
    )
)
BEST_MODEL_FILENAME = "best_model.pth"
BEST_MODEL_PATH = os.path.abspath(
    os.environ.get(
        "BEST_MODEL_PATH",
        os.path.join(MODEL_DIR, BEST_MODEL_FILENAME),
    )
)
TRAINED_MODEL_PATH = os.path.abspath(
    os.environ.get(
        "TRAINED_MODEL_PATH",
        os.path.join(OUTPUT_DIR, BEST_MODEL_FILENAME),
    )
)
_default_split_json = (
    os.path.join(
        FROZEN_SPLIT_DIR,
        "fold_01",
        "fold_01_split.json",
    )
    if RANDOM_SEED == 42
    else ""
)
SPLIT_JSON = os.path.abspath(
    os.environ.get("SPLIT_JSON", _default_split_json)
) if (os.environ.get("SPLIT_JSON", _default_split_json)) else ""
PLOT_PATH = os.path.join(OUTPUT_DIR, "prediction_scatterplot.png")
LOG_PATH = os.path.join(OUTPUT_DIR, "training.log")
TRAINING_AUDIT_PATH = os.path.join(OUTPUT_DIR, "training_audit.json")
VALIDATION_METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_metrics.csv",
)

# Dataset and precomputed-feature paths.
CSV_PATH = os.path.abspath(
    os.environ.get(
        "CSV_PATH",
        os.path.join(DATA_DIR, "SKEMPI_v2", "skempi_v2.csv"),
    )
)
PDB_DIR = os.path.abspath(
    os.environ.get(
        "PDB_DIR",
        os.path.join(DATA_DIR, "SKEMPI_v2", "PDBs"),
    )
)
PRECOMPUTED_DIR = os.path.abspath(
    os.environ.get(
        "PRECOMPUTED_DIR",
        os.path.join(DATA_DIR, "precomputed_samples"),
    )
)
SOURCE_PRECOMPUTED_DIR = os.path.abspath(
    os.environ.get(
        "SOURCE_PRECOMPUTED_DIR",
        os.path.join(DATA_DIR, "source_precomputed_samples"),
    )
)
SOURCE_STRUCTURE_CACHE_DIR = os.path.abspath(
    os.environ.get(
        "SOURCE_STRUCTURE_CACHE_DIR",
        os.path.join(DATA_DIR, "structure_cache"),
    )
)
DISK_CACHE_DIR = os.path.abspath(
    os.environ.get("DISK_CACHE_DIR", SOURCE_STRUCTURE_CACHE_DIR)
)

# FoldX is external licensed software and is not distributed with this package.
FOLDX_VERSION = os.environ.get("FOLDX_VERSION", "foldx5")
FOLDX_PATH = os.path.abspath(
    os.environ.get("FOLDX5_PATH")
    or os.environ.get("FOLDX_PATH")
    or os.path.join(BASE_DIR, "foldx")
)
FOLDX_TEMP_DIR = os.path.abspath(
    os.environ.get(
        "FOLDX_TEMP_DIR",
        os.path.join(BASE_DIR, "runtime", "foldx_temp"),
    )
)
FOLDX_CACHE_DIR = os.path.abspath(
    os.environ.get(
        "FOLDX_CACHE_DIR",
        os.path.join(DATA_DIR, "foldx_cache"),
    )
)
MUTATION_FOLDX_CACHE_DIR = os.path.abspath(
    os.environ.get(
        "MUTATION_FOLDX_CACHE_DIR",
        os.path.join(DATA_DIR, "foldx_mutation_cache"),
    )
)
FOLDX_FEATURE_MODE = "wt_mut_delta"
FOLDX_FEATURE_DIM = 3
ENABLE_FOLDX = _env_flag("ENABLE_FOLDX", True)
USE_MUTATION_FOLDX_FEATURES = _env_flag(
    "USE_MUTATION_FOLDX_FEATURES",
    True,
)
REQUIRE_MUTATION_FOLDX_FEATURES = _env_flag(
    "REQUIRE_MUTATION_FOLDX_FEATURES",
    True,
)
FILTER_FAILED_MUTATION_FOLDX = _env_flag(
    "FILTER_FAILED_MUTATION_FOLDX",
    True,
)
FILTER_STRUCTURE_ALIGNMENT_FAILURES = _env_flag(
    "FILTER_STRUCTURE_ALIGNMENT_FAILURES",
    True,
)
FOLDX_REPAIR_TIMEOUT = int(os.environ.get("FOLDX_REPAIR_TIMEOUT", "900"))
FOLDX_ANALYSE_TIMEOUT = int(os.environ.get("FOLDX_ANALYSE_TIMEOUT", "600"))
FOLDX_RETRY_FAILED = _env_flag("FOLDX_RETRY_FAILED", False)

# Case-study paths.
CASE_STUDY_DIR = os.path.abspath(
    os.environ.get(
        "CASE_STUDY_DIR",
        os.path.join(DATA_DIR, "case_studies"),
    )
)
CASE_STUDY_PREPARED_CACHE_DIR = os.path.abspath(
    os.environ.get(
        "CASE_STUDY_PREPARED_CACHE_DIR",
        os.path.join(DATA_DIR, "case_study_prepared_cache"),
    )
)
CASE_STUDY_BUILD_MUTANT_PDB = _env_flag(
    "CASE_STUDY_BUILD_MUTANT_PDB",
    USE_MUTATION_FOLDX_FEATURES,
)

# Model architecture.
ESM_MODEL_NAME = "esm2_t33_650M_UR50D"
HIDDEN_DIM = 384
DROPOUT = 0.3
USE_DYNAMIC_MODELING = True
USE_STRUCTURE_FEATURES = True
RAAD_HIDDEN_DIM = 256
RAAD_LAYERS = 3
EDGE_TYPES = 8
RBALL_RADIUS = 10.0
KNN_K = 10
USE_ATOM_FEATURES = True
USE_ATTENTION = True
USE_RESIDUAL = True
COORDS_AGG = "mean"
MUTATION_LOCAL_RADIUS = 10.0
ESM_MUTATION_WINDOW_RADIUS = 8
ESM_LOCAL_MAX_TOKENS = 32
STRUCT_LOCAL_MAX_RESIDUES = 32
ESM_LOCAL_TOKEN_VERSION = 1
MUTATION_TYPE_FEATURE_VERSION = 2
STRUCTURE_CHAIN_MAPPING_VERSION = 2
REQUIRE_MUTATION_TYPE_FEATURES = True
STRICT_MUTATION_WT_CHECK = True
MAX_STRUCTURE_ATOMS = 4096
DETERMINISTIC_STRUCTURE_SAMPLING = True

# Training.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
PARALLEL_BACKEND = "ddp"
DDP_FIND_UNUSED_PARAMETERS = True
LEARNING_RATE = 7.5e-5
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6
NUM_EPOCHS = 65
EARLY_STOPPING_PATIENCE = 8
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 18
DDP_BATCH_SIZE_PER_RANK = int(
    os.environ.get("DDP_BATCH_SIZE_PER_RANK", "16")
)
DDP_GRADIENT_ACCUMULATION_STEPS = int(
    os.environ.get("DDP_GRADIENT_ACCUMULATION_STEPS", "2")
)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "24"))
DDP_NUM_WORKERS_PER_RANK = max(1, NUM_WORKERS // max(NUM_GPUS, 1))
WEIGHT_DECAY = 2e-4
WARMUP_EPOCHS = 5
LR_SCHEDULE = "cosine"
LOSS_FUNCTION = "beneficial_group_rank"
SMOOTH_L1_BETA = 1.0
BENEFICIAL_THRESHOLD = 0.0
BENEFICIAL_SAMPLE_WEIGHT = 2.0
PAIRWISE_RANK_WEIGHT = 0.25
SITE_PAIR_WEIGHT = 1.0
COMPLEX_PAIR_WEIGHT = 0.25
PAIRWISE_TEMPERATURE = 0.5
PAIRWISE_MIN_LABEL_GAP = 0.2
PAIRWISE_MAX_PAIRS = 4096
GROUP_AWARE_BATCHING = True
RUN_TRAINING_AUDIT = True
MAX_ZERO_SAME_SITE_BATCH_RATE = 0.20
MODEL_SELECTION_METRIC = "val_loss"

# Runtime and caching.
STRUCTURE_CACHE_SIZE = 2000
FOLDX_CACHE_SIZE = 5000
MIXED_PRECISION = True
USE_BF16 = True
TORCH_COMPILE = False
PIN_MEMORY = True
ENABLE_GRADIENT_CHECKPOINTING = False
USE_PRECOMPUTED_ESM = True
VALIDATE_PRECOMPUTED_CACHE_ON_LOAD = False
USE_ONLY_CACHED_FOLDX = False
RESUME = False
RESUME_FROM = ""
