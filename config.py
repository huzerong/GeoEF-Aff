import os
import torch

# 获取当前文件所在目录的绝对路径
VARIANT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(VARIANT_DIR)
BASE_DIR = PROJECT_ROOT
FOLDX_FEATURE_MODE = "wt_mut_delta"
FOLDX_FEATURE_DIM = 3
REQUIRE_MUTATION_FOLDX_FEATURES = os.environ.get(
    "REQUIRE_MUTATION_FOLDX_FEATURES", "1"
).lower() in {"1", "true", "yes"}
PRECOMPUTED_DIR = os.environ.get(
    "PRECOMPUTED_DIR",
    os.path.join(PROJECT_ROOT, "foldx_3feature_retrain", "precomputed_samples_3feat"),
)
CV_NUM_FOLDS = 3
CV_SPLIT_MODE = "sample"
CV_RANDOM_SEED = 42
CV_OUTPUT_DIR = os.path.join(VARIANT_DIR, "three_fold_results")

CSV_PATH = os.environ.get(
    "CSV_PATH",
    os.path.join(BASE_DIR, "data", "SKEMPI_v2", "skempi_v2.csv"),
)
PDB_DIR = os.path.join(BASE_DIR, "data", "SKEMPI_v2", "PDBs")
FOLDX_VERSION = os.environ.get("FOLDX_VERSION", "foldx5")
FOLDX_PATH = (
    os.environ.get("FOLDX5_PATH")
    or os.environ.get("FOLDX_PATH")
    or os.path.join(BASE_DIR, "foldx")
)
FOLDX_TEMP_DIR = os.environ.get("FOLDX_TEMP_DIR", os.path.join(BASE_DIR, "foldx_temp"))
FOLDX_CACHE_DIR = os.environ.get("FOLDX_CACHE_DIR", os.path.join(BASE_DIR, "foldx_cache"))
MUTATION_FOLDX_CACHE_DIR = os.environ.get(
    "MUTATION_FOLDX_CACHE_DIR",
    os.path.join(BASE_DIR, "foldx_mutation_cache"),
)
USE_MUTATION_FOLDX_FEATURES = os.environ.get("USE_MUTATION_FOLDX_FEATURES", "1").lower() in {"1", "true", "yes"}
FOLDX_REPAIR_TIMEOUT = int(os.environ.get("FOLDX_REPAIR_TIMEOUT", "900"))
FOLDX_ANALYSE_TIMEOUT = int(os.environ.get("FOLDX_ANALYSE_TIMEOUT", "600"))
FOLDX_RETRY_FAILED = os.environ.get("FOLDX_RETRY_FAILED", "0").lower() in {"1", "true", "yes"}
ENABLE_FOLDX = os.environ.get("ENABLE_FOLDX", "1").lower() in {"1", "true", "yes"}
CASE_STUDY_BUILD_MUTANT_PDB = os.environ.get(
    "CASE_STUDY_BUILD_MUTANT_PDB",
    "1" if os.environ.get("USE_MUTATION_FOLDX_FEATURES", "1").lower() in {"1", "true", "yes"} else "0",
).lower() in {"1", "true", "yes"}
SPLIT_MODE = os.environ.get("SPLIT_MODE", "random")
SPLIT_GROUP_COL = os.environ.get("SPLIT_GROUP_COL", "#Pdb")
FOLDX_CACHE_DIR = os.path.join(BASE_DIR, "foldx_cache")  # 新增FoldX缓存目录
# 暂时禁用 FoldX 修复和分析
ENABLE_FOLDX = True
BEST_MODEL_PATH = os.path.join(CV_OUTPUT_DIR, "best_model_3feature_cv.pth")
CHECKPOINT_DIR = os.path.join(CV_OUTPUT_DIR, "checkpoints")
PLOT_PATH = os.path.join(CV_OUTPUT_DIR, "oof_prediction_scatterplot.png")
LOG_PATH = os.path.join(VARIANT_DIR, "training_3fold.log")
FOLDX_CACHE_DIR = os.environ.get("FOLDX_CACHE_DIR", FOLDX_CACHE_DIR)

ESM_MODEL_NAME = "esm2_t30_150M_UR50D"  # 150M 参数, 640-dim 嵌入
HIDDEN_DIM = 512  # 融合层维度，匹配 150M ESM
DROPOUT = 0.2

USE_DYNAMIC_MODELING = True
USE_STRUCTURE_FEATURES = True
RAAD_HIDDEN_DIM = 512   # GNN 隐藏维度 (3x98GB 显存充足，从 256 提升)
RAAD_LAYERS = 8          # GNN 层数
EDGE_TYPES = 8
RBALL_RADIUS = 10.0
KNN_K = 10
USE_ATOM_FEATURES = True
USE_ATTENTION = True
USE_RESIDUAL = True
COORDS_AGG = 'mean'
MUTATION_LOCAL_RADIUS = 10.0
ESM_MUTATION_WINDOW_RADIUS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 多卡支持 (3x RTX PRO 6000 Blackwell 98GB)
NUM_GPUS = torch.cuda.device_count()
PARALLEL_BACKEND = "ddp"  # launch with torchrun; use "dataparallel" only for legacy debugging
DDP_FIND_UNUSED_PARAMETERS = True  # dynamic GNN edge types can leave relation-specific layers unused per batch/rank
LEARNING_RATE = 1e-4
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-7
NUM_EPOCHS = 70
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 16           # 48 OOM (GNN edge_input 拼接爆显存)，16 per-GPU ~5 samples
GRADIENT_ACCUMULATION_STEPS = 18  # 等效 batch 16*18=288 ≈ 256
DDP_BATCH_SIZE_PER_RANK = 2
DDP_GRADIENT_ACCUMULATION_STEPS = None  # None keeps the original effective global batch approximately unchanged
NUM_WORKERS = 24          # 66 核 CPU，每卡 8 worker (原 16)
DDP_NUM_WORKERS_PER_RANK = max(1, NUM_WORKERS // max(NUM_GPUS, 1))
TRAIN_SPLIT = 0.8
WEIGHT_DECAY = 5e-5
MAX_STRUCTURE_ATOMS = 4096


# 学习率 warmup + cosine annealing
WARMUP_EPOCHS = 0
LR_SCHEDULE = 'reduce_lr_on_plateau'

DISK_CACHE_DIR = os.path.join(VARIANT_DIR, "structure_cache")
STRUCTURE_CACHE_SIZE = 2000
FOLDX_CACHE_SIZE = 5000
MIXED_PRECISION = True   # 混合精度加速
USE_BF16 = True          # Blackwell 原生 BF16，比 FP16 训练更稳定
TORCH_COMPILE = False    # 动态 shape (结构数据) 不适合 torch.compile reduce-overhead
PIN_MEMORY = True
ENABLE_GRADIENT_CHECKPOINTING = False  # 3x98GB 显存无需梯度检查点

# 预计算 ESM 嵌入后设为 True，训练时跳过加载 ESM 模型，节省显存
USE_PRECOMPUTED_ESM = True

# 调试选项：仅使用已缓存的 FoldX 数据进行训练
# 如果设置为 True，DataLoader 将只加载 foldx_cache 中存在的样本 (~200个)
# 这用于快速验证代码流程，正式训练请设为 False
USE_ONLY_CACHED_FOLDX = False

# ====================== 继续训练配置 ======================
RESUME = False                  # 从头训练，不自动加载旧权重
RESUME_FROM = ""                # 如需手动续训，再显式指定 checkpoint
# ==========================================================
