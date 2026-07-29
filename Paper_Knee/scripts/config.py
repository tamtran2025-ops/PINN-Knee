"""
Configuration for PINN-UQ Knee-Point Prediction in Li-Ion Batteries
====================================================================
Target: Applied Energy / Journal of Power Sources (Q1)
All hyperparameters in one place. Change here, affects everywhere.
"""

import os
import torch

# ================================================================
# PATHS
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
CHECKPOINTS_DIR = os.path.join(PROJECT_DIR, 'checkpoints')

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ================================================================
# DEVICE
# ================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================
# DATASETS
# ================================================================
NASA_DIR = os.path.join(DATA_DIR, 'nasa')
CALCE_DIR = os.path.join(DATA_DIR, 'calce')
SEVERSON_DIR = os.path.join(DATA_DIR, 'severson')

NASA_CELLS = ['B0005', 'B0006', 'B0007']
CALCE_CELLS = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']

# EOL threshold for RUL calculation (fraction of Q0)
EOL_THRESHOLD = 0.80  # RUL = cycles until SOH < 80%

# ================================================================
# KNEE-POINT DETECTION
# ================================================================
KNEE_DETECTION_METHOD = 'ensemble'  # 'bacon_watts', 'curvature', 'second_derivative', 'ensemble'
KNEE_SMOOTH_WINDOW = 5              # Smoothing window for capacity curve
KNEE_MIN_CYCLE = 30                 # Minimum cycle to consider for knee
KNEE_MAX_FRACTION = 0.95            # Maximum fraction of lifetime to search
KNEE_ACCELERATION_RATIO = 1.2       # Acceleration ratio threshold

# ================================================================
# TRAINING
# ================================================================
N_EPOCHS = 3000
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
SCHEDULER_T_MAX = 3000
N_SEEDS = 3  # 5-fold CV x 3 seeds = 15 evaluations per config
N_COLLOCATION = 500  # Physics loss collocation points
PHYSICS_LAMBDA = {
    'monotonic_decay': 1.0,
    'sei_sqrt_t': 1.0,
    'knee_transition': 1.0,
    'degradation_ode': 1.0,
    'initial_condition': 1.0,
}  # Weights for physics loss components

# Data split fractions
TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15  # For conformal prediction calibration
# Test fraction = 1 - TRAIN_FRACTION - CALIBRATION_FRACTION = 0.15

# Early prediction: how many early cycles to use
EARLY_CYCLE_COUNTS = [50, 100, 150]
EARLY_CYCLE_EXTENDED = [50, 75, 100, 125, 150, 200]

# Maximum cycle life (for normalization)
MAX_CYCLE_LIFE = 2500

# Legacy train fractions (for compatibility)
TRAIN_FRACTIONS = [0.15, 0.25, 0.40, 0.60]

# ================================================================
# MODEL ARCHITECTURE
# ================================================================
HIDDEN_SIZE = 128
N_LAYERS = 3
DROPOUT_RATE = 0.1  # For MC Dropout
WINDOW_SIZE = 10    # For sequence models (LSTM, GRU, Transformer)

# ================================================================
# UNCERTAINTY QUANTIFICATION
# ================================================================
MC_DROPOUT_SAMPLES = 100       # Forward passes for MC Dropout
CONFORMAL_ALPHA = 0.05         # 95% confidence level
CONFORMAL_CAL_FRACTION = 0.2   # 20% of training data for calibration
ENSEMBLE_SIZE = 5              # Number of models in ensemble

# ================================================================
# TRANSFER LEARNING
# ================================================================
TRANSFER_PRETRAIN_EPOCHS = 2000
TRANSFER_FINETUNE_EPOCHS = 1000
TRANSFER_FINETUNE_LR = 5e-4
TRANSFER_FRACTIONS = [0.05, 0.10, 0.15, 0.25]  # Few-shot fractions

# ================================================================
# MODELS TO RUN
# ================================================================
MODEL_NAMES = [
    'PINN_Knee',       # Proposed: PINN for knee-point prediction
    'PINN_UQ',         # PINN + Double-Exp + MC Dropout
    'Pure_NN',         # MLP without physics
    'LSTM',            # 2-layer LSTM
    'GRU',             # 2-layer GRU
    'Transformer',     # Encoder-only transformer
    'Informer',        # ProbSparse attention
    'Bayesian_LSTM',   # LSTM + MC Dropout (UQ baseline)
    'Ensemble_NN',     # 5x Pure NN ensemble (UQ baseline)
    'Neural_ODE',      # Continuous-time model
    'XGBoost',         # Gradient boosting (classical baseline)
]

# ================================================================
# CHECKPOINT / RESUME
# ================================================================
CHECKPOINT_INTERVAL = 1    # Save after every N completed runs
RESULTS_CSV = os.path.join(RESULTS_DIR, 'all_experiments.csv')
TRANSFER_CSV = os.path.join(RESULTS_DIR, 'transfer_experiments.csv')
UQ_CSV = os.path.join(RESULTS_DIR, 'uq_experiments.csv')
ABLATION_CSV = os.path.join(RESULTS_DIR, 'ablation_experiments.csv')
EARLY_CSV = os.path.join(RESULTS_DIR, 'early_experiments.csv')

# ================================================================
# FIGURES
# ================================================================
FIG_DPI = 300
FIG_FORMAT = 'png'  # 'png' or 'pdf'

# ================================================================
# STATISTICAL TESTING
# ================================================================
STAT_ALPHA = 0.05  # Significance level

# ================================================================
# LOGGING
# ================================================================
LOG_FILE = os.path.join(PROJECT_DIR, 'training.log')


def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("  PINN-UQ Knee-Point Prediction  -  Configuration")
    print("=" * 70)
    print(f"  Device:            {DEVICE}")
    print(f"  Epochs:            {N_EPOCHS}")
    print(f"  Seeds:             {N_SEEDS}")
    print(f"  Early cycles:      {EARLY_CYCLE_COUNTS}")
    print(f"  Models:            {len(MODEL_NAMES)} ({', '.join(MODEL_NAMES)})")
    print(f"  Hidden size:       {HIDDEN_SIZE}")
    print(f"  MC samples:        {MC_DROPOUT_SAMPLES}")
    print(f"  Conformal alpha:   {CONFORMAL_ALPHA}")
    print(f"  Physics lambda:    {PHYSICS_LAMBDA}")
    print(f"  Train fraction:    {TRAIN_FRACTION}")
    print(f"  Max cycle life:    {MAX_CYCLE_LIFE}")
    print(f"  Results CSV:       {RESULTS_CSV}")
    print(f"  Checkpoints:       {CHECKPOINTS_DIR}")
    print("=" * 70)
