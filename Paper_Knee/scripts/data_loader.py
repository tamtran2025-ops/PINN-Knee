"""
Data Loader for Paper: Physics-Constrained Deep Learning with Conformal
Prediction for Early Knee-Point Prediction in Lithium-Ion Batteries.

Reuses Paper 7's dataset loaders (NASA, CALCE, Severson) and augments each
cell with an ensemble-detected knee-point cycle. Provides cell-level
train / calibration / test splits for conformal prediction experiments.

Target: Applied Energy / Journal of Power Sources (Q1)
"""

import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
#   Imports from Paper 7 (use importlib to avoid ALL name collisions)
# ---------------------------------------------------------------------------
import importlib.util

_p7_scripts = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'scripts',
))

def _load_paper7_data_loader():
    """Load Paper 7's data_loader module without polluting sys.path/modules."""
    # Save current state
    saved_config = sys.modules.get('config')
    saved_path = list(sys.path)

    try:
        # Add Paper 7 scripts to path temporarily
        if _p7_scripts not in sys.path:
            sys.path.insert(0, _p7_scripts)

        # Load Paper 7's config under its own name
        cfg_path = os.path.join(_p7_scripts, 'config.py')
        spec_cfg = importlib.util.spec_from_file_location('paper7_config', cfg_path)
        p7_config = importlib.util.module_from_spec(spec_cfg)
        spec_cfg.loader.exec_module(p7_config)

        # Swap config so Paper 7's data_loader sees Paper 7's config
        sys.modules['config'] = p7_config

        # Load Paper 7's data_loader
        dl_path = os.path.join(_p7_scripts, 'data_loader.py')
        spec_dl = importlib.util.spec_from_file_location('paper7_data_loader', dl_path)
        p7_dl = importlib.util.module_from_spec(spec_dl)
        spec_dl.loader.exec_module(p7_dl)

        return p7_dl
    finally:
        # Restore original state
        if saved_config is not None:
            sys.modules['config'] = saved_config
        elif 'config' in sys.modules and sys.modules['config'].__file__ and \
             'Paper_Knee' not in sys.modules['config'].__file__:
            del sys.modules['config']
        sys.path[:] = saved_path

# Try to load Paper 7 legacy loaders; fall back gracefully if folder deleted.
try:
    _p7_loader = _load_paper7_data_loader()
    _load_nasa = _p7_loader.load_nasa_cells
    _load_calce = _p7_loader.load_calce_cells
    _load_severson = _p7_loader.load_severson_cells
    _LEGACY_OK = True
except (FileNotFoundError, AttributeError, ImportError) as _e:
    print(f"[data_loader] Legacy Paper 7 loaders unavailable ({_e}); "
          f"falling back to cached data only.")
    _load_nasa = lambda: []
    _load_calce = lambda: []
    _load_severson = lambda: []
    _LEGACY_OK = False

# ---------------------------------------------------------------------------
#   Imports from local Paper_Knee modules
# ---------------------------------------------------------------------------
from knee_detection import (                    # noqa: E402
    detect_knee_ensemble,
    validate_knee_point,
)
from config import (                            # noqa: E402
    DATA_DIR,
    KNEE_DETECTION_METHOD,
    KNEE_MIN_CYCLE,
    TRAIN_FRACTION,
    CALIBRATION_FRACTION,
)


# ======================================================================
#   Public API
# ======================================================================

def load_all_cells_with_knees(include_severson: bool = True):
    """Load every battery cell and annotate with a detected knee-point.

    Workflow
    --------
    1. Load raw cells via Paper 7 loaders (NASA, CALCE, Severson).
    2. Run the ensemble knee-detection algorithm on each cell.
    3. Validate each detected knee-point (plausibility checks).
    4. Keep only cells whose knee passes validation.

    Parameters
    ----------
    include_severson : bool, default True
        Whether to include the Severson dataset (~124 cells).

    Returns
    -------
    valid_cells : list[dict]
        Each dict contains all original keys plus:
        - ``'knee_cycle'``    : int, detected knee-point cycle
        - ``'knee_details'``  : dict, per-method results from ensemble
        - ``'knee_agreement'``: float in [0, 1], inter-method agreement
        - ``'knee_diagnostics'``: dict, validation diagnostics
    """
    print("\n" + "=" * 70)
    print("  Loading Battery Datasets (with Knee-Point Detection)")
    print("=" * 70)

    # Fast path: use cached Severson pickle if legacy loaders unavailable
    if not _LEGACY_OK:
        import pickle
        cache_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'results', '_severson_cache.pkl'
        )
        if os.path.exists(cache_path):
            print(f"\n  Loading from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            valid = [c for c in cached
                     if c.get('has_knee_point', False) and c.get('knee_cycle') is not None]
            print(f"  Cached cells with valid knee: {len(valid)}")
            return valid
        raise RuntimeError(
            "Legacy Paper 7 loaders and cache both unavailable. "
            "Restore Paper 7/scripts/ or re-generate results/_severson_cache.pkl."
        )

    # ------------------------------------------------------------------
    #   Step 1: load raw cells
    # ------------------------------------------------------------------
    all_cells = []

    print("\n[1/3] NASA Battery Dataset ...")
    nasa_cells = _load_nasa()
    all_cells.extend(nasa_cells)

    print("\n[2/3] CALCE CS2 Dataset ...")
    calce_cells = _load_calce()
    all_cells.extend(calce_cells)

    if include_severson:
        print("\n[3/3] Severson Dataset ...")
        sev_cells = _load_severson()
        all_cells.extend(sev_cells)

    print(f"\n  Raw cells loaded: {len(all_cells)}")

    # ------------------------------------------------------------------
    #   Step 2: detect knee-point for every cell
    # ------------------------------------------------------------------
    print("\n  Running ensemble knee-point detection ...")
    for cell in all_cells:
        cycles   = cell['cycles']
        capacity = cell['capacity']

        if len(capacity) < KNEE_MIN_CYCLE:
            cell['knee_cycle']       = None
            cell['knee_details']     = {}
            cell['knee_agreement']   = 0.0
            cell['knee_diagnostics'] = {'reason': 'too_short'}
            continue

        knee_cycle, per_method, agreement = detect_knee_ensemble(
            cycles, capacity,
        )

        cell['knee_cycle']     = knee_cycle
        cell['knee_details']   = per_method
        cell['knee_agreement'] = agreement

    # ------------------------------------------------------------------
    #   Step 3: validate each knee-point
    # ------------------------------------------------------------------
    valid_cells = []
    rejected    = {'too_short': 0, 'no_knee': 0, 'invalid': 0}

    for cell in all_cells:
        knee = cell.get('knee_cycle')
        if knee is None:
            reason = cell.get('knee_diagnostics', {}).get('reason', 'no_knee')
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        is_valid, diagnostics = validate_knee_point(
            cell['cycles'], cell['capacity'], knee,
        )
        cell['knee_diagnostics'] = diagnostics

        if is_valid:
            valid_cells.append(cell)
        else:
            rejected['invalid'] += 1

    # ------------------------------------------------------------------
    #   Step 4: summary
    # ------------------------------------------------------------------
    _print_summary(all_cells, valid_cells, rejected)
    return valid_cells


def get_train_cal_test_split(
    cells,
    train_frac: float = TRAIN_FRACTION,
    cal_frac:   float = CALIBRATION_FRACTION,
    seed:       int   = 42,
):
    """Cell-level split for conformal prediction experiments.

    The split is stratified by dataset origin so that each subset contains
    a representative mix of NASA, CALCE, and Severson cells.

    Parameters
    ----------
    cells : list[dict]
        List of cell dicts (must contain ``'dataset'`` key).
    train_frac : float, default 0.60
        Fraction of cells for training.
    cal_frac : float, default 0.20
        Fraction of cells for conformal calibration.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    train_cells : list[dict]
    cal_cells   : list[dict]
    test_cells  : list[dict]
    """
    rng = np.random.RandomState(seed)

    # Group cells by dataset for stratified splitting
    groups = {}
    for cell in cells:
        ds = cell.get('dataset', 'unknown')
        groups.setdefault(ds, []).append(cell)

    train_cells, cal_cells, test_cells = [], [], []

    for ds_name in sorted(groups.keys()):
        ds_cells = groups[ds_name]
        n = len(ds_cells)
        indices = rng.permutation(n)

        n_train = max(1, int(round(n * train_frac)))
        n_cal   = max(1, int(round(n * cal_frac)))
        # Remaining go to test (at least 1 if possible)
        n_test  = max(1, n - n_train - n_cal)

        # Rebalance if sum exceeds n
        if n_train + n_cal + n_test > n:
            n_cal = max(1, n - n_train - n_test)
        if n_train + n_cal + n_test > n:
            n_train = n - n_cal - n_test

        train_idx = indices[:n_train]
        cal_idx   = indices[n_train:n_train + n_cal]
        test_idx  = indices[n_train + n_cal:]

        train_cells.extend([ds_cells[i] for i in train_idx])
        cal_cells.extend([ds_cells[i] for i in cal_idx])
        test_cells.extend([ds_cells[i] for i in test_idx])

    # Final shuffle across datasets
    rng.shuffle(train_cells)
    rng.shuffle(cal_cells)
    rng.shuffle(test_cells)

    print("\n" + "-" * 50)
    print("  Cell-level data split (train / cal / test)")
    print("-" * 50)
    print(f"  Train:       {len(train_cells):>4d}  ({100 * len(train_cells) / max(1, len(cells)):.0f}%)")
    print(f"  Calibration: {len(cal_cells):>4d}  ({100 * len(cal_cells) / max(1, len(cells)):.0f}%)")
    print(f"  Test:        {len(test_cells):>4d}  ({100 * len(test_cells) / max(1, len(cells)):.0f}%)")

    # Per-dataset breakdown
    for subset_name, subset in [('Train', train_cells),
                                ('Cal',   cal_cells),
                                ('Test',  test_cells)]:
        ds_counts = {}
        for c in subset:
            ds_counts[c['dataset']] = ds_counts.get(c['dataset'], 0) + 1
        breakdown = ', '.join(f"{k}: {v}" for k, v in sorted(ds_counts.items()))
        print(f"    {subset_name:>5s} -> {breakdown}")
    print("-" * 50)

    return train_cells, cal_cells, test_cells


# ======================================================================
#   Private helpers
# ======================================================================

def _print_summary(all_cells, valid_cells, rejected):
    """Print a concise summary table after loading and knee detection."""
    print("\n" + "=" * 70)
    print("  Knee-Point Detection Summary")
    print("=" * 70)
    print(f"  Total cells loaded:       {len(all_cells)}")
    print(f"  Valid knees (kept):       {len(valid_cells)}")
    total_rejected = sum(rejected.values())
    print(f"  Rejected:                 {total_rejected}")
    for reason, count in sorted(rejected.items()):
        if count > 0:
            print(f"    - {reason:<25s}: {count}")

    # Per-dataset breakdown
    ds_total = {}
    ds_valid = {}
    for c in all_cells:
        ds = c['dataset']
        ds_total[ds] = ds_total.get(ds, 0) + 1
    for c in valid_cells:
        ds = c['dataset']
        ds_valid[ds] = ds_valid.get(ds, 0) + 1

    print("\n  Per-dataset:")
    for ds in sorted(ds_total.keys()):
        total = ds_total[ds]
        valid = ds_valid.get(ds, 0)
        print(f"    {ds:<12s}: {valid:>3d} / {total:>3d} cells with valid knee")

    # Knee cycle statistics for valid cells
    if valid_cells:
        knees = np.array([c['knee_cycle'] for c in valid_cells])
        print(f"\n  Knee-cycle statistics (valid cells):")
        print(f"    Min:    {int(knees.min()):>6d}")
        print(f"    Median: {int(np.median(knees)):>6d}")
        print(f"    Mean:   {int(knees.mean()):>6d}")
        print(f"    Max:    {int(knees.max()):>6d}")
        print(f"    Std:    {int(knees.std()):>6d}")
    print("=" * 70 + "\n")


# ======================================================================
#   CLI smoke test
# ======================================================================

if __name__ == '__main__':
    cells = load_all_cells_with_knees(include_severson=False)

    if cells:
        print("\nSample cells:")
        for c in cells[:5]:
            print(f"  {c['name']:<20s}  knee={c['knee_cycle']:>5d}  "
                  f"total={len(c['cycles']):>5d}  "
                  f"agreement={c['knee_agreement']:.2f}")

        train, cal, test = get_train_cal_test_split(cells)
        print(f"\nSplit sizes: train={len(train)}, cal={len(cal)}, test={len(test)}")
    else:
        print("\nNo valid cells found. Check data paths and knee-detection settings.")
