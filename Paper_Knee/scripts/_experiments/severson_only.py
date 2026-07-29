"""
Severson-only data loader helper that bypasses NASA/CALCE loading
to save ~5 minutes of startup time.

Pickles the result so subsequent runs in the same session are instant.
"""
import os
import sys
import pickle

_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "_severson_cache.pkl"
)


def load_severson_only():
    """Return Severson cells with knee points attached, cached to disk."""
    if os.path.exists(_CACHE):
        print(f"  Using cached Severson cells from {_CACHE}")
        with open(_CACHE, "rb") as f:
            return pickle.load(f)

    # Load directly via Paper 7 loader, then run knee detection
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util

    _p7_scripts = os.path.normpath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "scripts",
    ))

    # Load Paper 7 config + data_loader in isolated namespace
    saved_config = sys.modules.get("config")
    saved_path = list(sys.path)
    try:
        if _p7_scripts not in sys.path:
            sys.path.insert(0, _p7_scripts)

        cfg_path = os.path.join(_p7_scripts, "config.py")
        spec_cfg = importlib.util.spec_from_file_location("paper7_config", cfg_path)
        p7_config = importlib.util.module_from_spec(spec_cfg)
        spec_cfg.loader.exec_module(p7_config)
        sys.modules["config"] = p7_config

        dl_path = os.path.join(_p7_scripts, "data_loader.py")
        spec_dl = importlib.util.spec_from_file_location("paper7_data_loader", dl_path)
        p7_dl = importlib.util.module_from_spec(spec_dl)
        spec_dl.loader.exec_module(p7_dl)
    finally:
        if saved_config is not None:
            sys.modules["config"] = saved_config
        elif "config" in sys.modules:
            del sys.modules["config"]
        sys.path[:] = saved_path

    print("  Loading Severson (skipping NASA/CALCE)...")
    cells = p7_dl.load_severson_cells()
    print(f"  Loaded {len(cells)} raw Severson cells")

    # Annotate with knee points
    from knee_detection import detect_knee_ensemble, validate_knee_point
    from config import KNEE_MIN_CYCLE

    valid = []
    for cell in cells:
        if len(cell["capacity"]) < KNEE_MIN_CYCLE:
            continue
        knee, per_method, agreement = detect_knee_ensemble(
            cell["cycles"], cell["capacity"],
        )
        if knee is None:
            continue
        is_valid, diag = validate_knee_point(
            cell["cycles"], cell["capacity"], knee,
        )
        if not is_valid:
            continue
        cell["knee_cycle"] = knee
        cell["knee_details"] = per_method
        cell["knee_agreement"] = agreement
        cell["knee_diagnostics"] = diag
        valid.append(cell)

    print(f"  {len(valid)} valid Severson cells with knee points")

    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, "wb") as f:
        pickle.dump(valid, f)
    print(f"  Cached to {_CACHE}")

    return valid


if __name__ == "__main__":
    cells = load_severson_only()
    print(f"\nTotal: {len(cells)} cells")
