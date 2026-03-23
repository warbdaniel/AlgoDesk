"""
Project Alpha - ML Scalping Engine Configuration

Centralized configuration for the Alpha Engine data foundation.
Integrates with existing data-pipeline (port 5300) and fix-api (port 5200).
"""

from dataclasses import dataclass, field
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ENGINE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ENGINE_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "alpha"

# ---------------------------------------------------------------------------
# Default symbols (cTrader symbol IDs used by fix-api)
# ---------------------------------------------------------------------------
DEFAULT_SYMBOLS = ["1", "2", "3"]  # EURUSD, GBPUSD, USDJPY on IC Markets

SYMBOL_META = {
    "1": {"name": "EURUSD", "pip_size": 0.0001, "digits": 5},
    "2": {"name": "GBPUSD", "pip_size": 0.0001, "digits": 5},
    "3": {"name": "USDJPY", "pip_size": 0.01,   "digits": 3},
}

# ---------------------------------------------------------------------------
# Tick buffer defaults
# ---------------------------------------------------------------------------
TICK_BUFFER_SIZE = 50_000          # ticks per symbol in ring buffer
TICK_FLUSH_INTERVAL_SEC = 1.0      # micro-batch flush interval

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """Parameters for scalping feature computation."""

    # Tick-window sizes for microstructure features
    tick_windows: list[int] = field(default_factory=lambda: [10, 25, 50, 100, 250])

    # Time-window sizes in seconds for time-based features
    time_windows_sec: list[int] = field(default_factory=lambda: [5, 10, 30, 60, 120])

    # Spread EMA half-life in ticks
    spread_ema_halflife: int = 20

    # Order-flow imbalance lookback (ticks)
    ofi_lookback: int = 50

    # Microprice computation enabled
    microprice_enabled: bool = True

    # Volatility estimation lookbacks (ticks)
    volatility_lookbacks: list[int] = field(default_factory=lambda: [25, 50, 100])

    # Tick velocity lookbacks
    velocity_lookbacks: list[int] = field(default_factory=lambda: [5, 10, 25, 50])

    # Price level clustering (round-number detection)
    round_number_enabled: bool = True


# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
@dataclass
class LabelConfig:
    """Parameters for supervised ML label generation."""

    # Triple-barrier method
    tp_pips: float = 5.0             # take-profit distance in pips
    sl_pips: float = 5.0             # stop-loss distance in pips
    max_holding_sec: float = 300.0   # max holding period (5 min)

    # Forward return horizons (seconds)
    return_horizons_sec: list[float] = field(
        default_factory=lambda: [5.0, 10.0, 30.0, 60.0, 120.0]
    )

    # Classification thresholds (in pips) for ternary labels
    long_threshold_pips: float = 2.0
    short_threshold_pips: float = 2.0

    # Minimum ticks required in horizon window to compute label
    min_ticks_in_horizon: int = 3


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Parameters for ML dataset construction."""

    # Train / validation / test split ratios (by time)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Purge gap between train/val/test to prevent leakage (seconds)
    purge_gap_sec: float = 300.0

    # Feature normalization
    normalize: bool = True
    normalize_method: str = "zscore"   # "zscore" or "minmax"

    # Outlier clipping (standard deviations)
    clip_std: float = 5.0

    # Minimum samples required to build a dataset
    min_samples: int = 1000

    # Maximum samples per dataset (0 = unlimited)
    max_samples: int = 0

    # Random seed for reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class AlphaConfig:
    """Top-level configuration for Project Alpha."""

    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    tick_buffer_size: int = TICK_BUFFER_SIZE
    tick_flush_interval: float = TICK_FLUSH_INTERVAL_SEC

    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Integration endpoints
    data_pipeline_url: str = "http://127.0.0.1:5300"
    fix_api_url: str = "http://127.0.0.1:5200"

    # Storage
    db_path: str = str(DATA_DIR / "alpha.db")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AlphaConfig":
        """Load config from YAML file, merging with defaults."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = cls()
        for key, val in raw.items():
            if key == "features" and isinstance(val, dict):
                cfg.features = FeatureConfig(**val)
            elif key == "labels" and isinstance(val, dict):
                cfg.labels = LabelConfig(**val)
            elif key == "dataset" and isinstance(val, dict):
                cfg.dataset = DatasetConfig(**val)
            elif hasattr(cfg, key):
                setattr(cfg, key, val)
        return cfg
