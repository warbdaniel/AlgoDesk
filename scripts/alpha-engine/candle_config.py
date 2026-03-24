"""
Historical 5M Candle ML Pipeline - Configuration
=================================================

Centralised configuration for the historical candle-based ML feature
and labeling pipeline.  Operates on 5-minute OHLCV candles stored by
the data-pipeline (port 5300) MarketDataStore.

This is SEPARATE from the tick-level AlphaConfig used by the real-time
scalping engine.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ENGINE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ENGINE_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CANDLE_DATA_DIR = PROJECT_ROOT / "data" / "candle_ml"

# ---------------------------------------------------------------------------
# Symbol metadata (matches data-pipeline config.yaml)
# ---------------------------------------------------------------------------
CANDLE_SYMBOLS = {
    "EURUSD": {"pip_size": 0.0001, "digits": 5},
    "GBPUSD": {"pip_size": 0.0001, "digits": 5},
    "USDJPY": {"pip_size": 0.01,   "digits": 3},
    "USDCHF": {"pip_size": 0.0001, "digits": 5},
    "AUDUSD": {"pip_size": 0.0001, "digits": 5},
    "USDCAD": {"pip_size": 0.0001, "digits": 5},
    "NZDUSD": {"pip_size": 0.0001, "digits": 5},
    "EURGBP": {"pip_size": 0.0001, "digits": 5},
    "EURJPY": {"pip_size": 0.01,   "digits": 3},
    "GBPJPY": {"pip_size": 0.01,   "digits": 3},
    "CHFJPY": {"pip_size": 0.01,   "digits": 3},
    "CADJPY": {"pip_size": 0.01,   "digits": 3},
    "NZDJPY": {"pip_size": 0.01,   "digits": 3},
    "XAUUSD": {"pip_size": 0.1,    "digits": 2},
    "BTCUSD": {"pip_size": 1.0,    "digits": 2},
}

DEFAULT_CANDLE_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"]


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
@dataclass
class CandleFeatureConfig:
    """Parameters for 5M candle feature computation."""

    # SMA periods (in number of 5M candles)
    sma_periods: list[int] = field(default_factory=lambda: [10, 20, 50, 100])

    # EMA periods
    ema_periods: list[int] = field(default_factory=lambda: [8, 12, 21, 26, 50])

    # RSI period
    rsi_period: int = 14

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR period
    atr_period: int = 14

    # ADX period
    adx_period: int = 14

    # Stochastic parameters
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    # Return lookbacks (in candles)
    return_lookbacks: list[int] = field(default_factory=lambda: [1, 3, 6, 12, 24])

    # Volatility lookbacks (in candles)
    volatility_lookbacks: list[int] = field(default_factory=lambda: [6, 12, 24, 48])

    # Volume (tick count) MA period
    volume_ma_period: int = 20

    # Minimum candles needed for warmup (longest lookback + buffer)
    min_warmup: int = 100


# ---------------------------------------------------------------------------
# Label configuration
# ---------------------------------------------------------------------------
@dataclass
class CandleLabelConfig:
    """Parameters for candle-based ML label generation."""

    # Forward return horizons (in number of 5M candles)
    # 1 candle = 5min, 3 = 15min, 6 = 30min, 12 = 1h, 24 = 2h, 48 = 4h
    forward_horizons: list[int] = field(
        default_factory=lambda: [1, 3, 6, 12, 24]
    )

    # Classification thresholds (in pips) for ternary direction labels
    long_threshold_pips: float = 5.0
    short_threshold_pips: float = 5.0

    # Triple-barrier parameters (in pips and candles)
    barrier_tp_pips: float = 15.0
    barrier_sl_pips: float = 15.0
    barrier_max_candles: int = 24    # max holding = 24 * 5min = 2 hours

    # Maximum forward drawdown horizon (candles) for risk labels
    max_drawdown_horizon: int = 12


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
@dataclass
class CandleDatasetConfig:
    """Parameters for candle ML dataset construction."""

    # Train / validation / test split ratios (by time)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Purge gap between splits to prevent label leakage (in 5M candles)
    purge_gap_candles: int = 48   # 48 * 5min = 4 hours

    # Feature normalisation
    normalize: bool = True
    normalize_method: str = "zscore"   # "zscore" or "minmax"

    # Outlier clipping (standard deviations)
    clip_std: float = 5.0

    # Minimum samples required to build a dataset
    min_samples: int = 500

    # Maximum samples per dataset (0 = unlimited)
    max_samples: int = 0

    # Random seed for reproducibility
    seed: int = 42


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class CandleMLConfig:
    """Top-level configuration for the historical 5M candle ML pipeline."""

    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_CANDLE_SYMBOLS))
    interval: str = "5m"

    features: CandleFeatureConfig = field(default_factory=CandleFeatureConfig)
    labels: CandleLabelConfig = field(default_factory=CandleLabelConfig)
    dataset: CandleDatasetConfig = field(default_factory=CandleDatasetConfig)

    # Data source: path to data-pipeline's SQLite DB
    data_pipeline_db: str = "/opt/trading-desk/data/market_data.db"

    # Storage: path for candle ML artifacts
    db_path: str = str(CANDLE_DATA_DIR / "candle_ml.db")
