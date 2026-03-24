"""
Continuous Learning System - Configuration
==========================================

Centralised configuration for the CLS microservice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = CLS_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CLS_DATA_DIR = PROJECT_ROOT / "data" / "cls"

# Alpha-engine paths (for retraining integration)
ALPHA_ENGINE_DIR = SCRIPTS_DIR / "alpha-engine"
CANDLE_ML_DB = PROJECT_ROOT / "data" / "candle_ml" / "candle_ml.db"
MODELS_DIR = ALPHA_ENGINE_DIR / "models"


# ---------------------------------------------------------------------------
# Performance monitoring
# ---------------------------------------------------------------------------
@dataclass
class PerformanceConfig:
    """Thresholds for model performance monitoring."""

    # Minimum AUC before triggering a retrain alert
    min_auc: float = 0.52

    # Minimum accuracy before triggering alert
    min_accuracy: float = 0.52

    # Maximum log loss before triggering alert
    max_logloss: float = 0.72

    # Minimum win rate from live trades
    min_win_rate: float = 0.45

    # Minimum profit factor from live trades
    min_profit_factor: float = 0.8

    # Rolling window size for computing live metrics (number of predictions)
    rolling_window: int = 200

    # Minimum predictions before evaluating performance
    min_predictions: int = 50

    # How often to evaluate performance (seconds)
    eval_interval_seconds: int = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
@dataclass
class DriftConfig:
    """Configuration for feature and concept drift detection."""

    # Feature drift: PSI (Population Stability Index) threshold
    # PSI < 0.1 = no drift, 0.1-0.25 = moderate, > 0.25 = significant
    psi_threshold: float = 0.20

    # Feature drift: KS (Kolmogorov-Smirnov) test p-value threshold
    ks_p_value_threshold: float = 0.01

    # Concept drift: ADWIN window parameter (smaller = more sensitive)
    adwin_delta: float = 0.002

    # Number of bins for PSI calculation
    psi_bins: int = 10

    # Reference window size (number of samples from training distribution)
    reference_window: int = 1000

    # Detection window size (recent production samples)
    detection_window: int = 200

    # How often to run drift detection (seconds)
    check_interval_seconds: int = 600  # 10 minutes

    # Minimum samples before running drift detection
    min_samples: int = 100

    # Features to monitor (empty = all features)
    monitored_features: list[str] = field(default_factory=list)

    # Maximum number of features to track for drift (top by importance)
    max_tracked_features: int = 20


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass
class RegistryConfig:
    """Configuration for the model registry."""

    # Directory for model artifacts
    models_dir: str = str(MODELS_DIR)

    # Maximum models to keep per symbol (oldest pruned)
    max_versions_per_symbol: int = 10

    # Champion promotion: challenger must beat champion by this margin
    promotion_auc_margin: float = 0.005
    promotion_accuracy_margin: float = 0.005

    # Minimum evaluation period before a challenger can be promoted (seconds)
    min_challenger_eval_period: int = 3600  # 1 hour

    # Minimum predictions on challenger before promotion decision
    min_challenger_predictions: int = 100


# ---------------------------------------------------------------------------
# Retrain orchestrator
# ---------------------------------------------------------------------------
@dataclass
class RetrainConfig:
    """Configuration for automated retraining."""

    # Cooldown between retrains for the same symbol (seconds)
    retrain_cooldown_seconds: int = 14400  # 4 hours

    # Maximum concurrent retrains
    max_concurrent_retrains: int = 2

    # Optuna trials for retraining (lower than full training for speed)
    retrain_n_trials: int = 30

    # Path to candle ML database
    candle_ml_db: str = str(CANDLE_ML_DB)

    # Auto-promote if challenger beats champion
    auto_promote: bool = False

    # Triggers: which conditions trigger a retrain
    trigger_on_drift: bool = True
    trigger_on_performance_degradation: bool = True
    trigger_on_regime_change: bool = True

    # Schedule: periodic retraining (0 = disabled, value in seconds)
    periodic_retrain_interval: int = 86400  # daily


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------
@dataclass
class FeedbackConfig:
    """Configuration for the trade outcome feedback loop."""

    # Dashboard API URL for fetching trade outcomes
    dashboard_url: str = "http://localhost:5100"

    # Data pipeline API URL for feature data
    data_pipeline_url: str = "http://localhost:5300"

    # Regime detector API URL
    regime_detector_url: str = "http://localhost:5000"

    # How often to poll for new trade outcomes (seconds)
    poll_interval_seconds: int = 60

    # Batch size for processing feedback
    batch_size: int = 50


# ---------------------------------------------------------------------------
# Service URLs (for integration)
# ---------------------------------------------------------------------------
@dataclass
class ServiceURLs:
    """URLs for other AlgoDesk microservices."""

    data_pipeline: str = "http://localhost:5300"
    regime_detector: str = "http://localhost:5000"
    fix_api: str = "http://localhost:5200"
    dashboard: str = "http://localhost:5100"
    claude_api: str = "http://localhost:5400"


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------
@dataclass
class CLSConfig:
    """Top-level CLS configuration."""

    # Service
    host: str = "0.0.0.0"
    port: int = 5500
    log_level: str = "INFO"

    # Database
    db_path: str = str(CLS_DATA_DIR / "cls.db")

    # Symbols to monitor (empty = auto-discover from model registry)
    symbols: list[str] = field(default_factory=list)

    # Sub-configs
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    retrain: RetrainConfig = field(default_factory=RetrainConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    services: ServiceURLs = field(default_factory=ServiceURLs)
