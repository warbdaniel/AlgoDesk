"""
Project Alpha - ML Scalping Engine
===================================

Data foundation for a tick-level ML scalping system.  Integrates with
AlgoDesk's existing data-pipeline (port 5300) and FIX API (port 5200)
to consume live tick data, compute microstructure features, generate
supervised labels, and build normalised train/val/test datasets.

Modules:
    config             – Centralised configuration (AlphaConfig)
    tick_buffer        – High-performance ring buffer (TickBufferManager)
    scalping_features  – Microstructure feature engine (ScalpingFeatureEngine)
    label_engine       – ML label generation (LabelEngine)
    dataset_builder    – Dataset assembly with temporal splits (DatasetBuilder)
    alpha_store        – SQLite persistence (AlphaStore)

Quick start:

    from alpha_engine import (
        AlphaConfig, TickBufferManager, ScalpingFeatureEngine,
        LabelEngine, DatasetBuilder, AlphaStore,
    )

    cfg = AlphaConfig()
    buf = TickBufferManager(cfg.tick_buffer_size)
    features = ScalpingFeatureEngine(cfg.features)
    labeller = LabelEngine(cfg.labels)
    builder  = DatasetBuilder(cfg.dataset)
    store    = AlphaStore(cfg.db_path)
"""

from config import (
    AlphaConfig,
    FeatureConfig,
    LabelConfig,
    DatasetConfig,
    SYMBOL_META,
    DEFAULT_SYMBOLS,
)
from tick_buffer import AlphaTick, TickRing, TickBufferManager
from scalping_features import ScalpingFeatureVector, ScalpingFeatureEngine
from label_engine import ScalpingLabel, LabelEngine, BarrierHit, Direction
from dataset_builder import (
    DatasetBuilder,
    AlphaDataset,
    DatasetSplit,
    Sample,
    NormStats,
)
from alpha_store import AlphaStore

__all__ = [
    # Config
    "AlphaConfig",
    "FeatureConfig",
    "LabelConfig",
    "DatasetConfig",
    "SYMBOL_META",
    "DEFAULT_SYMBOLS",
    # Tick buffer
    "AlphaTick",
    "TickRing",
    "TickBufferManager",
    # Features
    "ScalpingFeatureVector",
    "ScalpingFeatureEngine",
    # Labels
    "ScalpingLabel",
    "LabelEngine",
    "BarrierHit",
    "Direction",
    # Dataset
    "DatasetBuilder",
    "AlphaDataset",
    "DatasetSplit",
    "Sample",
    "NormStats",
    # Storage
    "AlphaStore",
]
