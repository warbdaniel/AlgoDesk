"""
Historical 5M Candle ML Pipeline - Dataset Builder
===================================================

Assembles CandleFeatureVector + CandleLabel pairs into train/val/test
datasets with proper temporal splitting, purge gaps, normalisation,
and outlier clipping.

Key design principles:
  - Strictly temporal splits (no random shuffle)
  - Purge gaps between splits (in candle count) to prevent label leakage
  - Feature normalisation fitted ONLY on training set
  - Outlier clipping before normalisation
  - Reproducible via seed
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from candle_config import CandleDatasetConfig
from candle_features import CandleFeatureVector
from candle_labels import CandleLabel


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
@dataclass
class CandleNormStats:
    """Per-feature normalisation statistics fitted on training data."""

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)
    method: str = "zscore"

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "means": self.means,
            "stds": self.stds,
            "mins": self.mins,
            "maxs": self.maxs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CandleNormStats:
        return cls(
            method=d.get("method", "zscore"),
            means=d.get("means", {}),
            stds=d.get("stds", {}),
            mins=d.get("mins", {}),
            maxs=d.get("maxs", {}),
        )


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CandleSample:
    """Single aligned (feature, label) observation from a 5M candle."""

    ts: float
    symbol: str
    features: dict[str, float]
    labels: dict[str, float]


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
@dataclass
class CandleDatasetSplit:
    """One split (train / val / test) of the candle dataset."""

    name: str
    samples: list[CandleSample] = field(default_factory=list)
    start_ts: float = 0.0
    end_ts: float = 0.0

    @property
    def size(self) -> int:
        return len(self.samples)

    def feature_matrix(self) -> list[list[float]]:
        """Return N x D feature matrix."""
        if not self.samples:
            return []
        keys = sorted(self.samples[0].features.keys())
        return [[s.features[k] for k in keys] for s in self.samples]

    def label_vector(self, label_name: str) -> list[float]:
        """Return values for a single label column."""
        return [s.labels.get(label_name, 0.0) for s in self.samples]

    def feature_names(self) -> list[str]:
        if not self.samples:
            return []
        return sorted(self.samples[0].features.keys())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
        }


# ---------------------------------------------------------------------------
# Complete dataset
# ---------------------------------------------------------------------------
@dataclass
class CandleDataset:
    """Complete candle ML dataset with all splits and metadata."""

    symbol: str = ""
    interval: str = "5m"
    created_at: float = 0.0
    config: CandleDatasetConfig = field(default_factory=CandleDatasetConfig)
    norm_stats: CandleNormStats = field(default_factory=CandleNormStats)

    train: CandleDatasetSplit = field(
        default_factory=lambda: CandleDatasetSplit(name="train")
    )
    val: CandleDatasetSplit = field(
        default_factory=lambda: CandleDatasetSplit(name="val")
    )
    test: CandleDatasetSplit = field(
        default_factory=lambda: CandleDatasetSplit(name="test")
    )

    total_raw_samples: int = 0
    total_valid_samples: int = 0

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "created_at": self.created_at,
            "total_raw": self.total_raw_samples,
            "total_valid": self.total_valid_samples,
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "norm_method": self.norm_stats.method,
            "feature_count": (
                len(self.train.feature_names()) if self.train.samples else 0
            ),
        }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
class CandleDatasetBuilder:
    """Builds ML-ready datasets from aligned candle features and labels."""

    def __init__(self, config: CandleDatasetConfig | None = None):
        self._cfg = config or CandleDatasetConfig()

    def build(
        self,
        features: list[CandleFeatureVector],
        labels: list[CandleLabel],
        symbol: str = "",
        interval: str = "5m",
    ) -> CandleDataset:
        """Build a complete dataset from candle features and labels.

        Features and labels are matched by timestamp (ts field).
        Both lists must be sorted by time ascending.
        """
        ds = CandleDataset(
            symbol=symbol,
            interval=interval,
            created_at=time.time(),
            config=self._cfg,
        )

        # 1. Align features and labels by timestamp
        samples = self._align(features, labels)
        ds.total_raw_samples = len(samples)

        # 2. Filter out samples with NaN labels
        samples = self._filter_valid(samples)
        ds.total_valid_samples = len(samples)

        if len(samples) < self._cfg.min_samples:
            return ds

        # 3. Cap at max_samples if configured
        if self._cfg.max_samples > 0 and len(samples) > self._cfg.max_samples:
            samples = samples[-self._cfg.max_samples:]

        # 4. Temporal split with purge gaps
        train, val, test = self._temporal_split(samples)

        # 5. Clip outliers (fit on train)
        if self._cfg.clip_std > 0:
            clip_bounds = self._compute_clip_bounds(train)
            train = self._clip_features(train, clip_bounds)
            val = self._clip_features(val, clip_bounds)
            test = self._clip_features(test, clip_bounds)

        # 6. Normalise (fit on train, apply to all)
        if self._cfg.normalize:
            norm_stats = self._fit_normalisation(train)
            train = self._apply_normalisation(train, norm_stats)
            val = self._apply_normalisation(val, norm_stats)
            test = self._apply_normalisation(test, norm_stats)
            ds.norm_stats = norm_stats

        # 7. Package into splits
        ds.train = self._make_split("train", train)
        ds.val = self._make_split("val", val)
        ds.test = self._make_split("test", test)

        return ds

    # ── alignment ─────────────────────────────────────────────

    def _align(
        self,
        features: list[CandleFeatureVector],
        labels: list[CandleLabel],
    ) -> list[CandleSample]:
        """Match features to labels by exact timestamp.

        Since both come from the same candle list, timestamps should
        match exactly. Uses merge-join with 1-second tolerance as safety.
        """
        samples: list[CandleSample] = []
        j = 0
        tolerance = 1.0  # seconds

        for fv in features:
            while j < len(labels) - 1 and labels[j].ts < fv.ts - tolerance:
                j += 1

            if j >= len(labels):
                break

            lbl = labels[j]
            if abs(lbl.ts - fv.ts) <= tolerance:
                samples.append(CandleSample(
                    ts=fv.ts,
                    symbol=fv.symbol,
                    features=fv.to_dict(),
                    labels=lbl.to_dict(),
                ))

        return samples

    def _filter_valid(self, samples: list[CandleSample]) -> list[CandleSample]:
        """Remove samples with NaN in any label value."""
        valid = []
        for s in samples:
            has_nan = False
            for val in s.labels.values():
                if isinstance(val, float) and math.isnan(val):
                    has_nan = True
                    break
            if not has_nan:
                valid.append(s)
        return valid

    # ── temporal split ────────────────────────────────────────

    def _temporal_split(
        self,
        samples: list[CandleSample],
    ) -> tuple[list[CandleSample], list[CandleSample], list[CandleSample]]:
        """Split samples temporally with purge gaps between splits."""
        n = len(samples)
        train_end = int(n * self._cfg.train_ratio)
        val_end = train_end + int(n * self._cfg.val_ratio)
        purge = self._cfg.purge_gap_candles

        # Train: [0, train_end)
        train = samples[:train_end]

        # Purge between train and val
        val_start = min(train_end + purge, n)
        val_end_adj = min(val_end, n)
        val = samples[val_start:val_end_adj]

        # Purge between val and test
        test_start = min(val_end_adj + purge, n)
        test = samples[test_start:]

        return train, val, test

    # ── normalisation ─────────────────────────────────────────

    def _compute_clip_bounds(
        self, samples: list[CandleSample],
    ) -> dict[str, tuple[float, float]]:
        if not samples:
            return {}
        keys = sorted(samples[0].features.keys())
        bounds: dict[str, tuple[float, float]] = {}
        for k in keys:
            vals = [s.features[k] for s in samples]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var) if var > 0 else 1.0
            bounds[k] = (mean - self._cfg.clip_std * std,
                         mean + self._cfg.clip_std * std)
        return bounds

    def _clip_features(
        self,
        samples: list[CandleSample],
        bounds: dict[str, tuple[float, float]],
    ) -> list[CandleSample]:
        for s in samples:
            for k, (lo, hi) in bounds.items():
                if k in s.features:
                    s.features[k] = max(lo, min(hi, s.features[k]))
        return samples

    def _fit_normalisation(
        self, samples: list[CandleSample],
    ) -> CandleNormStats:
        if not samples:
            return CandleNormStats(method=self._cfg.normalize_method)

        keys = sorted(samples[0].features.keys())
        stats = CandleNormStats(method=self._cfg.normalize_method)

        for k in keys:
            vals = [s.features[k] for s in samples]
            n = len(vals)
            mean = sum(vals) / n
            var = sum((v - mean) ** 2 for v in vals) / n
            std = math.sqrt(var) if var > 0 else 1.0
            stats.means[k] = mean
            stats.stds[k] = std
            stats.mins[k] = min(vals)
            stats.maxs[k] = max(vals)

        return stats

    def _apply_normalisation(
        self,
        samples: list[CandleSample],
        stats: CandleNormStats,
    ) -> list[CandleSample]:
        for s in samples:
            for k in list(s.features.keys()):
                val = s.features[k]
                if stats.method == "zscore":
                    mean = stats.means.get(k, 0.0)
                    std = stats.stds.get(k, 1.0)
                    s.features[k] = (val - mean) / std if std > 0 else 0.0
                elif stats.method == "minmax":
                    lo = stats.mins.get(k, 0.0)
                    hi = stats.maxs.get(k, 1.0)
                    rng = hi - lo
                    s.features[k] = (val - lo) / rng if rng > 0 else 0.0
        return samples

    # ── helpers ───────────────────────────────────────────────

    def _make_split(
        self, name: str, samples: list[CandleSample],
    ) -> CandleDatasetSplit:
        split = CandleDatasetSplit(name=name, samples=samples)
        if samples:
            split.start_ts = samples[0].ts
            split.end_ts = samples[-1].ts
        return split
