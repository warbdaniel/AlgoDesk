"""
Project Alpha - ML Dataset Builder

Assembles feature vectors + labels into train/validation/test datasets
with proper temporal splitting, purge gaps, normalisation, and export.

Key design principles:
  - Strictly temporal splits (no random shuffle) to prevent look-ahead bias
  - Purge gaps between splits to avoid label leakage across boundaries
  - Feature normalisation fitted ONLY on the training set
  - Outlier clipping before normalisation
  - Reproducible via seed-controlled operations
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from config import DatasetConfig
from scalping_features import ScalpingFeatureVector
from label_engine import ScalpingLabel


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------
@dataclass
class NormStats:
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
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(
            method=d.get("method", "zscore"),
            means=d.get("means", {}),
            stds=d.get("stds", {}),
            mins=d.get("mins", {}),
            maxs=d.get("maxs", {}),
        )


# ---------------------------------------------------------------------------
# Dataset sample
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Sample:
    """Single aligned (feature, label) observation."""

    ts: float
    symbol: str
    features: dict[str, float]
    labels: dict[str, float]


# ---------------------------------------------------------------------------
# Split result
# ---------------------------------------------------------------------------
@dataclass
class DatasetSplit:
    """One split (train / val / test) of the dataset."""

    name: str                                   # "train", "val", "test"
    samples: list[Sample] = field(default_factory=list)
    start_ts: float = 0.0
    end_ts: float = 0.0

    @property
    def size(self) -> int:
        return len(self.samples)

    def feature_matrix(self) -> list[list[float]]:
        """Return N x D feature matrix (list of lists)."""
        if not self.samples:
            return []
        keys = sorted(self.samples[0].features.keys())
        return [[s.features[k] for k in keys] for s in self.samples]

    def label_vector(self, label_name: str) -> list[float]:
        """Return label values for a single label column."""
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
# Built dataset
# ---------------------------------------------------------------------------
@dataclass
class AlphaDataset:
    """Complete dataset with all splits and metadata."""

    symbol: str = ""
    created_at: float = 0.0
    config: DatasetConfig = field(default_factory=DatasetConfig)
    norm_stats: NormStats = field(default_factory=NormStats)

    train: DatasetSplit = field(default_factory=lambda: DatasetSplit(name="train"))
    val: DatasetSplit = field(default_factory=lambda: DatasetSplit(name="val"))
    test: DatasetSplit = field(default_factory=lambda: DatasetSplit(name="test"))

    total_raw_samples: int = 0
    total_valid_samples: int = 0

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "created_at": self.created_at,
            "total_raw": self.total_raw_samples,
            "total_valid": self.total_valid_samples,
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "norm_method": self.norm_stats.method,
            "feature_count": len(self.train.feature_names()) if self.train.samples else 0,
        }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
class DatasetBuilder:
    """Builds ML-ready datasets from aligned features and labels."""

    def __init__(self, config: DatasetConfig | None = None):
        self._cfg = config or DatasetConfig()

    def build(
        self,
        features: list[ScalpingFeatureVector],
        labels: list[ScalpingLabel],
        symbol: str = "",
    ) -> AlphaDataset:
        """Build a complete dataset from time-aligned features and labels.

        Features and labels are matched by timestamp. Both lists must
        be sorted by time ascending.
        """
        ds = AlphaDataset(
            symbol=symbol,
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
            return ds  # return empty dataset with metadata

        # 3. Temporal split with purge gaps
        train, val, test = self._temporal_split(samples)

        # 4. Clip outliers (fit on train)
        if self._cfg.clip_std > 0:
            clip_bounds = self._compute_clip_bounds(train)
            train = self._clip_features(train, clip_bounds)
            val = self._clip_features(val, clip_bounds)
            test = self._clip_features(test, clip_bounds)

        # 5. Normalise (fit on train, apply to all)
        if self._cfg.normalize:
            norm_stats = self._fit_normalisation(train)
            train = self._apply_normalisation(train, norm_stats)
            val = self._apply_normalisation(val, norm_stats)
            test = self._apply_normalisation(test, norm_stats)
            ds.norm_stats = norm_stats

        # 6. Package into splits
        ds.train = self._make_split("train", train)
        ds.val = self._make_split("val", val)
        ds.test = self._make_split("test", test)

        return ds

    # ---- alignment --------------------------------------------------------

    def _align(
        self,
        features: list[ScalpingFeatureVector],
        labels: list[ScalpingLabel],
    ) -> list[Sample]:
        """Match features to labels by closest timestamp.

        Uses a merge-join on sorted timestamps with 0.5s tolerance.
        """
        samples: list[Sample] = []
        j = 0
        tolerance = 0.5  # seconds

        for fv in features:
            # Advance label pointer to closest match
            while j < len(labels) - 1 and labels[j].entry_ts < fv.ts - tolerance:
                j += 1

            if j >= len(labels):
                break

            lbl = labels[j]
            if abs(lbl.entry_ts - fv.ts) <= tolerance:
                samples.append(Sample(
                    ts=fv.ts,
                    symbol=fv.symbol,
                    features=fv.to_dict(),
                    labels=lbl.to_dict(),
                ))
        return samples

    def _filter_valid(self, samples: list[Sample]) -> list[Sample]:
        """Remove samples with NaN in any forward-return label."""
        valid = []
        for s in samples:
            has_nan = False
            for key, val in s.labels.items():
                if isinstance(val, float) and math.isnan(val):
                    has_nan = True
                    break
            if not has_nan:
                valid.append(s)
        return valid

    # ---- temporal split ---------------------------------------------------

    def _temporal_split(
        self, samples: list[Sample],
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        n = len(samples)
        train_end = int(n * self._cfg.train_ratio)
        val_end = train_end + int(n * self._cfg.val_ratio)

        # Compute purge boundaries in ticks based on purge_gap_sec
        purge_gap = self._cfg.purge_gap_sec

        # Train: [0, train_end)
        train = samples[:train_end]

        # Purge between train and val
        val_start = train_end
        if train and val_start < n:
            cutoff_ts = samples[train_end - 1].ts + purge_gap
            while val_start < n and samples[val_start].ts < cutoff_ts:
                val_start += 1

        # Val: [val_start, val_end_adj)
        val_end_adj = min(val_end, n)
        val = samples[val_start:val_end_adj]

        # Purge between val and test
        test_start = val_end_adj
        if val and test_start < n:
            cutoff_ts = samples[val_end_adj - 1].ts + purge_gap
            while test_start < n and samples[test_start].ts < cutoff_ts:
                test_start += 1

        test = samples[test_start:]

        return train, val, test

    # ---- normalisation ----------------------------------------------------

    def _compute_clip_bounds(
        self, samples: list[Sample],
    ) -> dict[str, tuple[float, float]]:
        """Compute clip bounds (mean ± clip_std * std) from training data."""
        if not samples:
            return {}
        keys = sorted(samples[0].features.keys())
        bounds: dict[str, tuple[float, float]] = {}
        for k in keys:
            vals = [s.features[k] for s in samples]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var) if var > 0 else 1.0
            lo = mean - self._cfg.clip_std * std
            hi = mean + self._cfg.clip_std * std
            bounds[k] = (lo, hi)
        return bounds

    def _clip_features(
        self, samples: list[Sample], bounds: dict[str, tuple[float, float]],
    ) -> list[Sample]:
        for s in samples:
            for k, (lo, hi) in bounds.items():
                if k in s.features:
                    s.features[k] = max(lo, min(hi, s.features[k]))
        return samples

    def _fit_normalisation(self, samples: list[Sample]) -> NormStats:
        """Compute normalisation statistics from training samples."""
        if not samples:
            return NormStats(method=self._cfg.normalize_method)

        keys = sorted(samples[0].features.keys())
        stats = NormStats(method=self._cfg.normalize_method)

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
        self, samples: list[Sample], stats: NormStats,
    ) -> list[Sample]:
        """Apply normalisation in-place."""
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

    # ---- helpers ----------------------------------------------------------

    def _make_split(self, name: str, samples: list[Sample]) -> DatasetSplit:
        split = DatasetSplit(name=name, samples=samples)
        if samples:
            split.start_ts = samples[0].ts
            split.end_ts = samples[-1].ts
        return split
