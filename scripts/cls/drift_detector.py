"""
Continuous Learning System - Drift Detector
============================================

Detects feature drift (input distribution changes) and concept drift
(relationship between features and target changes) using:

- **Feature drift:** Population Stability Index (PSI) and
  Kolmogorov-Smirnov (KS) tests.
- **Concept drift:** ADWIN-inspired adaptive windowing on prediction
  error rates.

When drift is detected, the system flags the model for retraining.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from cls_store import CLSStore
from config import DriftConfig

logger = logging.getLogger("cls.drift")


@dataclass
class DriftResult:
    """Result of a drift detection check."""

    symbol: str
    model_id: int
    drift_type: str  # "feature" or "concept"
    drift_detected: bool
    severity: str  # "none", "low", "moderate", "high", "critical"
    details: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "model_id": self.model_id,
            "drift_type": self.drift_type,
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "details": self.details,
            "ts": self.ts,
        }


class DriftDetector:
    """Detects feature and concept drift in model inputs/outputs."""

    def __init__(self, store: CLSStore, config: DriftConfig | None = None):
        self._store = store
        self._cfg = config or DriftConfig()
        self._last_check: dict[str, float] = {}  # symbol -> last check ts

        # Reference distributions per symbol (set from training data)
        self._reference_distributions: dict[str, dict[str, np.ndarray]] = {}

        # ADWIN state for concept drift per symbol
        self._error_windows: dict[str, list[float]] = {}

    # ── Reference distribution management ────────────────────────

    def set_reference_distribution(
        self,
        symbol: str,
        feature_distributions: dict[str, np.ndarray],
    ):
        """Set the reference (training) distribution for a symbol.

        feature_distributions: {feature_name: array of training values}
        """
        self._reference_distributions[symbol] = feature_distributions
        logger.info("Set reference distribution for %s (%d features)",
                     symbol, len(feature_distributions))

    def build_reference_from_predictions(
        self, symbol: str, model_id: int,
    ) -> bool:
        """Build reference distribution from the earliest predictions.

        Uses the first `reference_window` predictions' stored features.
        """
        predictions = self._store.get_recent_predictions(
            symbol=symbol,
            model_id=model_id,
            limit=self._cfg.reference_window,
        )

        if len(predictions) < self._cfg.min_samples:
            logger.warning(
                "Not enough predictions for reference (%d < %d)",
                len(predictions), self._cfg.min_samples,
            )
            return False

        # Sort chronologically (oldest first) and take first N
        predictions.sort(key=lambda p: p["ts"])
        predictions = predictions[:self._cfg.reference_window]

        ref_distributions: dict[str, list[float]] = {}
        for pred in predictions:
            features = json.loads(pred["features"]) if isinstance(pred["features"], str) else pred["features"]
            if not features:
                continue
            for feat_name, val in features.items():
                if feat_name not in ref_distributions:
                    ref_distributions[feat_name] = []
                try:
                    ref_distributions[feat_name].append(float(val))
                except (TypeError, ValueError):
                    pass

        self._reference_distributions[symbol] = {
            k: np.array(v) for k, v in ref_distributions.items() if len(v) > 10
        }
        logger.info("Built reference distribution for %s from %d predictions (%d features)",
                     symbol, len(predictions), len(self._reference_distributions[symbol]))
        return True

    # ── Feature drift detection ──────────────────────────────────

    def detect_feature_drift(
        self, symbol: str, model_id: int,
    ) -> DriftResult:
        """Detect feature distribution drift using PSI and KS tests.

        Compares recent production features against the reference
        (training) distribution.
        """
        ref = self._reference_distributions.get(symbol, {})
        if not ref:
            return DriftResult(
                symbol=symbol, model_id=model_id,
                drift_type="feature", drift_detected=False,
                severity="none",
                details={"error": "no_reference_distribution"},
            )

        # Get recent predictions with features
        predictions = self._store.get_recent_predictions(
            symbol=symbol,
            model_id=model_id,
            limit=self._cfg.detection_window,
        )

        if len(predictions) < self._cfg.min_samples:
            return DriftResult(
                symbol=symbol, model_id=model_id,
                drift_type="feature", drift_detected=False,
                severity="none",
                details={"error": "insufficient_samples", "count": len(predictions)},
            )

        # Extract current feature distributions
        current: dict[str, list[float]] = {}
        for pred in predictions:
            features = json.loads(pred["features"]) if isinstance(pred["features"], str) else pred["features"]
            if not features:
                continue
            for feat_name, val in features.items():
                if feat_name not in current:
                    current[feat_name] = []
                try:
                    current[feat_name].append(float(val))
                except (TypeError, ValueError):
                    pass

        # Select features to check
        features_to_check = self._cfg.monitored_features or list(ref.keys())
        if self._cfg.max_tracked_features > 0:
            features_to_check = features_to_check[:self._cfg.max_tracked_features]

        # Compute PSI for each feature
        psi_results: dict[str, float] = {}
        ks_results: dict[str, dict] = {}
        drifted_features: list[str] = []

        for feat_name in features_to_check:
            if feat_name not in ref or feat_name not in current:
                continue
            ref_vals = ref[feat_name]
            cur_vals = np.array(current[feat_name])

            if len(cur_vals) < 10:
                continue

            # PSI
            psi = self._compute_psi(ref_vals, cur_vals)
            psi_results[feat_name] = round(psi, 4)

            # KS test
            ks_stat, ks_p = self._ks_test(ref_vals, cur_vals)
            ks_results[feat_name] = {
                "statistic": round(ks_stat, 4),
                "p_value": round(ks_p, 6),
            }

            if psi > self._cfg.psi_threshold or ks_p < self._cfg.ks_p_value_threshold:
                drifted_features.append(feat_name)

        # Determine overall drift severity
        drift_ratio = len(drifted_features) / max(len(features_to_check), 1)
        max_psi = max(psi_results.values()) if psi_results else 0

        drift_detected = len(drifted_features) > 0
        severity = self._classify_drift_severity(drift_ratio, max_psi)

        result = DriftResult(
            symbol=symbol,
            model_id=model_id,
            drift_type="feature",
            drift_detected=drift_detected,
            severity=severity,
            details={
                "drifted_features": drifted_features,
                "drift_ratio": round(drift_ratio, 4),
                "max_psi": round(max_psi, 4),
                "features_checked": len(features_to_check),
                "samples_used": len(predictions),
                "top_psi": dict(sorted(
                    psi_results.items(), key=lambda x: x[1], reverse=True,
                )[:10]),
            },
        )

        # Persist
        self._store.save_drift_snapshot(
            symbol, model_id, "feature",
            drift_detected, severity, result.details,
        )

        if drift_detected:
            logger.warning(
                "Feature drift detected for %s: %d/%d features drifted (max PSI=%.3f, severity=%s)",
                symbol, len(drifted_features), len(features_to_check), max_psi, severity,
            )
        else:
            logger.info("No feature drift for %s (max PSI=%.3f)", symbol, max_psi)

        return result

    # ── Concept drift detection ──────────────────────────────────

    def detect_concept_drift(
        self, symbol: str, model_id: int,
    ) -> DriftResult:
        """Detect concept drift using ADWIN-inspired error rate monitoring.

        Tracks the prediction error rate over time and detects significant
        changes using an adaptive windowing approach.
        """
        predictions = self._store.get_recent_predictions(
            symbol=symbol,
            model_id=model_id,
            limit=self._cfg.reference_window + self._cfg.detection_window,
            resolved_only=True,
        )

        if len(predictions) < self._cfg.min_samples * 2:
            return DriftResult(
                symbol=symbol, model_id=model_id,
                drift_type="concept", drift_detected=False,
                severity="none",
                details={"error": "insufficient_resolved_predictions", "count": len(predictions)},
            )

        # Sort chronologically
        predictions.sort(key=lambda p: p["ts"])

        # Compute error sequence (1 = incorrect, 0 = correct)
        errors = []
        for pred in predictions:
            predicted = 1 if pred["prediction"] >= 0.5 else 0
            actual = 1 if pred["actual_outcome"] >= 0.5 else 0
            errors.append(1 if predicted != actual else 0)

        errors = np.array(errors, dtype=np.float64)

        # Split into reference (first half) and detection (second half) windows
        split = len(errors) // 2
        ref_errors = errors[:split]
        det_errors = errors[split:]

        ref_error_rate = float(ref_errors.mean())
        det_error_rate = float(det_errors.mean())
        error_rate_change = det_error_rate - ref_error_rate

        # Statistical test: is the error rate change significant?
        drift_detected, p_value = self._adwin_test(ref_errors, det_errors)

        severity = "none"
        if drift_detected:
            if abs(error_rate_change) > 0.15:
                severity = "critical"
            elif abs(error_rate_change) > 0.10:
                severity = "high"
            elif abs(error_rate_change) > 0.05:
                severity = "moderate"
            else:
                severity = "low"

        result = DriftResult(
            symbol=symbol,
            model_id=model_id,
            drift_type="concept",
            drift_detected=drift_detected,
            severity=severity,
            details={
                "ref_error_rate": round(ref_error_rate, 4),
                "det_error_rate": round(det_error_rate, 4),
                "error_rate_change": round(error_rate_change, 4),
                "p_value": round(p_value, 6),
                "ref_window_size": len(ref_errors),
                "det_window_size": len(det_errors),
                "total_predictions": len(predictions),
            },
        )

        # Persist
        self._store.save_drift_snapshot(
            symbol, model_id, "concept",
            drift_detected, severity, result.details,
        )

        if drift_detected:
            logger.warning(
                "Concept drift detected for %s: error rate %.3f -> %.3f (change=%.3f, severity=%s)",
                symbol, ref_error_rate, det_error_rate, error_rate_change, severity,
            )
        else:
            logger.info(
                "No concept drift for %s (error rate %.3f -> %.3f)",
                symbol, ref_error_rate, det_error_rate,
            )

        return result

    # ── Combined check ───────────────────────────────────────────

    def check_all_drift(self, symbol: str, model_id: int) -> dict:
        """Run both feature and concept drift detection."""
        feature_result = self.detect_feature_drift(symbol, model_id)
        concept_result = self.detect_concept_drift(symbol, model_id)

        any_drift = feature_result.drift_detected or concept_result.drift_detected
        max_severity = max(
            self._severity_rank(feature_result.severity),
            self._severity_rank(concept_result.severity),
        )
        severity_names = ["none", "low", "moderate", "high", "critical"]

        self._last_check[symbol] = time.time()

        return {
            "symbol": symbol,
            "model_id": model_id,
            "any_drift_detected": any_drift,
            "overall_severity": severity_names[max_severity],
            "feature_drift": feature_result.to_dict(),
            "concept_drift": concept_result.to_dict(),
        }

    def should_check(self, symbol: str) -> bool:
        """Check if enough time has passed since last drift check."""
        last = self._last_check.get(symbol, 0)
        return (time.time() - last) >= self._cfg.check_interval_seconds

    # ── Statistical methods ──────────────────────────────────────

    def _compute_psi(
        self, reference: np.ndarray, current: np.ndarray,
    ) -> float:
        """Compute Population Stability Index (PSI).

        PSI < 0.1: no significant change
        PSI 0.1-0.25: moderate change
        PSI > 0.25: significant change
        """
        n_bins = self._cfg.psi_bins

        # Use reference quantiles as bin edges for stability
        edges = np.percentile(
            reference,
            np.linspace(0, 100, n_bins + 1),
        )
        # Ensure unique edges
        edges = np.unique(edges)
        if len(edges) < 3:
            return 0.0

        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)

        # Convert to proportions with smoothing
        eps = 1e-4
        ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
        cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * len(cur_counts))

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(psi, 0.0)

    def _ks_test(
        self, reference: np.ndarray, current: np.ndarray,
    ) -> tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test (no scipy dependency).

        Returns (ks_statistic, approximate_p_value).
        """
        n1 = len(reference)
        n2 = len(current)

        all_values = np.concatenate([reference, current])
        all_values.sort()

        cdf1 = np.searchsorted(np.sort(reference), all_values, side="right") / n1
        cdf2 = np.searchsorted(np.sort(current), all_values, side="right") / n2

        ks_stat = float(np.max(np.abs(cdf1 - cdf2)))

        # Approximate p-value using asymptotic formula
        n_eff = (n1 * n2) / (n1 + n2)
        lambda_val = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * ks_stat

        # Kolmogorov distribution approximation
        if lambda_val < 0.001:
            p_value = 1.0
        else:
            p_value = 2.0 * np.exp(-2.0 * lambda_val * lambda_val)
            p_value = float(min(max(p_value, 0.0), 1.0))

        return ks_stat, p_value

    def _adwin_test(
        self, ref_errors: np.ndarray, det_errors: np.ndarray,
    ) -> tuple[bool, float]:
        """ADWIN-inspired test for change in error rate.

        Uses a z-test for difference in proportions.
        Returns (drift_detected, p_value).
        """
        n1 = len(ref_errors)
        n2 = len(det_errors)
        p1 = ref_errors.mean()
        p2 = det_errors.mean()

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se < 1e-10:
            return False, 1.0

        z = abs(p2 - p1) / se

        # Approximate p-value from z-score (standard normal)
        p_value = 2.0 * (1.0 - self._norm_cdf(z))

        drift_detected = p_value < self._cfg.adwin_delta
        return drift_detected, float(p_value)

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Approximate standard normal CDF (Abramowitz & Stegun)."""
        if x < -8:
            return 0.0
        if x > 8:
            return 1.0
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x) / np.sqrt(2)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return 0.5 * (1.0 + sign * y)

    @staticmethod
    def _classify_drift_severity(drift_ratio: float, max_psi: float) -> str:
        """Classify overall drift severity from feature drift metrics."""
        if max_psi > 0.5 or drift_ratio > 0.5:
            return "critical"
        if max_psi > 0.25 or drift_ratio > 0.3:
            return "high"
        if max_psi > 0.15 or drift_ratio > 0.15:
            return "moderate"
        if max_psi > 0.1 or drift_ratio > 0.05:
            return "low"
        return "none"

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"none": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}.get(severity, 0)
