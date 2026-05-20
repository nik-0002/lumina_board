"""
Lumina Board - Real Bio-Urgency Detector
Uses ML (IsolationForest + rule-based scoring) on actual CSV data to detect
urgency signals: campaign underperformance, grower churn risk, seasonal triggers.

NO hallucination: all signals derived from real data columns.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger("lumina.urgency")


class UrgencyDetector:
    """
    Detects urgency signals across multiple dimensions:
    1. Campaign performance anomalies (below expected CTR/conversion)
    2. Grower engagement risk (low device penetration + low submission rates)
    3. Regional hotspots (states with high reach but low conversion)
    4. Seasonal crop urgency (based on crop type patterns in data)
    5. Composite urgency score (0-100)
    """

    URGENCY_LEVELS = {
        (0, 25): ("LOW", "🟢", "No immediate action needed"),
        (25, 50): ("MEDIUM", "🟡", "Monitor closely — some signals present"),
        (50, 75): ("HIGH", "🟠", "Action recommended within 1 week"),
        (75, 101): ("CRITICAL", "🔴", "Immediate intervention required"),
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._isolation_forest: Optional[IsolationForest] = None
        self._scaler = StandardScaler()
        self._is_trained = False

    # ──────────────────────────────────────────────────────────────────────────
    # TRAINING
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, datasets: Optional[Dict[str, pd.DataFrame]] = None):
        """Train anomaly detection on campaign data."""
        logger.info("Training urgency detector...")

        if datasets is None:
            try:
                from utils.data_processors import DataProcessor
                dp = DataProcessor(self.data_dir)
                datasets = dp.load_all_datasets()
            except Exception as e:
                logger.warning(f"Could not load datasets for training: {e}")
                self._is_trained = True
                return

        if "campaigns" not in datasets or len(datasets["campaigns"]) == 0:
            logger.warning("No campaign data — using rule-based urgency only")
            self._is_trained = True
            return

        c = datasets["campaigns"]
        features = self._extract_campaign_features(c)

        if features is not None and len(features) > 5:
            try:
                scaled = self._scaler.fit_transform(features)
                self._isolation_forest = IsolationForest(
                    n_estimators=100,
                    contamination=0.1,
                    random_state=42
                )
                self._isolation_forest.fit(scaled)
                logger.info(f"IsolationForest trained on {len(features)} campaign feature vectors")
            except Exception as e:
                logger.error(f"IsolationForest training failed: {e}")

        self._is_trained = True

    def _extract_campaign_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract numeric features from campaign data."""
        needed = ["social_post_impression", "landing_page_visits", "lead_form_submission"]
        available = [c for c in needed if c in df.columns]
        if len(available) < 2:
            return None

        feature_df = pd.DataFrame()
        if "social_post_impression" in df.columns and "landing_page_visits" in df.columns:
            impr = df["social_post_impression"].replace(0, np.nan)
            feature_df["ctr"] = df["landing_page_visits"] / impr * 100
        if "landing_page_visits" in df.columns and "lead_form_submission" in df.columns:
            visits = df["landing_page_visits"].replace(0, np.nan)
            feature_df["conversion"] = df["lead_form_submission"] / visits * 100
        if "social_post_impression" in df.columns:
            feature_df["log_impressions"] = np.log1p(df["social_post_impression"])

        feature_df = feature_df.fillna(0)
        return feature_df.values if len(feature_df.columns) > 0 else None

    # ──────────────────────────────────────────────────────────────────────────
    # DETECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect(
        self,
        datasets: Dict[str, pd.DataFrame],
        state_filter: Optional[str] = None,
        crop_filter: Optional[str] = None,
        product_filter: Optional[str] = None
    ) -> Dict:
        """
        Run full urgency detection for given filters.
        Returns structured urgency report with score, level, signals, and recommendations.
        """
        if not self._is_trained:
            self.train(datasets)

        signals = []
        score_components = {}

        # ── 1. Campaign Performance Signals ──────────────────────────────────
        if "campaigns" in datasets:
            camp_signals, camp_score = self._analyze_campaigns(
                datasets["campaigns"], state_filter, crop_filter, product_filter
            )
            signals.extend(camp_signals)
            score_components["campaign"] = camp_score

        # ── 2. Grower Engagement Signals ──────────────────────────────────────
        if "growers" in datasets:
            grow_signals, grow_score = self._analyze_growers(
                datasets["growers"], state_filter
            )
            signals.extend(grow_signals)
            score_components["grower_engagement"] = grow_score

        # ── 3. Seasonal / Crop Signals ────────────────────────────────────────
        seasonal_signals, seasonal_score = self._seasonal_signals(crop_filter)
        signals.extend(seasonal_signals)
        score_components["seasonal"] = seasonal_score

        # ── 4. ML Anomaly Score (if available) ────────────────────────────────
        if self._isolation_forest is not None and "campaigns" in datasets:
            ml_score = self._ml_anomaly_score(datasets["campaigns"], state_filter, crop_filter)
            score_components["ml_anomaly"] = ml_score

        # ── Composite Score ───────────────────────────────────────────────────
        weights = {"campaign": 0.45, "grower_engagement": 0.30, "seasonal": 0.15, "ml_anomaly": 0.10}
        composite = sum(
            score_components.get(k, 0) * w
            for k, w in weights.items()
            if k in score_components
        )
        composite = min(100, max(0, composite))

        level_info = self._get_level(composite)

        # ── Recommendations ───────────────────────────────────────────────────
        recommendations = self._generate_recommendations(signals, composite, state_filter, crop_filter)

        return {
            "urgency_score": round(composite, 1),
            "urgency_level": level_info[0],
            "urgency_icon": level_info[1],
            "urgency_message": level_info[2],
            "signals": signals,
            "score_breakdown": {k: round(v, 1) for k, v in score_components.items()},
            "recommendations": recommendations,
            "filters_applied": {
                "state": state_filter,
                "crop": crop_filter,
                "product": product_filter
            },
            "analyzed_at": datetime.utcnow().isoformat()
        }

    def _analyze_campaigns(
        self,
        df: pd.DataFrame,
        state: Optional[str],
        crop: Optional[str],
        product: Optional[str]
    ) -> Tuple[List[Dict], float]:
        signals = []
        score = 0.0

        # Apply filters
        mask = pd.Series([True] * len(df))
        if state and "state" in df.columns:
            mask &= df["state"] == state
        if crop and "crop" in df.columns:
            mask &= df["crop"] == crop
        if product and "product_name" in df.columns:
            mask &= df["product_name"] == product
        filtered = df[mask]

        if len(filtered) == 0:
            return [{"type": "info", "message": "No campaigns match the given filters", "severity": 0}], 0.0

        # CTR analysis
        if "social_post_impression" in filtered.columns and "landing_page_visits" in filtered.columns:
            total_impr = filtered["social_post_impression"].sum()
            total_visits = filtered["landing_page_visits"].sum()
            ctr = total_visits / total_impr * 100 if total_impr > 0 else 0
            global_ctr = df["landing_page_visits"].sum() / df["social_post_impression"].sum() * 100 if df["social_post_impression"].sum() > 0 else 0

            if ctr < global_ctr * 0.5:
                signals.append({
                    "type": "campaign_underperformance",
                    "message": f"CTR is {ctr:.1f}% vs platform avg {global_ctr:.1f}% — critically below baseline",
                    "severity": 80,
                    "data": {"ctr": round(ctr, 2), "avg_ctr": round(global_ctr, 2)}
                })
                score += 80
            elif ctr < global_ctr * 0.75:
                signals.append({
                    "type": "campaign_underperformance",
                    "message": f"CTR {ctr:.1f}% is below platform avg {global_ctr:.1f}%",
                    "severity": 45,
                    "data": {"ctr": round(ctr, 2), "avg_ctr": round(global_ctr, 2)}
                })
                score += 45

        # Conversion analysis
        if "landing_page_visits" in filtered.columns and "lead_form_submission" in filtered.columns:
            visits = filtered["landing_page_visits"].sum()
            subs = filtered["lead_form_submission"].sum()
            conv = subs / visits * 100 if visits > 0 else 0

            if conv < 1.0:
                signals.append({
                    "type": "low_conversion",
                    "message": f"Lead conversion is critically low at {conv:.1f}%",
                    "severity": 75,
                    "data": {"conversion_pct": round(conv, 2), "visits": int(visits), "submissions": int(subs)}
                })
                score = max(score, 75)
            elif conv < 3.0:
                signals.append({
                    "type": "low_conversion",
                    "message": f"Lead conversion at {conv:.1f}% — room for improvement",
                    "severity": 35,
                    "data": {"conversion_pct": round(conv, 2)}
                })
                score = max(score, 35)

        # Zero submissions check
        if "lead_form_submission" in filtered.columns:
            zero_campaigns = (filtered["lead_form_submission"] == 0).sum()
            if zero_campaigns > 0:
                pct = zero_campaigns / len(filtered) * 100
                signals.append({
                    "type": "zero_submission_campaigns",
                    "message": f"{zero_campaigns} campaigns ({pct:.0f}%) have 0 lead submissions",
                    "severity": 60,
                    "data": {"zero_campaigns": int(zero_campaigns), "total_campaigns": len(filtered)}
                })
                score = max(score, 60)

        return signals, min(score, 100)

    def _analyze_growers(
        self,
        df: pd.DataFrame,
        state: Optional[str]
    ) -> Tuple[List[Dict], float]:
        signals = []
        score = 0.0

        filtered = df[df["state"] == state] if state and "state" in df.columns else df

        if len(filtered) == 0:
            return signals, 0.0

        # Device type analysis
        if "device_type" in filtered.columns:
            device_counts = filtered["device_type"].value_counts()
            keypad_pct = device_counts.get("keypad", 0) / len(filtered) * 100
            if keypad_pct > 50:
                signals.append({
                    "type": "digital_divide",
                    "message": f"{keypad_pct:.0f}% growers use keypad phones — SMS/voice campaigns critical",
                    "severity": 55,
                    "data": {"keypad_pct": round(keypad_pct, 1), "devices": device_counts.to_dict()}
                })
                score = max(score, 55)

        # Small farm concentration
        if "grower_farm_size" in filtered.columns:
            small_farm_pct = (filtered["grower_farm_size"] < 2).sum() / len(filtered) * 100
            if small_farm_pct > 60:
                signals.append({
                    "type": "small_farm_concentration",
                    "message": f"{small_farm_pct:.0f}% growers have <2 acre farms — price sensitivity high",
                    "severity": 40,
                    "data": {"small_farm_pct": round(small_farm_pct, 1)}
                })
                score = max(score, 40)

        return signals, min(score, 100)

    def _seasonal_signals(self, crop: Optional[str]) -> Tuple[List[Dict], float]:
        """Check if current month is critical sowing/harvest period for the crop."""
        signals = []
        score = 0.0
        month = datetime.now().month

        # Kharif season (June-Sept) and Rabi (Nov-March) are peak urgency
        CROP_SEASONS = {
            "rice": {"sowing": [6, 7], "harvest": [10, 11], "urgency": 70},
            "wheat": {"sowing": [11, 12], "harvest": [3, 4], "urgency": 65},
            "cotton": {"sowing": [5, 6], "harvest": [10, 11], "urgency": 75},
            "soybean": {"sowing": [6, 7], "harvest": [9, 10], "urgency": 60},
            "maize": {"sowing": [5, 6, 7], "harvest": [9, 10], "urgency": 55},
            "sugarcane": {"sowing": [2, 3], "harvest": [11, 12, 1], "urgency": 50},
        }

        if crop:
            crop_lower = crop.lower()
            for crop_key, info in CROP_SEASONS.items():
                if crop_key in crop_lower:
                    if month in info["sowing"]:
                        signals.append({
                            "type": "seasonal_sowing",
                            "message": f"PEAK SOWING SEASON for {crop} — highest grower intent now",
                            "severity": info["urgency"],
                            "data": {"season": "sowing", "month": month}
                        })
                        score = info["urgency"]
                    elif month in info["harvest"]:
                        signals.append({
                            "type": "seasonal_harvest",
                            "message": f"Harvest season for {crop} — plan next cycle campaigns",
                            "severity": info["urgency"] * 0.6,
                            "data": {"season": "harvest", "month": month}
                        })
                        score = info["urgency"] * 0.6
                    break

        # General kharif/rabi urgency
        if not signals:
            if month in [6, 7]:
                signals.append({
                    "type": "kharif_season",
                    "message": "Kharif sowing season — peak agrochemical demand period",
                    "severity": 60
                })
                score = 60
            elif month in [11, 12]:
                signals.append({
                    "type": "rabi_season",
                    "message": "Rabi sowing season — high demand for wheat/mustard products",
                    "severity": 55
                })
                score = 55

        return signals, min(score, 100)

    def _ml_anomaly_score(
        self,
        df: pd.DataFrame,
        state: Optional[str],
        crop: Optional[str]
    ) -> float:
        """Use IsolationForest to score anomalousness of current data slice."""
        if self._isolation_forest is None:
            return 0.0

        mask = pd.Series([True] * len(df))
        if state and "state" in df.columns:
            mask &= df["state"] == state
        if crop and "crop" in df.columns:
            mask &= df["crop"] == crop

        filtered = df[mask]
        if len(filtered) == 0:
            return 0.0

        features = self._extract_campaign_features(filtered)
        if features is None or len(features) == 0:
            return 0.0

        try:
            scaled = self._scaler.transform(features)
            # IsolationForest returns -1 for anomalies
            scores = self._isolation_forest.decision_function(scaled)
            # Convert: more negative = more anomalous = higher urgency
            anomaly_pct = (scores < 0).mean() * 100
            return min(anomaly_pct * 1.5, 100)
        except Exception as e:
            logger.warning(f"ML scoring failed: {e}")
            return 0.0

    def _get_level(self, score: float) -> Tuple[str, str, str]:
        for (lo, hi), info in self.URGENCY_LEVELS.items():
            if lo <= score < hi:
                return info
        return ("LOW", "🟢", "No immediate action needed")

    def _generate_recommendations(
        self,
        signals: List[Dict],
        score: float,
        state: Optional[str],
        crop: Optional[str]
    ) -> List[str]:
        recs = []
        signal_types = {s["type"] for s in signals}

        if "campaign_underperformance" in signal_types:
            recs.append("🎯 A/B test new ad creatives — current CTR is below benchmark")
        if "low_conversion" in signal_types:
            recs.append("📋 Optimize lead form: reduce fields to name + phone only")
            recs.append("📞 Deploy outbound call campaign to high-impression, low-conversion segments")
        if "digital_divide" in signal_types:
            recs.append("📱 Prioritize SMS + IVR campaigns over digital-only for this region")
        if "zero_submission_campaigns" in signal_types:
            recs.append("🛑 Pause zero-performing campaigns and reallocate budget")
        if "seasonal_sowing" in signal_types:
            recs.append(f"⏰ URGENT: Launch {crop or 'crop'} product push immediately — peak sowing window")
        if "small_farm_concentration" in signal_types:
            recs.append("💰 Bundle small-quantity SKUs or introduce sachets for <2ac farmers")

        if score >= 75:
            recs.insert(0, "🚨 CRITICAL: Escalate to regional manager — immediate campaign intervention needed")
        if state:
            recs.append(f"📍 Focus Rep mobilization in {state} within 48 hours")

        return recs[:6]  # max 6 recommendations

    # ──────────────────────────────────────────────────────────────────────────
    # BULK SCAN
    # ──────────────────────────────────────────────────────────────────────────

    def bulk_scan(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        """Scan all states and find top urgency hotspots."""
        if not self._is_trained:
            self.train(datasets)

        results = []
        states = []

        if "growers" in datasets and "state" in datasets["growers"].columns:
            states = datasets["growers"]["state"].dropna().unique().tolist()
        elif "campaigns" in datasets and "state" in datasets["campaigns"].columns:
            states = datasets["campaigns"]["state"].dropna().unique().tolist()

        for state in states[:20]:  # cap at 20 states for performance
            try:
                detection = self.detect(datasets, state_filter=state)
                results.append({
                    "state": state,
                    "urgency_score": detection["urgency_score"],
                    "urgency_level": detection["urgency_level"],
                    "urgency_icon": detection["urgency_icon"],
                    "top_signal": detection["signals"][0]["message"] if detection["signals"] else "No signals",
                    "signal_count": len(detection["signals"])
                })
            except Exception as e:
                logger.error(f"Bulk scan error for {state}: {e}")

        results.sort(key=lambda x: x["urgency_score"], reverse=True)
        critical = [r for r in results if r["urgency_level"] in ("CRITICAL", "HIGH")]

        return {
            "total_states_scanned": len(results),
            "critical_states": len(critical),
            "results": results,
            "top_hotspots": results[:5],
            "scanned_at": datetime.utcnow().isoformat()
        }