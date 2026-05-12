# Optimization Recommendations — Implementation Guide

**Prepared for**: Disaster Risk Assessment System - DigitalOcean Deployment  
**Status**: Ready for Implementation  
**Estimated Effort**: 20–25 hours (3 business days)

---

## 1. Fix Fusion Layer Scale Consistency

### **File**: `modules/fusion.py`

**Current Code (BROKEN)**:
```python
def fuse_risk(predicted: float, rule_score: float, alpha: float = 0.5, beta: float = 0.5) -> float:
    predicted_norm = max(0.0, min(1.0, predicted / 3.0))
    rule_norm      = max(0.0, min(1.0, rule_score))  # WRONG: assumes rule_score ≤ 1.0
    fused_norm = (alpha * predicted_norm) + (beta * rule_norm)
    fused = fused_norm * 3.0
    return fused
```

**Issues**:
- Assumes `rule_score ≤ 1.0` but indicator weights can produce scores > 1.0
- Silent truncation when rule_score exceeds 1.0 (loses information)
- No context about barangay for adaptive weighting

**Optimized Code**:
```python
def fuse_risk(
    predicted: float, 
    rule_score: float,
    barangay_id: int = None,
    hazard_profile: dict = None,
    alpha: float = None,
    beta: float = None
) -> float:
    """
    Fuses ML prediction (0–3) and rule-based score (0–1) with adaptive per-barangay weights.
    
    Both inputs are normalized to [0,1] using their known ranges, then fused and rescaled to [0,3].
    
    Args:
        predicted:      CNN-LSTM output (0–3 range)
        rule_score:     Weighted sum of rule engine indicators (0–1 range, but can exceed)
        barangay_id:    Optional for adaptive weighting lookup
        hazard_profile: Optional barangay context for weight computation
        alpha:          ML prediction weight (default computed from hazard_profile)
        beta:           Rule engine weight (default computed from hazard_profile)
    
    Returns:
        Final fused risk score (0–3 range)
    """
    # ── Determine adaptive weights if not explicitly provided ────────────────
    if alpha is None or beta is None:
        alpha, beta = _compute_fusion_weights(barangay_id, hazard_profile)
    
    # Validate weights
    if abs((alpha + beta) - 1.0) > 1e-6:
        logger.warning("alpha + beta = %.2f (expected 1.0). Normalizing.", alpha + beta)
        total = alpha + beta
        alpha, beta = alpha / total, beta / total
    
    # ── Normalize both inputs to [0,1] using their known ranges ──────────────
    # ML output is guaranteed 0–3 (clamped in CNN-LSTM forward)
    predicted_norm = max(0.0, min(1.0, predicted / 3.0))
    
    # Rule score is bounded by sum of weights (which should sum to 1.0)
    # If weights are correctly normalized, rule_score ≤ 1.0
    # If they exceed 1.0 due to bugs, we clip and log warning
    rule_norm = max(0.0, min(1.0, rule_score / 1.0))
    if rule_score > 1.0:
        logger.warning(
            "Rule score exceeds 1.0: %.4f. This indicates indicator weights don't sum to 1.0. "
            "Check compute_barangay_weights() and indicator normalization.",
            rule_score
        )
    
    # ── Fuse on normalized scale ──────────────────────────────────────────────
    fused_norm = (alpha * predicted_norm) + (beta * rule_norm)
    
    # ── Rescale to output range [0,3] ─────────────────────────────────────────
    fused = fused_norm * 3.0
    
    logger.debug(
        "Fusion | barangay_id=%s | predicted=%.4f (norm=%.4f) | "
        "rule=%.4f (norm=%.4f) | alpha=%.2f beta=%.2f | fused=%.4f",
        barangay_id, predicted, predicted_norm, rule_score, rule_norm, alpha, beta, fused
    )
    
    return fused


def _compute_fusion_weights(barangay_id: int = None, hazard_profile: dict = None) -> tuple:
    """
    Compute per-barangay fusion weights (alpha, beta).
    
    HIGH hazard zones (accurate GIS data) → trust rule engine more (beta=0.6)
    LOW hazard zones (uncertain exposure) → balance both sources (beta=0.5)
    MODERATE zones → slight rule engine bias (beta=0.55)
    
    Returns:
        (alpha, beta) where alpha + beta = 1.0
    """
    if hazard_profile is None and barangay_id is not None:
        try:
            from modules.database import get_barangay_hazard_profile
            hazard_profile = get_barangay_hazard_profile(barangay_id)
        except Exception as e:
            logger.warning("Could not load hazard profile for barangay %d: %s", barangay_id, e)
            hazard_profile = {}
    
    overall = hazard_profile.get("overall_hazard", "MODERATE") if hazard_profile else "MODERATE"
    
    if overall == "HIGH":
        # GIS data is highly certain (shapefiles, site surveys)
        # Trust rule engine more
        alpha, beta = 0.40, 0.60
    elif overall == "LOW":
        # Minimal structural hazard, weather is primary indicator
        # More balanced, slight ML advantage (captures temporal anomalies)
        alpha, beta = 0.55, 0.45
    else:  # MODERATE
        # Default: balanced with slight rule engine bias
        alpha, beta = 0.45, 0.55
    
    logger.debug(
        "Fusion weights for barangay_id=%s | overall=%s → alpha=%.2f beta=%.2f",
        barangay_id, overall, alpha, beta
    )
    
    return alpha, beta
```

### **Usage Updates**

**In prediction_routes.py**:
```python
# Before (current):
final_risk = fuse_risk(predicted, rule_score_total)

# After (optimized):
final_risk = fuse_risk(
    predicted, 
    rule_score_total,
    barangay_id=req.barangay_id,
    hazard_profile=hazard_profile
)
```

**In simulation.py**:
```python
# Before:
final_score = fuse_risk(ml_score, rule_score)

# After:
final_score = fuse_risk(
    ml_score,
    rule_score,
    barangay_id=barangay_id,
    hazard_profile=hazard_profile
)
```

---

## 2. Add Per-Barangay Weather Sequences

### **Step 2a: Update Database Schema**

**File**: `sql/supabase_training_schema.sql`

```sql
-- Add barangay_id to weather_data table
ALTER TABLE weather_data 
ADD COLUMN IF NOT EXISTS barangay_id INTEGER DEFAULT NULL;

-- Add index for barangay-time queries
CREATE INDEX IF NOT EXISTS idx_weather_barangay_timestamp 
ON weather_data(barangay_id, timestamp DESC);

-- Backfill existing data (optional, for historical consistency)
-- UPDATE weather_data SET barangay_id = NULL WHERE barangay_id IS NULL;

-- Add barangay_location table for coordinate lookup
CREATE TABLE IF NOT EXISTS barangay_location (
    barangay_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL
);

-- Populate from barangay_list (if available)
-- INSERT INTO barangay_location (barangay_id, name, lat, lon)
-- SELECT barangay_id, name, lat, lon FROM barangay_list
-- ON CONFLICT DO NOTHING;
```

### **Step 2b: Update weather_api.py**

**File**: `modules/weather_api.py`

```python
def save_weather_data(weather: dict, barangay_id: int = None) -> None:
    """Save weather data with optional per-barangay tracking."""
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO weather_data (
                    barangay_id, city, temperature, pressure, humidity,
                    wind_speed, rainfall, timestamp
                )
                VALUES (:barangay_id, :city, :temperature, :pressure, :humidity,
                        :wind_speed, :rainfall, :timestamp)
            """), {
                "barangay_id": barangay_id,
                "city":        "Mamburao",
                "temperature": weather.get("temperature"),
                "pressure":    weather.get("pressure"),
                "humidity":    weather.get("humidity"),
                "wind_speed":  weather.get("wind_speed"),
                "rainfall":    weather.get("rainfall"),
                "timestamp":   datetime.now(timezone.utc),
            })
        logger.info(
            "Weather data saved for barangay_id=%s: rain=%.1f mm, temp=%.1f°C",
            barangay_id, weather.get("rainfall", 0), weather.get("temperature", 0)
        )
    except Exception as e:
        logger.error("save_weather_data error: %s", e)


def save_training_samples(weather: dict, barangay_id: int = None) -> None:
    """Save training samples with per-barangay weather records."""
    from sqlalchemy import text

    barangay_profiles = {
        1:  {"soil": 2.07, "flood": 1.8, "storm_surge": 4.2},
        # ... (keep existing)
    }

    rainfall     = weather.get("rainfall", 0)
    humidity     = weather.get("humidity", 0)
    current_time = datetime.now(timezone.utc)

    try:
        with engine.begin() as conn:
            for bid, profile in barangay_profiles.items():
                risk_label = _risk_label_for_barangay(rainfall, bid)
                
                conn.execute(text("""
                    INSERT INTO barangay_training_data (
                        barangay_id, timestamp, rainfall, humidity,
                        soil, flood, storm_surge, risk_label
                    )
                    VALUES (:barangay_id, :timestamp, :rainfall, :humidity,
                            :soil, :flood, :storm_surge, :risk_label)
                """), {
                    "barangay_id": bid,
                    "timestamp":   current_time,
                    "rainfall":    rainfall,
                    "humidity":    humidity,
                    "soil":        profile["soil"],
                    "flood":       profile["flood"],
                    "storm_surge": profile["storm_surge"],
                    "risk_label":  risk_label,
                })
        logger.info("Training samples saved for all 15 barangays.")
    except Exception as e:
        logger.error("save_training_samples error: %s", e)
```

### **Step 2c: Update database.py**

**File**: `modules/database.py`

```python
def get_recent_weather(barangay_id: int, limit: int = 10) -> list:
    """
    Returns recent weather observations for a **specific barangay**.
    
    Uses barangay_id to retrieve spatially-aware weather sequence.
    Falls back to "Mamburao" city data if no barangay-specific records.
    """
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            # Try to fetch barangay-specific weather first
            rows = conn.execute(text("""
                SELECT timestamp, temperature, pressure, humidity,
                       wind_speed, rainfall, rainfall_category,
                       season, risk_level
                FROM weather_data
                WHERE barangay_id = :barangay_id
                ORDER BY timestamp DESC
                LIMIT :limit
            """), {"barangay_id": barangay_id, "limit": limit}).fetchall()
            
            # Fallback to city-wide data if no barangay-specific records
            if not rows:
                logger.debug(
                    "No barangay-specific weather for %d. Falling back to city data.",
                    barangay_id
                )
                rows = conn.execute(text("""
                    SELECT timestamp, temperature, pressure, humidity,
                           wind_speed, rainfall, rainfall_category,
                           season, risk_level
                    FROM weather_data
                    WHERE city = 'Mamburao' AND barangay_id IS NULL
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """), {"limit": limit}).fetchall()

        return [
            {
                "timestamp":       row[0],
                "temperature":     float(row[1]) if row[1] is not None else 0.0,
                "pressure":        float(row[2]) if row[2] is not None else 0.0,
                "humidity":        float(row[3]) if row[3] is not None else 0.0,
                "wind_speed":      float(row[4]) if row[4] is not None else 0.0,
                "rainfall":        float(row[5]) if row[5] is not None else 0.0,
                "rainfall_category": row[6] or "UNKNOWN",
                "season":          row[7] or "UNKNOWN",
                "risk_level":      row[8] or "UNKNOWN",
            }
            for row in rows
        ]
    except Exception as e:
        logger.error("get_recent_weather error for barangay_id=%d: %s", barangay_id, e)
        return []
```

---

## 3. Add Storm Surge & Humidity to Rule Engine

### **File**: `modules/context.py`

```python
import logging

logger = logging.getLogger(__name__)

# ── Updated rules ─────────────────────────────────────────────────────────
DEFAULT_RULES = [
    {"priority": 1, "threshold": 2.5, "action": "HIGH"},
    {"priority": 2, "threshold": 1.5, "action": "MODERATE"},
    {"priority": 3, "threshold": 0.0, "action": "LOW"},
]

# ── Updated base weights (now 5 indicators) ────────────────────────────────
BASE_WEIGHTS = {
    "rainfall":    0.35,   # Reduced from 0.40 (now includes humidity)
    "soil":        0.25,   # Reduced from 0.30
    "flood":       0.20,
    "humidity":    0.10,   # NEW: Saturation indicator
    "storm_surge": 0.10,   # NEW: Coastal hazard
}

INDICATOR_SET = ["rainfall", "soil", "flood", "humidity", "storm_surge"]


def compute_barangay_weights(barangay_id: int, hazard_profile: dict = None) -> dict:
    """
    Computes per-barangay weights based on hazard profile.
    
    Updated to include humidity and storm_surge.
    
    HIGH barangays (Tayamaan, Poblacion 8):
      - High flood risk + SSA3 storm surge
      - Flood and storm surge dominant
    
    LOW barangays (San Luis, Tangkalan):
      - Minimal structural hazard
      - Rainfall is primary early warning
    
    MODERATE barangays:
      - Balanced exposure
      - Use base weights with slight adjustments
    """
    profile = hazard_profile or BARANGAY_PROFILES.get(barangay_id, {})
    overall   = profile.get("overall", "MODERATE")
    ssa_level = profile.get("ssa_level", 0)

    if overall == "HIGH":
        if ssa_level == 3:
            # HIGH flood + SSA3 storm surge
            weights = {
                "rainfall":    0.15,  # Reduced: already exposed
                "soil":        0.20,
                "flood":       0.35,  # Primary risk factor
                "humidity":    0.08,
                "storm_surge": 0.22,  # Significant coastal risk
            }
        else:
            # HIGH flood only (unlikely in study area)
            weights = {
                "rainfall":    0.20,
                "soil":        0.25,
                "flood":       0.35,
                "humidity":    0.10,
                "storm_surge": 0.10,
            }

    elif overall == "LOW":
        # Minimal structural hazard — rainfall + humidity are early warnings
        weights = {
            "rainfall":    0.40,   # Primary indicator
            "soil":        0.25,
            "flood":       0.10,
            "humidity":    0.15,   # Pre-saturation indicator
            "storm_surge": 0.10,
        }

    else:  # MODERATE
        # Use base weights with coastal adjustment
        if ssa_level == 3:
            # Coastal moderate: add storm surge weighting
            weights = {
                "rainfall":    0.32,
                "soil":        0.23,
                "flood":       0.18,
                "humidity":    0.09,
                "storm_surge": 0.18,  # Elevated for coastal exposure
            }
        else:
            # Inland moderate: base weights
            weights = BASE_WEIGHTS.copy()

    # Normalize to sum to 1.0
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    logger.info(
        "Barangay %d (%s) | overall=%s ssa=%d → weights=%s",
        barangay_id, profile.get("name", "Unknown"), overall, ssa_level, weights,
    )
    return weights


def load_context(HR: dict, hazard_profile: dict = None) -> tuple:
    """Loads context with per-barangay weights including new indicators."""
    hazard_type  = HR.get("type", "Unknown")
    barangay     = HR.get("location", "Unknown")
    barangay_id  = HR.get("barangay_id", 0)

    weights = compute_barangay_weights(barangay_id, hazard_profile)
    sorted_rules = sorted(DEFAULT_RULES, key=lambda r: r["threshold"], reverse=True)

    logger.debug(
        "Context loaded | barangay_id=%d name=%s hazard=%s weights=%s",
        barangay_id, barangay, hazard_type, weights
    )

    return hazard_type, barangay, INDICATOR_SET, weights, sorted_rules
```

### **File**: `modules/normalization.py` (add humidity & storm_surge bounds)

```python
# ── Add global ranges for new indicators ───────────────────────────────────
INDICATOR_RANGES = {
    "rainfall":    (0.0, 40.0),
    "soil":        (0.0, 3.0),
    "flood":       (0.0, 1.0),
    "humidity":    (0.0, 100.0),      # NEW
    "storm_surge": (0.0, 1.0),        # NEW (normalized score)
}

# ── Add per-barangay humidity bounds ──────────────────────────────────────
BARANGAY_HUMIDITY_BOUNDS = {
    # All barangays: 40–100% RH (tropical)
    # Values > 85% = saturation risk
    i: (40.0, 95.0) for i in range(1, 16)
}

# ── Add per-barangay storm surge bounds ──────────────────────────────────
BARANGAY_STORM_SURGE_BOUNDS = {
    # Coastal (SSA3): 0–1.0 (high exposure)
    1: (0.0, 1.0),   # Balansay — SSA3
    2: (0.0, 1.0),   # Fatima — SSA3
    3: (0.0, 1.0),   # Payompon — SSA3
    4: (0.0, 0.2),   # San Luis — inland
    5: (0.0, 1.0),   # Talabaan — SSA3
    6: (0.0, 0.2),   # Tangkalan — inland
    7: (0.0, 1.0),   # Tayamaan — SSA3
    8: (0.0, 1.0),   # Poblacion 1 — SSA3
    9: (0.0, 1.0),   # Poblacion 2 — SSA3
    10: (0.0, 1.0),  # Poblacion 3 — SSA3
    11: (0.0, 1.0),  # Poblacion 4 — SSA3
    12: (0.0, 0.2),  # Poblacion 5 — inland
    13: (0.0, 1.0),  # Poblacion 6 — SSA3
    14: (0.0, 0.2),  # Poblacion 7 — inland
    15: (0.0, 1.0),  # Poblacion 8 — SSA3
}


def get_indicator_bounds(indicator: str, barangay_id: int) -> tuple:
    """Returns per-barangay bounds for indicators."""
    if indicator == "rainfall" and barangay_id in BARANGAY_RAINFALL_BOUNDS:
        return BARANGAY_RAINFALL_BOUNDS[barangay_id]
    
    if indicator == "flood" and barangay_id in BARANGAY_FLOOD_BOUNDS:
        return BARANGAY_FLOOD_BOUNDS[barangay_id]
    
    if indicator == "humidity" and barangay_id in BARANGAY_HUMIDITY_BOUNDS:
        return BARANGAY_HUMIDITY_BOUNDS[barangay_id]
    
    if indicator == "storm_surge" and barangay_id in BARANGAY_STORM_SURGE_BOUNDS:
        return BARANGAY_STORM_SURGE_BOUNDS[barangay_id]
    
    return INDICATOR_RANGES.get(indicator, (0.0, 1.0))
```

---

## 4. Extend Temporal Sequence Length

### **File**: `modules/cnn_lstm.py`

```python
# Change SEQ_LEN from 10 to 90
SEQ_LEN = 90  # Changed from 10: Use 90 days of historical data

# Optionally increase training epochs for larger sequences
_TRAIN_EPOCHS = 100  # Changed from 50

# Batch size can be reduced (larger sequences = more memory)
_BATCH_SIZE = 128  # Changed from 256
```

**Action**: After this change:
1. Clear weights directory: `rm weights/cnn_lstm_*.pt`
2. Retrain all models (this will happen automatically on first `/predict-risk` call)
3. Monitor training logs for convergence

---

## 5. Implement GIS Hazard Caching

### **File**: `modules/database.py`

```python
import logging
from functools import lru_cache

import psycopg2
from sqlalchemy import create_engine

from modules.config import get_database_url

logger = logging.getLogger(__name__)

DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)

# ── Hazard profile cache (loaded at app startup) ───────────────────────────
_hazard_cache: dict = {}


def load_hazard_cache() -> None:
    """Load all hazard profiles into memory at startup."""
    global _hazard_cache
    try:
        profiles = list_barangay_profiles()
        _hazard_cache = {p["barangay_id"]: p for p in profiles}
        logger.info("Loaded %d hazard profiles into cache", len(_hazard_cache))
    except Exception as e:
        logger.error("Failed to load hazard cache: %s. Will use per-request DB queries.", e)


def get_barangay_hazard_profile(barangay_id: int) -> dict:
    """
    Returns full hazard profile — from cache if available, else from DB.
    Cache is loaded at application startup for production use.
    """
    # Try cache first (O(1) lookup)
    if _hazard_cache and barangay_id in _hazard_cache:
        return _hazard_cache[barangay_id]
    
    # Fallback to DB query if cache empty (development, or cache load failed)
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT
                    flood_hazard_level, flood_hazard_score,
                    in_ssa1, in_ssa2, in_ssa3,
                    max_ssa_level, storm_surge_score, overall_hazard
                FROM barangay_hazard_profile
                WHERE barangay_id = :id
            """), {"id": barangay_id}).fetchone()

        if row:
            profile = {
                "flood_hazard_level": row[0],
                "flood_hazard_score": float(row[1]),
                "in_ssa1":            bool(row[2]),
                "in_ssa2":            bool(row[3]),
                "in_ssa3":            bool(row[4]),
                "max_ssa_level":      int(row[5]),
                "storm_surge_score":  float(row[6]),
                "overall_hazard":     row[7],
                "overall":            row[7],
                "ssa_level":          int(row[5]),
                "flood_score":        float(row[1]),
            }
            return profile
    except Exception as e:
        logger.error("get_barangay_hazard_profile DB error: %s", e)

    logger.warning("No hazard profile for barangay_id=%d — using LOW defaults", barangay_id)
    return {
        "flood_hazard_level": "Low",
        "flood_hazard_score": 0.20,
        "in_ssa1":            False,
        "in_ssa2":            False,
        "in_ssa3":            False,
        "max_ssa_level":      0,
        "storm_surge_score":  0.0,
        "overall_hazard":     "LOW",
        "overall":            "LOW",
        "ssa_level":          0,
        "flood_score":        0.20,
    }
```

### **File**: `main_api.py` (update lifespan to load cache)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load hazard profiles cache
    from modules.database import load_hazard_cache
    try:
        load_hazard_cache()
        logger.info("Hazard profile cache loaded successfully.")
    except Exception as e:
        logger.warning("Failed to preload hazard cache: %s", e)

    # Optionally preload models
    if _env_enabled("PRELOAD_MODELS_ON_STARTUP", "false"):
        import threading
        threading.Thread(target=preload_all_models, daemon=True).start()
        logger.info("Background model preload started.")
    
    yield

    # Cleanup (if needed)
    logger.info("Application shutting down.")
```

---

## 6. Update Prediction Route to Use Optimized Components

### **File**: `routes/prediction_routes.py`

```python
@router.post("/predict-risk")
def predict(req: PredictRequest):
    """
    Full prediction pipeline with optimized fusion and per-barangay weather.
    """
    try:
        # ── 1. Resolve barangay ────────────────────────────────────────────
        lat, lon = get_barangay_centroid(req.barangay_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Barangay {req.barangay_id} not found."
        )

    try:
        barangay_name   = get_barangay_name(req.barangay_id)
        hazard_profile  = get_barangay_hazard_profile(req.barangay_id)
        B               = hazard_profile

        HR = {
            "type":        req.hazard_type,
            "location":    barangay_name,
            "barangay_id": req.barangay_id,
            "lat":         lat,
            "lon":         lon,
        }

        E = {"osm_lat": lat, "osm_lon": lon}

        # ── 2. Weather ─────────────────────────────────────────────────────
        weather = get_weather(lat, lon)
        E.update(weather)

        # Override flood/storm_surge with structural hazard scores
        E["flood"]       = hazard_profile["flood_hazard_score"]
        E["storm_surge"] = hazard_profile["storm_surge_score"]

        # Fall back to DB weather if API returned zeros
        if E.get("rainfall", 0) == 0:
            db_features = get_barangay_features(req.barangay_id)
            if db_features["rainfall"] > 0:
                E["rainfall"] = db_features["rainfall"]

        # ── 3. Rule engine ─────────────────────────────────────────────────
        hazard_type, barangay, indicators, weights, rules = load_context(HR, B)

        weighted_scores  = compute_weighted_scores(HR, E, B, weights, req.barangay_id)
        rule_score_total = compute_risk_score(weighted_scores)

        # ── 4. ML prediction ───────────────────────────────────────────────
        # OPTIMIZED: Get per-barangay weather sequence
        recent_history = list(reversed(get_recent_weather(req.barangay_id, limit=90)))
        predicted      = predict_risk(req.barangay_id, E, {"history": recent_history})

        # ── 5. Fusion + classification ─────────────────────────────────────
        # OPTIMIZED: Use adaptive fusion weights
        final_risk = fuse_risk(
            predicted, 
            rule_score_total,
            barangay_id=req.barangay_id,
            hazard_profile=hazard_profile
        )
        risk_level = apply_rules(rules, final_risk)

        # ── 6. Persist assessment ──────────────────────────────────────────
        store_data(HR, rule_score_total, predicted, final_risk, risk_level)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("predict-risk failed for barangay_id=%d: %s", req.barangay_id, e)
        raise HTTPException(
            status_code=500,
            detail="Risk assessment failed. Please try again later."
        )

    return {
        "barangay_id":   req.barangay_id,
        "barangay_name": barangay_name,
        "final_risk":    round(final_risk, 4),
        "risk_level":    risk_level,
    }
```

---

## 7. Validation Checklist

### **Before Deployment**

- [ ] **Fusion Layer**: Test with high flood + storm surge scenario → should hit HIGH
- [ ] **Weather Sequences**: Verify `get_recent_weather(1)` returns Balansay data, not generic "Mamburao"
- [ ] **Storm Surge Weights**: Tayamaan (HIGH) should have 0.20+ storm_surge weight
- [ ] **Temporal Sequence**: Confirm SEQ_LEN=90 models trained successfully
- [ ] **GIS Cache**: Verify `_hazard_cache` loads at startup (check logs)
- [ ] **Simulation vs Live**: Same inputs should produce identical outputs

### **Performance Tests**

```bash
# Prediction latency (target: < 200ms)
time curl -X POST http://localhost:8000/predict-risk \
  -H "Content-Type: application/json" \
  -d '{"barangay_id": 1, "hazard_type": "Flood"}'

# Batch simulation (target: < 2s for all 15 barangays)
time curl -X POST http://localhost:8000/simulate/ \
  -H "Content-Type: application/json" \
  -d '{"rainfall": 25.0, "humidity": 85.0, "wind_speed": 40.0, "temperature": 30.0}'
```

---

## 8. Implementation Timeline

### **Day 1 (Morning): Critical Fixes**
1. Fix fusion layer (30 min)
2. Add storm surge & humidity to rule engine (60 min)
3. Extend SEQ_LEN to 90 (15 min)
4. Update context.py weights (45 min)
5. Update normalization.py bounds (30 min)

**Testing**: Verify fusion, weight computation, normalization

### **Day 1 (Afternoon): Database & Weather**
6. Update database schema (barangay_id in weather_data) (30 min)
7. Update database.py cache & queries (60 min)
8. Update weather_api.py (45 min)
9. Update prediction_routes.py (30 min)

**Testing**: Per-barangay weather queries, cache loading

### **Day 2 (Overnight): Model Retraining**
- All 15 models retrain automatically on first prediction call
- Monitor logs for convergence
- Expected: 2–4 hours with 90-day sequences

### **Day 2 (Morning): Validation**
10. Manual test scenarios (HIGH/MODERATE/LOW for each barangay type) (60 min)
11. Simulation vs live prediction comparison (30 min)
12. Performance profiling (latency, DB queries) (30 min)

### **Day 3: Deployment**
- Deploy to DigitalOcean
- Run smoke tests against production database
- Monitor logs for 24 hours

---

## 9. Rollback Plan

If issues detected post-deployment:

1. **Revert fusion.py** to use `alpha=0.5, beta=0.5` (comment out adaptive logic)
2. **Revert context.py** to use only rainfall/soil/flood weights
3. **Revert SEQ_LEN** to 10 (requires model retraining)
4. **Keep** barangay_id column in weather_data (non-breaking)

**Rollback time**: < 15 minutes (only config + code changes, no data loss)

---

## Conclusion

These 7 optimizations address all critical issues while preserving the overall architecture. The system remains rule-engine-authoritative with ML-enhanced predictions, per-barangay behavior, and adaptive context awareness.

**Estimated deployment readiness**: 3 business days (including overnight model retraining)

