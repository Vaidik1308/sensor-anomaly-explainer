"""
anomaly_processor.py — Telemetry Parsing, Schema Validation & Anomaly Pre-Processing

This module validates incoming IoT sensor telemetry data against a canonical
8-sensor schema for industrial monitoring operations. It rejects any data
that does not conform, and computes anomaly deviation metrics for valid inputs.
"""

import re
from datetime import datetime
from typing import Any


# ── Canonical Sensor Schema ──────────────────────────────────────────────────
# The ONLY sensor types and units this system will ever accept.
# Each entry maps a sensor code to its expected unit, human-readable name,
# and a brief description of what the sensor monitors.
ALLOWED_SENSOR_TYPES: dict[str, dict[str, str]] = {
    "TEMP": {
        "unit": "°C",
        "name": "Temperature Sensor",
        "description": "Temperature monitoring in industrial zones",
    },
    "PRESSURE": {
        "unit": "PSI",
        "name": "Pressure Sensor",
        "description": "Pipeline, tank, hydraulic system pressure",
    },
    "HUMIDITY": {
        "unit": "%RH",
        "name": "Humidity Sensor",
        "description": "Relative humidity in storage/production areas",
    },
    "VIBRATION": {
        "unit": "mm/s",
        "name": "Vibration Sensor",
        "description": "Machinery vibration monitoring",
    },
    "CURRENT": {
        "unit": "Amps",
        "name": "Electrical Current Sensor",
        "description": "Electrical current draw monitoring",
    },
    "FLOW": {
        "unit": "L/min",
        "name": "Flow Rate Sensor",
        "description": "Fluid flow rate in pipelines",
    },
    "GAS": {
        "unit": "PPM",
        "name": "Gas Leak Sensor",
        "description": "Toxic/flammable gas detection",
    },
    "SMOKE": {
        "unit": "PPM",
        "name": "Smoke / Fire Sensor",
        "description": "Smoke/fire detection",
    },
}

# Sensor types that are always CRITICAL when an anomaly is detected,
# regardless of the deviation percentage.
ALWAYS_CRITICAL_TYPES: set[str] = {"GAS", "SMOKE"}

# Regex for sensor_id: uppercase letters, digits, and hyphens, max 32 chars.
# Examples: "TEMP-3621", "GAS-0012"
SENSOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9\-_]{1,32}$")


def validate_telemetry(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a single telemetry reading against the canonical sensor schema.

    Expected input fields:
        sensor_id        — str, alphanumeric + hyphens/underscores, max 32 chars
        sensor_type      — str, must be a key in ALLOWED_SENSOR_TYPES
        value            — int or float, the measured reading
        unit             — str, must match the unit defined for the sensor_type
        location         — str, non-empty, max 128 chars
        timestamp        — str, ISO 8601 format
        normal_range_min — int or float, lower bound of normal range
        normal_range_max — int or float, upper bound of normal range

    Args:
        data: A dictionary representing one sensor reading.

    Returns:
        A dict with:
            'valid'  (bool)         — whether the input passed all checks
            'errors' (list[str])    — list of validation error messages
            'data'   (dict)         — the original (or cleaned) input data
    """
    errors: list[str] = []

    # ── Required fields check ────────────────────────────────────────────
    required_fields = [
        "sensor_id",
        "sensor_type",
        "value",
        "unit",
        "timestamp",
        "location",
        "normal_range_min",
        "normal_range_max",
    ]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    # If any required fields are missing, return early — can't validate further
    if errors:
        return {"valid": False, "errors": errors, "data": data}

    # ── sensor_id validation ─────────────────────────────────────────────
    sensor_id = data["sensor_id"]
    if not isinstance(sensor_id, str) or not SENSOR_ID_PATTERN.match(sensor_id):
        errors.append(
            f"Invalid sensor_id: '{sensor_id}'. "
            "Must be alphanumeric with hyphens/underscores, max 32 chars."
        )

    # ── sensor_type validation ───────────────────────────────────────────
    sensor_type = data["sensor_type"]
    if sensor_type not in ALLOWED_SENSOR_TYPES:
        allowed = ", ".join(sorted(ALLOWED_SENSOR_TYPES.keys()))
        errors.append(
            f"Unsupported sensor_type: '{sensor_type}'. Allowed types: {allowed}"
        )
        # Cannot validate unit if the type itself is invalid — return early
        return {"valid": False, "errors": errors, "data": data}

    # ── unit validation (must match the expected unit for this sensor_type) ──
    expected_unit = ALLOWED_SENSOR_TYPES[sensor_type]["unit"]
    if data["unit"] != expected_unit:
        errors.append(
            f"Invalid unit '{data['unit']}' for sensor_type '{sensor_type}'. "
            f"Expected: '{expected_unit}'"
        )

    # ── value validation (must be numeric) ───────────────────────────────
    value = data["value"]
    if value is None or not isinstance(value, (int, float)):
        errors.append(
            f"Invalid value: '{value}'. Must be a valid number (int or float)."
        )

    # ── timestamp validation (ISO 8601) ──────────────────────────────────
    timestamp = data["timestamp"]
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        errors.append(
            f"Invalid timestamp: '{timestamp}'. Must be ISO 8601 format."
        )

    # ── location validation ──────────────────────────────────────────────
    location = data["location"]
    if not isinstance(location, str) or len(location.strip()) == 0:
        errors.append("Invalid location: must be a non-empty string.")
    elif len(location) > 128:
        errors.append(
            f"Invalid location: exceeds maximum length of 128 characters "
            f"(got {len(location)})."
        )

    # ── normal_range_min / normal_range_max validation ───────────────────
    range_min = data["normal_range_min"]
    range_max = data["normal_range_max"]

    if not isinstance(range_min, (int, float)):
        errors.append(
            f"Invalid normal_range_min: '{range_min}'. Must be a number."
        )
    if not isinstance(range_max, (int, float)):
        errors.append(
            f"Invalid normal_range_max: '{range_max}'. Must be a number."
        )

    # Only check ordering if both values are numeric
    if (
        isinstance(range_min, (int, float))
        and isinstance(range_max, (int, float))
        and range_min >= range_max
    ):
        errors.append(
            f"Invalid range: normal_range_min ({range_min}) must be less than "
            f"normal_range_max ({range_max})."
        )

    return {"valid": len(errors) == 0, "errors": errors, "data": data}


def compute_anomaly_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """
    Compute anomaly deviation metrics for a validated telemetry reading.

    Determines whether the sensor value falls outside the normal range, and
    if so, calculates how far it deviates and assigns a severity level.

    Severity thresholds (percentage of range span):
        - Low:      deviation <= 15%
        - Medium:   deviation <= 40%
        - High:     deviation <= 80%
        - Critical: deviation >  80%

    Special rule: GAS and SMOKE sensors are ALWAYS Critical when out of range,
    regardless of the deviation percentage, because they represent immediate
    safety hazards.

    Args:
        data: A validated telemetry dictionary (must have passed validate_telemetry).

    Returns:
        A dict with:
            'is_anomaly'        (bool)  — True if the value is outside normal range
            'deviation_value'   (float) — absolute deviation from the nearest boundary
            'deviation_percent' (float) — deviation as a percentage of range span
            'direction'         (str)   — 'above', 'below', or 'normal'
            'severity'          (str)   — 'Normal', 'Low', 'Medium', 'High', or 'Critical'
            'summary'           (str)   — human-readable description of the result
    """
    value = data["value"]
    range_min = data["normal_range_min"]
    range_max = data["normal_range_max"]
    range_span = range_max - range_min
    sensor_type = data["sensor_type"]
    unit = data["unit"]

    # ── Determine if the value is outside the expected range ─────────────
    if value > range_max:
        deviation_value = value - range_max
        direction = "above"
    elif value < range_min:
        deviation_value = range_min - value
        direction = "below"
    else:
        # Value is within normal range — no anomaly
        return {
            "is_anomaly": False,
            "deviation_value": 0,
            "deviation_percent": 0.0,
            "direction": "normal",
            "severity": "Normal",
            "summary": (
                f"Sensor {data['sensor_id']} at {data['location']}: "
                f"value {value} {unit} is within normal range "
                f"[{range_min}, {range_max}]."
            ),
        }

    # ── Calculate percentage deviation relative to range span ────────────
    # Guard against zero-span ranges (would cause division by zero)
    if range_span > 0:
        deviation_percent = (deviation_value / range_span) * 100
    else:
        deviation_percent = 100.0

    # ── Determine severity ───────────────────────────────────────────────
    # GAS and SMOKE anomalies are ALWAYS Critical (safety-critical sensors)
    if sensor_type in ALWAYS_CRITICAL_TYPES:
        severity = "Critical"
    elif deviation_percent <= 15:
        severity = "Low"
    elif deviation_percent <= 40:
        severity = "Medium"
    elif deviation_percent <= 80:
        severity = "High"
    else:
        severity = "Critical"

    # ── Build human-readable summary ─────────────────────────────────────
    direction_word = "exceeds maximum" if direction == "above" else "below minimum"
    boundary = range_max if direction == "above" else range_min
    summary = (
        f"Sensor {data['sensor_id']} at {data['location']}: "
        f"value {value} {unit} {direction_word} {boundary} {unit} "
        f"(deviation: {deviation_value:.1f} {unit}, "
        f"{deviation_percent:.1f}% of range span). "
        f"Severity: {severity}."
    )

    return {
        "is_anomaly": True,
        "deviation_value": round(deviation_value, 2),
        "deviation_percent": round(deviation_percent, 2),
        "direction": direction,
        "severity": severity,
        "summary": summary,
    }


def process_telemetry(data: dict[str, Any]) -> dict[str, Any]:
    """
    Full processing pipeline: validate telemetry, then compute anomaly metrics.

    This is the main entry point called by routes.py. It orchestrates
    validation and anomaly computation in sequence.

    Args:
        data: Raw telemetry dictionary from the API request.

    Returns:
        A dict with:
            'validation' — result of validate_telemetry()
            'anomaly'    — result of compute_anomaly_metrics() (None if invalid)
    """
    # Step 1: Validate the input against the canonical schema
    validation = validate_telemetry(data)

    if not validation["valid"]:
        return {"validation": validation, "anomaly": None}

    # Step 2: Compute anomaly metrics for valid telemetry
    anomaly = compute_anomaly_metrics(data)

    return {"validation": validation, "anomaly": anomaly}
