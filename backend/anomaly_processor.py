"""
anomaly_processor.py — Telemetry Parsing, Schema Validation & Anomaly Pre-Processing

This module validates incoming IoT sensor telemetry data against the canonical
7-sensor schema for marine and oil industry operations. It rejects any data
that does not conform, and computes anomaly deviation metrics for valid inputs.
"""

import re
from datetime import datetime
from typing import Any


# ── Canonical Sensor Schema ──────────────────────────────────────────────────
# The ONLY sensor types and units this system will ever accept.
ALLOWED_SENSOR_TYPES = {
    "temperature": {"unit": "C", "description": "Celsius — engine, boiler, exhaust readings"},
    "pressure":    {"unit": "Pa", "description": "Pascals — pipeline, tank, hydraulic pressure"},
    "humidity":    {"unit": "%", "description": "Relative humidity — cargo holds, storage areas"},
    "vibration":   {"unit": "Hz", "description": "Frequency — engine, motor, shaft vibration"},
    "voltage":     {"unit": "V", "description": "Volts — electrical systems, generator output"},
    "flow_rate":   {"unit": "L/min", "description": "Litres per minute — fuel and oil flow"},
    "gas_level":   {"unit": "ppm", "description": "Parts per million — gas leak detection"},
}

# Regex for sensor_id: alphanumeric + underscores, max 32 chars
SENSOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,32}$")


def validate_telemetry(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate a single telemetry reading against the canonical marine/oil sensor schema.

    Args:
        data: A dictionary representing one sensor reading.

    Returns:
        A dict with 'valid' (bool), 'errors' (list of strings), and 'data' (cleaned input).
    """
    errors: list[str] = []

    # ── Required fields check ────────────────────────────────────────────
    required_fields = ["sensor_id", "sensor_type", "value", "unit", "timestamp", "location", "expected_range"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    # If critical fields are missing, return early — can't validate further
    if errors:
        return {"valid": False, "errors": errors, "data": data}

    # ── sensor_id validation ─────────────────────────────────────────────
    sensor_id = data["sensor_id"]
    if not isinstance(sensor_id, str) or not SENSOR_ID_PATTERN.match(sensor_id):
        errors.append(
            f"Invalid sensor_id: '{sensor_id}'. Must be alphanumeric + underscores, max 32 chars."
        )

    # ── sensor_type validation ───────────────────────────────────────────
    sensor_type = data["sensor_type"]
    if sensor_type not in ALLOWED_SENSOR_TYPES:
        allowed = ", ".join(sorted(ALLOWED_SENSOR_TYPES.keys()))
        errors.append(
            f"Unsupported sensor_type: '{sensor_type}'. Allowed types: {allowed}"
        )
        # Can't validate unit if type is invalid, so return early
        return {"valid": False, "errors": errors, "data": data}

    # ── unit validation (must match sensor_type) ─────────────────────────
    expected_unit = ALLOWED_SENSOR_TYPES[sensor_type]["unit"]
    if data["unit"] != expected_unit:
        errors.append(
            f"Invalid unit '{data['unit']}' for sensor_type '{sensor_type}'. Expected: '{expected_unit}'"
        )

    # ── value validation (must be numeric, not null) ─────────────────────
    value = data["value"]
    if value is None or not isinstance(value, (int, float)):
        errors.append(f"Invalid value: '{value}'. Must be a valid number (int or float).")

    # ── timestamp validation (ISO 8601) ──────────────────────────────────
    timestamp = data["timestamp"]
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        errors.append(f"Invalid timestamp: '{timestamp}'. Must be ISO 8601 format.")

    # ── location validation ──────────────────────────────────────────────
    location = data["location"]
    if not isinstance(location, str) or len(location) == 0 or len(location) > 64:
        errors.append(f"Invalid location: must be a non-empty string, max 64 chars.")

    # ── expected_range validation ────────────────────────────────────────
    expected_range = data["expected_range"]
    if (
        not isinstance(expected_range, list)
        or len(expected_range) != 2
        or not all(isinstance(v, (int, float)) for v in expected_range)
    ):
        errors.append("Invalid expected_range: must be [min, max] with two numeric values.")
    elif expected_range[0] >= expected_range[1]:
        errors.append(
            f"Invalid expected_range: min ({expected_range[0]}) must be less than max ({expected_range[1]})."
        )

    return {"valid": len(errors) == 0, "errors": errors, "data": data}


def compute_anomaly_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """
    Compute anomaly deviation metrics for a validated telemetry reading.

    Args:
        data: A validated telemetry dictionary.

    Returns:
        A dict with anomaly metrics: is_anomaly, deviation_value, deviation_percent,
        direction ('above'/'below'/'normal'), severity, and a human-readable summary.
    """
    value = data["value"]
    range_min, range_max = data["expected_range"]
    range_span = range_max - range_min
    sensor_type = data["sensor_type"]

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
            "summary": f"Sensor {data['sensor_id']} at {data['location']}: value {value} {data['unit']} is within normal range [{range_min}, {range_max}].",
        }

    # ── Calculate percentage deviation relative to range span ────────────
    deviation_percent = (deviation_value / range_span * 100) if range_span > 0 else 100.0

    # ── Determine severity ───────────────────────────────────────────────
    # Gas level anomalies are ALWAYS critical per the guardrail rules
    if sensor_type == "gas_level":
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
        f"value {value} {data['unit']} {direction_word} {boundary} {data['unit']} "
        f"(deviation: {deviation_value:.1f} {data['unit']}, {deviation_percent:.1f}% of range span). "
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

    Args:
        data: Raw telemetry dictionary from the API request.

    Returns:
        A dict with 'validation' results and 'anomaly' metrics (if valid).
    """
    # Step 1: Validate the input against the canonical schema
    validation = validate_telemetry(data)

    if not validation["valid"]:
        return {"validation": validation, "anomaly": None}

    # Step 2: Compute anomaly metrics for valid telemetry
    anomaly = compute_anomaly_metrics(data)

    return {"validation": validation, "anomaly": anomaly}
