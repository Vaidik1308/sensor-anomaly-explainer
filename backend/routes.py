"""
routes.py — FastAPI Route Definitions for the Sensor Anomaly Explanation API

Endpoints:
  POST /explain  — Accept telemetry JSON (ExplainRequest), return SSE stream
  POST /upload   — Accept a JSON file upload with telemetry data
  GET  /health   — System health check (checks GENAI_API_KEY env var)
  GET  /schema   — Return the canonical sensor schema with sample inputs

Schema changes (v2):
  - expected_range [min, max] replaced by normal_range_min / normal_range_max
  - sensor_type enum: TEMP, PRESSURE, HUMIDITY, VIBRATION, CURRENT, FLOW, GAS, SMOKE
  - units: °C, PSI, %RH, mm/s, Amps, L/min, PPM
"""

import json
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from anomaly_processor import (
    ALLOWED_SENSOR_TYPES,
    process_telemetry,
    validate_telemetry,
    compute_anomaly_metrics,
)
from rag_engine import generate_explanation_stream, generate_follow_up_suggestions

# ── Router ───────────────────────────────────────────────────────────────────
router = APIRouter()


# ── Request/Response Models ──────────────────────────────────────────────────

class TelemetryInput(BaseModel):
    """
    Pydantic model for telemetry input — matches the canonical sensor schema.

    Fields:
      sensor_id         — Unique identifier for the sensor (e.g. "TEMP-3621")
      sensor_type       — One of TEMP, PRESSURE, HUMIDITY, VIBRATION, CURRENT,
                          FLOW, GAS, SMOKE
      value             — The measured numeric value
      unit              — Unit of measurement (°C, PSI, %RH, mm/s, Amps, L/min, PPM)
      timestamp         — ISO 8601 timestamp of the reading
      location          — Human-readable location string
      normal_range_min  — Lower bound of expected normal range
      normal_range_max  — Upper bound of expected normal range
    """
    sensor_id: str
    sensor_type: str
    value: float | int
    unit: str
    timestamp: str
    location: str
    normal_range_min: float | int
    normal_range_max: float | int


class ExplainRequest(BaseModel):
    """
    Request body for the /explain endpoint.

    Fields:
      telemetry — A single TelemetryInput reading
      query     — Optional user question about the anomaly
      history   — Optional conversation history for follow-up context
    """
    telemetry: TelemetryInput
    query: str = ""
    history: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _convert_to_processor_format(telemetry_dict: dict) -> dict:
    """
    Convert the new schema (normal_range_min / normal_range_max) into the
    expected_range [min, max] format that anomaly_processor.process_telemetry
    expects.  Returns a *new* dict — the original is not mutated.
    """
    converted = dict(telemetry_dict)
    converted["expected_range"] = [
        converted.pop("normal_range_min"),
        converted.pop("normal_range_max"),
    ]
    return converted


# ── POST /explain — Main anomaly explanation endpoint ────────────────────────
@router.post("/explain")
async def explain_anomaly(request: ExplainRequest):
    """
    Accept a telemetry reading, validate it, retrieve context from the
    knowledge base, and stream a natural language anomaly explanation.

    Returns a Server-Sent Events (SSE) stream for real-time token output.
    """
    # Dump the Pydantic model to a plain dict
    telemetry_dict = request.telemetry.model_dump()

    # Convert normal_range_min / normal_range_max → expected_range for the processor
    processor_dict = _convert_to_processor_format(telemetry_dict)

    # Step 1: Validate and process the telemetry
    result = process_telemetry(processor_dict)

    if not result["validation"]["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Telemetry validation failed. Please check your input.",
                "errors": result["validation"]["errors"],
            },
        )

    anomaly_metrics = result["anomaly"]

    # Step 2: Generate follow-up suggestions (returned in the first SSE event)
    follow_ups = generate_follow_up_suggestions(processor_dict, anomaly_metrics)

    # Step 3: Stream the LLM explanation via SSE
    async def event_stream():
        # First event: metadata (anomaly metrics + follow-up suggestions)
        metadata = json.dumps({
            "type": "metadata",
            "anomaly": anomaly_metrics,
            "follow_ups": follow_ups,
        })
        yield f"data: {metadata}\n\n"

        # Subsequent events: streamed explanation tokens
        async for chunk in generate_explanation_stream(
            telemetry=processor_dict,
            anomaly_metrics=anomaly_metrics,
            history=request.history,
            query=request.query,
        ):
            # Each token is sent as a separate SSE data line
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        # Final event: signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for SSE
        },
    )


# ── POST /upload — File upload endpoint ──────────────────────────────────────
@router.post("/upload")
async def upload_telemetry(file: UploadFile = File(...)):
    """
    Accept a JSON file containing a single telemetry reading or an array of
    readings.  Returns validation results and anomaly metrics for each reading.

    The uploaded JSON should use the new schema (normal_range_min /
    normal_range_max).  Legacy expected_range format is also accepted.
    """
    # Read and parse the uploaded file
    try:
        contents = await file.read()
        data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

    # Normalise to a list of readings
    if isinstance(data, dict):
        readings = [data]
    elif isinstance(data, list):
        readings = data
    else:
        raise HTTPException(status_code=400, detail="Expected a JSON object or array.")

    # Validate and process each reading
    results = []
    for i, reading in enumerate(readings):
        # Convert new-schema fields if present
        if "normal_range_min" in reading and "normal_range_max" in reading:
            reading = _convert_to_processor_format(reading)

        result = process_telemetry(reading)
        results.append({
            "index": i,
            "sensor_id": reading.get("sensor_id", "unknown"),
            "validation": result["validation"]["valid"],
            "errors": result["validation"]["errors"],
            "anomaly": result["anomaly"],
        })

    return {"count": len(results), "results": results}


# ── GET /health — System health check ───────────────────────────────────────
@router.get("/health")
async def health_check():
    """
    Return system health status including knowledge base availability and
    whether the GENAI_API_KEY environment variable is configured.
    """
    import os
    from pathlib import Path

    # Check knowledge base files exist
    kb_path = Path(__file__).resolve().parent.parent / "knowledge_base"
    faiss_exists = (kb_path / "faiss_index.bin").exists()
    chunks_exist = (kb_path / "chunks.json").exists()

    # Check for the GenAI API key (not ANTHROPIC_API_KEY)
    api_key_set = bool(os.environ.get("GENAI_API_KEY"))

    return {
        "status": "healthy" if (faiss_exists and chunks_exist and api_key_set) else "degraded",
        "knowledge_base_indexed": faiss_exists and chunks_exist,
        "api_key_configured": api_key_set,
        "supported_sensors": list(ALLOWED_SENSOR_TYPES.keys()),
    }


# ── GET /schema — Return canonical sensor schema ────────────────────────────
@router.get("/schema")
async def get_schema():
    """
    Return the canonical sensor schema so the frontend can validate inputs.
    Includes field descriptions and 8 representative sample anomaly inputs.
    """
    schema = {
        "sensor_types": {
            name: {
                "unit": info["unit"],
                "description": info["description"],
            }
            for name, info in ALLOWED_SENSOR_TYPES.items()
        },
        "fields": {
            "sensor_id": "String identifier for the sensor (e.g. TEMP-3621)",
            "sensor_type": "One of: " + ", ".join(sorted(ALLOWED_SENSOR_TYPES.keys())),
            "value": "Numeric value (int or float)",
            "unit": "Must match sensor_type — one of: °C, PSI, %RH, mm/s, Amps, L/min, PPM",
            "timestamp": "ISO 8601 format (e.g. 2024-01-26T19:00:00Z)",
            "location": "Human-readable location string",
            "normal_range_min": "Lower bound of the normal operating range (number)",
            "normal_range_max": "Upper bound of the normal operating range (number)",
        },
        "sample_inputs": _get_sample_inputs(),
    }
    return schema


# ── Sample Inputs ────────────────────────────────────────────────────────────

def _get_sample_inputs() -> list[dict[str, Any]]:
    """
    Return 8 canonical sample anomaly inputs covering every supported sensor
    type.  Each sample represents a realistic anomaly scenario with a value
    well outside the normal operating range.
    """
    return [
        # 1. TEMP — Critical Overheating
        {
            "sensor_id": "TEMP-3621",
            "sensor_type": "TEMP",
            "value": 86.89,
            "unit": "°C",
            "timestamp": "2024-01-26T19:00:00Z",
            "location": "Warehouse Section D",
            "normal_range_min": 18,
            "normal_range_max": 35,
        },
        # 2. PRESSURE — Overpressure Event
        {
            "sensor_id": "PRESSURE-4710",
            "sensor_type": "PRESSURE",
            "value": 247.11,
            "unit": "PSI",
            "timestamp": "2024-01-26T19:05:00Z",
            "location": "Underground Pipeline Monitor",
            "normal_range_min": 30,
            "normal_range_max": 90,
        },
        # 3. HUMIDITY — Excess Humidity Alert
        {
            "sensor_id": "HUMIDITY-2290",
            "sensor_type": "HUMIDITY",
            "value": 94.82,
            "unit": "%RH",
            "timestamp": "2024-01-26T19:10:00Z",
            "location": "Underground Pipeline Monitor",
            "normal_range_min": 30,
            "normal_range_max": 65,
        },
        # 4. VIBRATION — Abnormal Vibration
        {
            "sensor_id": "VIBRATION-8812",
            "sensor_type": "VIBRATION",
            "value": 26.13,
            "unit": "mm/s",
            "timestamp": "2024-01-26T19:15:00Z",
            "location": "Warehouse Section D",
            "normal_range_min": 0,
            "normal_range_max": 5,
        },
        # 5. CURRENT — Overcurrent Fault
        {
            "sensor_id": "CURRENT-5501",
            "sensor_type": "CURRENT",
            "value": 158.04,
            "unit": "Amps",
            "timestamp": "2024-01-26T19:20:00Z",
            "location": "Underground Pipeline Monitor",
            "normal_range_min": 5,
            "normal_range_max": 50,
        },
        # 6. FLOW — Zero Flow / Blockage
        {
            "sensor_id": "FLOW-3345",
            "sensor_type": "FLOW",
            "value": 0.15,
            "unit": "L/min",
            "timestamp": "2024-01-26T19:25:00Z",
            "location": "Underground Pipeline Monitor",
            "normal_range_min": 10,
            "normal_range_max": 100,
        },
        # 7. GAS — Hazardous Gas Leak
        {
            "sensor_id": "GAS-7789",
            "sensor_type": "GAS",
            "value": 121.17,
            "unit": "PPM",
            "timestamp": "2024-01-26T19:30:00Z",
            "location": "Warehouse Section C",
            "normal_range_min": 0,
            "normal_range_max": 5,
        },
        # 8. SMOKE — Smoke Concentration Spike
        {
            "sensor_id": "SMOKE-6610",
            "sensor_type": "SMOKE",
            "value": 110.7,
            "unit": "PPM",
            "timestamp": "2024-01-26T19:35:00Z",
            "location": "Boiler Room East",
            "normal_range_min": 0,
            "normal_range_max": 10,
        },
    ]
