"""
routes.py — FastAPI Route Definitions for the Sensor Anomaly Explanation API

Endpoints:
  POST /explain  — Accept telemetry JSON, return streamed anomaly explanation
  POST /upload   — Accept a JSON file upload with telemetry data
  GET  /health   — System health check
  GET  /schema   — Return the canonical sensor schema for frontend validation
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
    """Pydantic model for telemetry input — matches the canonical sensor schema."""
    sensor_id: str
    sensor_type: str
    value: float | int
    unit: str
    timestamp: str
    location: str
    expected_range: list[float | int]


class ExplainRequest(BaseModel):
    """Request body for the /explain endpoint."""
    telemetry: TelemetryInput
    query: str = ""
    history: str = ""


# ── POST /explain — Main anomaly explanation endpoint ────────────────────────
@router.post("/explain")
async def explain_anomaly(request: ExplainRequest):
    """
    Accept a telemetry reading, validate it, retrieve context from the
    knowledge base, and stream a natural language anomaly explanation.

    Returns a Server-Sent Events stream for real-time token output.
    """
    telemetry_dict = request.telemetry.model_dump()

    # Step 1: Validate and process the telemetry
    result = process_telemetry(telemetry_dict)

    if not result["validation"]["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Telemetry validation failed. Please check your input.",
                "errors": result["validation"]["errors"],
            },
        )

    anomaly_metrics = result["anomaly"]

    # Step 2: Generate follow-up suggestions (returned in headers/metadata)
    follow_ups = generate_follow_up_suggestions(telemetry_dict, anomaly_metrics)

    # Step 3: Stream the LLM explanation via SSE
    async def event_stream():
        # Send metadata as the first SSE event
        metadata = json.dumps({
            "type": "metadata",
            "anomaly": anomaly_metrics,
            "follow_ups": follow_ups,
        })
        yield f"data: {metadata}\n\n"

        # Stream the explanation tokens
        async for chunk in generate_explanation_stream(
            telemetry=telemetry_dict,
            anomaly_metrics=anomaly_metrics,
            history=request.history,
            query=request.query,
        ):
            # Escape newlines for SSE format
            escaped = chunk.replace("\n", "\\n")
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── POST /upload — File upload endpoint ──────────────────────────────────────
@router.post("/upload")
async def upload_telemetry(file: UploadFile = File(...)):
    """
    Accept a JSON file containing a single telemetry reading or an array of readings.
    Returns validation results and anomaly metrics for each reading.
    """
    # Read and parse the uploaded file
    try:
        contents = await file.read()
        data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

    # Handle single reading or array
    if isinstance(data, dict):
        readings = [data]
    elif isinstance(data, list):
        readings = data
    else:
        raise HTTPException(status_code=400, detail="Expected a JSON object or array.")

    # Validate and process each reading
    results = []
    for i, reading in enumerate(readings):
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
    """Return system health status including knowledge base and LLM availability."""
    import os
    from pathlib import Path

    kb_path = Path(__file__).resolve().parent.parent / "knowledge_base"
    faiss_exists = (kb_path / "faiss_index.bin").exists()
    chunks_exist = (kb_path / "chunks.json").exists()
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))

    return {
        "status": "healthy" if (faiss_exists and chunks_exist and api_key_set) else "degraded",
        "knowledge_base_indexed": faiss_exists and chunks_exist,
        "api_key_configured": api_key_set,
        "supported_sensors": list(ALLOWED_SENSOR_TYPES.keys()),
    }


# ── GET /schema — Return canonical sensor schema ────────────────────────────
@router.get("/schema")
async def get_schema():
    """Return the canonical sensor schema so the frontend can validate inputs."""
    schema = {
        "sensor_types": {
            name: {
                "unit": info["unit"],
                "description": info["description"],
            }
            for name, info in ALLOWED_SENSOR_TYPES.items()
        },
        "fields": {
            "sensor_id": "Alphanumeric + underscores, max 32 chars (e.g. TEMP_ENG_01)",
            "sensor_type": "One of: " + ", ".join(sorted(ALLOWED_SENSOR_TYPES.keys())),
            "value": "Numeric value (int or float)",
            "unit": "Must match sensor_type (see sensor_types above)",
            "timestamp": "ISO 8601 format (e.g. 2026-04-10T10:00:00Z)",
            "location": "String, max 64 chars (e.g. Engine Room, Fuel Pipeline A)",
            "expected_range": "[min, max] — two numbers where min < max",
        },
        "sample_inputs": _get_sample_inputs(),
    }
    return schema


def _get_sample_inputs() -> list[dict[str, Any]]:
    """Return the 7 canonical sample anomaly inputs."""
    return [
        {
            "sensor_id": "TEMP_ENG_01", "sensor_type": "temperature", "value": 92,
            "unit": "C", "timestamp": "2026-04-10T10:00:00Z",
            "location": "Engine Room", "expected_range": [40, 85],
        },
        {
            "sensor_id": "FLOW_FUEL_01", "sensor_type": "flow_rate", "value": 15,
            "unit": "L/min", "timestamp": "2026-04-10T10:00:10Z",
            "location": "Fuel Pipeline A", "expected_range": [30, 60],
        },
        {
            "sensor_id": "PRES_TANK_01", "sensor_type": "pressure", "value": 85000,
            "unit": "Pa", "timestamp": "2026-04-10T10:00:20Z",
            "location": "Fuel Tank B", "expected_range": [100000, 200000],
        },
        {
            "sensor_id": "HUM_CARGO_01", "sensor_type": "humidity", "value": 91,
            "unit": "%", "timestamp": "2026-04-10T10:00:30Z",
            "location": "Cargo Hold C", "expected_range": [30, 70],
        },
        {
            "sensor_id": "VIB_ENG_01", "sensor_type": "vibration", "value": 142,
            "unit": "Hz", "timestamp": "2026-04-10T10:00:40Z",
            "location": "Propulsion Shaft", "expected_range": [10, 80],
        },
        {
            "sensor_id": "VOLT_GEN_01", "sensor_type": "voltage", "value": 178,
            "unit": "V", "timestamp": "2026-04-10T10:00:50Z",
            "location": "Generator Room", "expected_range": [210, 240],
        },
        {
            "sensor_id": "GAS_ENG_01", "sensor_type": "gas_level", "value": 320,
            "unit": "ppm", "timestamp": "2026-04-10T10:01:00Z",
            "location": "Engine Room", "expected_range": [0, 50],
        },
    ]
