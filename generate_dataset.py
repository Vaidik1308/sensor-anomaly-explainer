#!/usr/bin/env python3
"""
generate_dataset.py
~~~~~~~~~~~~~~~~~~~
Generates a synthetic marine / offshore-oil IoT anomaly knowledge base.

The output is a collection of detailed documents covering 7 sensor types,
each with 6 anomaly categories, plus variation documents to exceed 50 000 words.

No external dependencies -- pure Python stdlib.
"""

import json
import os
import random
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_TYPES = [
    "temperature",
    "pressure",
    "humidity",
    "vibration",
    "voltage",
    "flow_rate",
    "gas_level",
]

CATEGORIES = [
    "spike",
    "drift",
    "flatline",
    "oscillation",
    "threshold_breach",
    "cross_sensor",
]

EQUIPMENT = [
    "main engine",
    "auxiliary engine",
    "cargo pumps",
    "fuel pumps",
    "generators",
    "HVAC system",
    "bilge system",
]

SEVERITIES = ["Low", "Medium", "High", "Critical"]

# Regulatory references used throughout documents
REGULATIONS = [
    "SOLAS Chapter II-1 Regulation 26 (Machinery Installations)",
    "SOLAS Chapter II-2 (Fire Protection)",
    "MARPOL Annex VI Regulation 13 (NOx Emissions)",
    "MARPOL Annex I Regulation 14 (Oil Discharge Monitoring)",
    "ISM Code Section 10 (Maintenance of Ship and Equipment)",
    "ISM Code Section 9 (Non-Conformities and Corrective Actions)",
    "IEC 60092 (Electrical Installations in Ships)",
    "SOLAS Chapter III (Life-Saving Appliances)",
    "MARPOL Annex VI Regulation 14 (SOx Emissions)",
    "DNV GL Rules for Classification of Ships",
]

# ---------------------------------------------------------------------------
# Sensor-specific metadata: units, normal ranges, thresholds, locations
# ---------------------------------------------------------------------------

SENSOR_META = {
    "temperature": {
        "unit": "deg C",
        "normal_range": (60, 95),
        "warning": 105,
        "critical": 120,
        "locations": [
            "main engine cylinder head",
            "exhaust gas manifold",
            "turbocharger outlet",
            "lubricating oil cooler",
            "jacket cooling water outlet",
            "auxiliary engine block",
            "cargo pump motor housing",
            "generator stator winding",
            "HVAC chiller compressor",
            "bilge pump motor casing",
        ],
    },
    "pressure": {
        "unit": "bar",
        "normal_range": (2.5, 7.0),
        "warning": 8.5,
        "critical": 10.0,
        "locations": [
            "main engine lube oil system",
            "fuel injection rail",
            "hydraulic cargo pump manifold",
            "cooling water circuit",
            "compressed air starting system",
            "auxiliary engine fuel supply",
            "fire main line",
            "bilge pump discharge",
            "steam drum boiler",
            "stern tube seal",
        ],
    },
    "humidity": {
        "unit": "% RH",
        "normal_range": (30, 65),
        "warning": 80,
        "critical": 90,
        "locations": [
            "engine control room",
            "cargo hold ventilation duct",
            "switchboard room",
            "accommodation HVAC return",
            "battery room",
            "paint store",
            "provision cold room",
            "steering gear compartment",
            "emergency generator room",
            "purifier room",
        ],
    },
    "vibration": {
        "unit": "mm/s RMS",
        "normal_range": (1.0, 4.5),
        "warning": 7.0,
        "critical": 11.0,
        "locations": [
            "main engine bearing pedestal",
            "turbocharger casing",
            "propeller shaft bearing",
            "auxiliary engine foundation",
            "cargo pump impeller housing",
            "generator bearing DE",
            "generator bearing NDE",
            "HVAC fan assembly",
            "fuel transfer pump base",
            "bilge pump mounting frame",
        ],
    },
    "voltage": {
        "unit": "V",
        "normal_range": (380, 440),
        "warning": 460,
        "critical": 480,
        "locations": [
            "main switchboard bus bar",
            "emergency switchboard",
            "shore connection panel",
            "generator output terminal",
            "cargo pump motor starter",
            "bow thruster drive panel",
            "battery charger output",
            "UPS system output",
            "steering gear motor feeder",
            "engine room lighting circuit",
        ],
    },
    "flow_rate": {
        "unit": "m3/h",
        "normal_range": (15, 80),
        "warning": 95,
        "critical": 110,
        "locations": [
            "main engine fuel supply line",
            "cooling water main circuit",
            "lube oil circulation pump",
            "ballast water transfer line",
            "cargo discharge manifold",
            "bilge water separator inlet",
            "boiler feed water line",
            "fire fighting main line",
            "fresh water generator feed",
            "fuel oil transfer line",
        ],
    },
    "gas_level": {
        "unit": "% LEL",
        "normal_range": (0, 10),
        "warning": 20,
        "critical": 40,
        "locations": [
            "cargo tank ullage space",
            "pump room atmosphere",
            "engine room bilge well",
            "cofferdam void space",
            "forecastle store",
            "double bottom void",
            "pipe tunnel",
            "paint locker exhaust",
            "battery room ventilation",
            "accommodation galley hood",
        ],
    },
}

# ---------------------------------------------------------------------------
# Root cause pools
# ---------------------------------------------------------------------------

ROOT_CAUSES = {
    "hardware_fault": [
        "Faulty thermocouple junction causing intermittent open-circuit readings",
        "Corroded sensor wiring due to prolonged saltwater spray exposure",
        "Cracked sensing element from mechanical shock during heavy weather",
        "Degraded signal cable shielding causing electromagnetic interference",
        "Failed analog-to-digital converter in the sensor transmitter module",
        "Loose terminal connection at the junction box vibrating free over time",
        "Membrane rupture in pressure transducer from water-hammer event",
        "Bearing wear in flowmeter turbine causing drag and under-reading",
        "Oxidised contact pins in sensor connector from tropical humidity",
        "Internal capacitor failure in the sensor's signal conditioning board",
    ],
    "environmental": [
        "Extreme sea state (Beaufort 9+) inducing vibration coupling into sensor mounts",
        "Tropical ambient temperature exceeding 45 deg C in unventilated compartments",
        "High salinity atmosphere accelerating galvanic corrosion on sensor housings",
        "Condensation build-up inside cable glands during rapid cooling in northern waters",
        "Engine room heat soak after prolonged full-ahead maneuvering in port approaches",
        "Lightning strike inducing transient voltages on unprotected signal loops",
        "Heavy rain ingress through damaged deck penetrations flooding junction boxes",
        "Biofouling on submerged flow sensor probes reducing effective cross-section",
        "Ice accretion on exposed deck sensors in Arctic transit conditions",
        "Solar radiation heating exposed sensor housings beyond calibration range",
    ],
    "calibration": [
        "Sensor last calibrated 18 months ago, exceeding OEM 12-month interval",
        "Incorrect zero-point offset entered during last dry-dock calibration",
        "Calibration performed at ambient 20 deg C but sensor operates at 70 deg C",
        "Reference standard used during calibration found to be out of tolerance",
        "Span adjustment drift due to ageing reference resistor in transmitter",
        "Linearisation table not updated after sensor element replacement",
        "Calibration certificate mismatch -- wrong sensor serial number recorded",
        "HART configuration overwritten during software update, resetting range",
        "Calibration gas cylinder past expiry date, concentration uncertain",
        "Two-point calibration used where five-point was required by class",
    ],
    "operational": [
        "Rapid load change on main engine during crash-stop maneuver",
        "Simultaneous start of multiple cargo pumps exceeding rated capacity",
        "Operator bypassed interlock to maintain production during sensor alarm",
        "Fuel changeover from HFO to LSFO causing transient combustion instability",
        "Ballast exchange in progress creating suction-side pressure transients",
        "Generator paralleling mismatch causing voltage and frequency swings",
        "Cargo heating system set to maximum during cold-weather discharge",
        "Emergency fire pump test run pressurizing the fire main unexpectedly",
        "Shore power connection with phase-sequence mismatch causing voltage spikes",
        "Bilge pump cycling on high-level alarm due to stern tube seal weepage",
    ],
}

# ---------------------------------------------------------------------------
# Historical resolution templates
# ---------------------------------------------------------------------------

HISTORICAL_RESOLUTIONS = [
    (
        "MV Pacific Carrier, 2022",
        "The vessel experienced a similar anomaly during monsoon transit in the Indian Ocean. "
        "The engineering team isolated the affected circuit, replaced the sensor transmitter, "
        "and performed a full loop calibration. Downtime was limited to 4 hours and the "
        "classification society accepted a remote survey for the interim period.",
    ),
    (
        "FPSO Northern Spirit, 2021",
        "During a scheduled turnaround on the FPSO, operators noticed identical readings. "
        "Root cause analysis traced the fault to a corroded cable tray in the splash zone. "
        "All signal cables in the zone were re-routed through a sealed conduit. Total repair "
        "cost was USD 12 000 and no production was lost because redundant sensors were available.",
    ),
    (
        "OSV Gulf Responder, 2023",
        "The offshore supply vessel recorded the anomaly while operating in the Gulf of Mexico. "
        "A portable calibration rig was helicoptered to the vessel and the sensor was re-calibrated "
        "in-situ. The flag state accepted a condition-of-class notation pending next scheduled dry-dock.",
    ),
    (
        "Tanker Coral Sea, 2020",
        "During cargo discharge at a single-point mooring, the same pattern appeared. Investigation "
        "revealed that the root cause was operational: a cargo valve had been left in a partially open "
        "position, creating turbulence at the sensor location. Closing the valve fully and reopening it "
        "in the correct sequence resolved the anomaly. Lessons learned were added to the vessel SMS.",
    ),
    (
        "Bulk Carrier Iron Duke, 2023",
        "While loading iron ore at Port Hedland, the monitoring system flagged identical behaviour. "
        "The shore-side loading arm vibrations coupled into the hull, exciting a resonance frequency "
        "at the sensor mount. Adding a vibration isolation pad and re-torquing the sensor bracket "
        "eliminated the false alarm. This modification was subsequently fleet-rolled.",
    ),
    (
        "LNG Carrier Arctic Pioneer, 2021",
        "During laden passage through the Suez Canal, elevated readings triggered a manual alarm "
        "verification. The engineering officer confirmed the anomaly was caused by heat radiation "
        "from the adjacent boil-off gas compressor. A reflective heat shield was fabricated on board "
        "and installed within 6 hours, returning readings to normal limits.",
    ),
    (
        "Chemical Tanker Chem Voyager, 2022",
        "Post-discharge tank cleaning operations led to a comparable sensor event. Aggressive cleaning "
        "chemicals had contaminated the sensing element via a leaking gland. The sensor was replaced, "
        "and the gland specification was upgraded to a chemical-resistant type (Viton O-rings).",
    ),
    (
        "Offshore Jack-Up Rig Titan III, 2019",
        "During pre-drilling commissioning, the BOP control system showed the anomaly. The "
        "integrated control system vendor dispatched a technician who identified a firmware "
        "mismatch between the sensor module and the SCADA gateway. A firmware flash resolved "
        "the issue and the rig was operational within 8 hours of first detection.",
    ),
    (
        "Container Vessel MSC Orient, 2023",
        "A reefer monitoring alert during transpacific passage revealed the same signature. "
        "Seventeen reefer containers were powered from the same bus section. Redistributing "
        "the reefer plugs across three bus sections reduced harmonic distortion and cleared "
        "the sensor anomaly. The electrical officer documented the procedure in the PMS.",
    ),
    (
        "Cruise Ship Horizon Star, 2022",
        "Guest-comfort HVAC sensors triggered the anomaly while the vessel transited from "
        "the Caribbean to Northern Europe. The drastic ambient temperature change exceeded "
        "the HVAC controller's adaptive algorithm range. A software parameter adjustment "
        "by the HVAC OEM's remote support team corrected the issue without any hardware change.",
    ),
]

# ---------------------------------------------------------------------------
# Document generation helpers
# ---------------------------------------------------------------------------


def _pick(lst, n=1):
    """Return n random items from lst (without replacement if possible)."""
    n = min(n, len(lst))
    return random.sample(lst, n)


def _rand_value(meta, mode="normal"):
    """Generate a realistic telemetry value based on the mode."""
    lo, hi = meta["normal_range"]
    if mode == "normal":
        return round(random.uniform(lo, hi), 1)
    elif mode == "spike_high":
        return round(random.uniform(meta["critical"], meta["critical"] * 1.5 + 10), 1)
    elif mode == "spike_low":
        return round(random.uniform(lo * 0.3, lo * 0.7), 1)
    elif mode == "warning":
        return round(random.uniform(meta["warning"], meta["critical"]), 1)
    elif mode == "drift":
        # a value that's been slowly climbing
        return round(random.uniform(hi, meta["warning"]), 1)
    elif mode == "flatline":
        # stuck at a single value
        return round(random.uniform(lo, hi), 1)
    elif mode == "oscillation_low":
        return round(random.uniform(lo * 0.5, lo), 1)
    elif mode == "oscillation_high":
        return round(random.uniform(hi, meta["critical"]), 1)
    return round(random.uniform(lo, hi), 1)


def _severity_justification(severity, sensor_type, equipment):
    """Return a marine-specific justification for the severity level."""
    justifications = {
        "Low": (
            f"This {sensor_type} anomaly on the {equipment} is classified as Low severity "
            f"because it remains within the manufacturer's extended operating tolerance and "
            f"does not immediately affect vessel seaworthiness. The watch engineer should "
            f"log the observation in the engine room log book and monitor the trend over "
            f"the next 4-hour watch period. No voyage disruption is anticipated."
        ),
        "Medium": (
            f"This {sensor_type} anomaly on the {equipment} is classified as Medium severity "
            f"because the readings have entered the cautionary band defined by the equipment "
            f"OEM. While the vessel can continue normal operations, the Chief Engineer should "
            f"be notified and a maintenance work order raised. Under ISM Code requirements, "
            f"the deviation must be recorded and tracked to closure."
        ),
        "High": (
            f"This {sensor_type} anomaly on the {equipment} is classified as High severity "
            f"because continued operation at these levels risks secondary damage to connected "
            f"systems. SOLAS requires that essential machinery remains fully operational; "
            f"therefore, the Master should be informed, a risk assessment conducted, and "
            f"contingency plans prepared including possible speed reduction or port diversion."
        ),
        "Critical": (
            f"This {sensor_type} anomaly on the {equipment} is classified as Critical severity "
            f"because the measured values indicate an imminent risk of equipment failure, "
            f"potential fire, or environmental release. Under SOLAS Chapter II-1 and the ISM "
            f"Code, the Master must be informed immediately, the affected equipment should be "
            f"shut down if safe to do so, and the company Designated Person Ashore (DPA) "
            f"must be notified. Depending on the vessel's location, the nearest Coast Guard "
            f"or MRCC should be alerted."
        ),
    }
    return justifications[severity]


def _operator_actions(category, sensor_type, equipment, meta):
    """Generate step-by-step operator actions tailored to the anomaly category."""
    # Common opening steps
    steps = [
        f"Step 1: Acknowledge the {sensor_type} alarm on the Alarm Monitoring System (AMS) and "
        f"silence the audible alarm. Record the timestamp, current reading ({_rand_value(meta, 'warning')} {meta['unit']}), "
        f"and the sensor tag number in the engine room log book.",
        f"Step 2: Visually inspect the {equipment} and surrounding area for obvious signs of "
        f"abnormality such as leaks, smoke, unusual noise, or discolouration.",
    ]

    # Category-specific middle steps
    if category == "spike":
        steps.append(
            f"Step 3: Compare the current {sensor_type} reading against the redundant sensor "
            f"(if installed) or use a portable calibrated instrument to verify the spike is genuine. "
            f"If the portable instrument reads normal ({_rand_value(meta, 'normal')} {meta['unit']}), suspect a sensor fault."
        )
        steps.append(
            f"Step 4: If the spike is confirmed genuine, reduce load on the {equipment} by 25% "
            f"and observe whether the reading returns to the normal band "
            f"({meta['normal_range'][0]}-{meta['normal_range'][1]} {meta['unit']}) within 10 minutes."
        )
    elif category == "drift":
        steps.append(
            f"Step 3: Review the {sensor_type} trend graph for the past 24 hours on the AMS. "
            f"Calculate the drift rate (change per hour). If the rate exceeds "
            f"{round((meta['warning'] - meta['normal_range'][1]) / 24, 2)} {meta['unit']}/hr, "
            f"escalate to the Chief Engineer."
        )
        steps.append(
            f"Step 4: Check related process parameters (e.g., flow rates, pressures, ambient "
            f"conditions) to determine if the drift correlates with a process change rather than "
            f"a sensor fault."
        )
    elif category == "flatline":
        steps.append(
            f"Step 3: Gently tap the sensor housing and observe whether the reading changes. "
            f"Check the signal cable for continuity using a multimeter at the junction box. "
            f"A reading of 4.00 mA exactly on a 4-20 mA loop strongly indicates a wiring fault "
            f"or failed transmitter."
        )
        steps.append(
            f"Step 4: If the sensor is confirmed dead, switch to the backup sensor if available. "
            f"If no backup exists, install a portable {sensor_type} gauge at the measurement point "
            f"and implement manual logging at 30-minute intervals."
        )
    elif category == "oscillation":
        steps.append(
            f"Step 3: Check the {sensor_type} reading update rate. If oscillations are faster "
            f"than the physical process can change (e.g., swinging {_rand_value(meta, 'oscillation_low')}-"
            f"{_rand_value(meta, 'oscillation_high')} {meta['unit']} within seconds), the cause is likely "
            f"electrical noise or a loose connection rather than a genuine process oscillation."
        )
        steps.append(
            f"Step 4: Inspect the sensor cable route for proximity to high-current cables, "
            f"VFD drives, or welding operations. Re-route or add shielding if interference is found."
        )
    elif category == "threshold_breach":
        steps.append(
            f"Step 3: Verify the alarm set-points in the AMS configuration against the OEM manual "
            f"and the approved Safety Management System thresholds. Confirm that the threshold has "
            f"not been inadvertently changed during a recent software update."
        )
        steps.append(
            f"Step 4: If the breach is genuine and the reading is {_rand_value(meta, 'spike_high')} {meta['unit']}, "
            f"prepare to shut down the {equipment} following the emergency shutdown procedure "
            f"posted in the engine room."
        )
    elif category == "cross_sensor":
        steps.append(
            f"Step 3: Open the AMS multi-trend display and overlay the {sensor_type} reading "
            f"with related sensor channels (e.g., temperature, pressure, vibration on the same "
            f"equipment). Look for correlated deviations that indicate a common root cause."
        )
        steps.append(
            f"Step 4: If multiple sensors on the {equipment} are deviating simultaneously, this "
            f"strongly suggests a genuine equipment problem rather than a sensor fault. Initiate "
            f"the equipment-specific emergency response plan."
        )

    # Common closing steps
    steps.append(
        f"Step 5: Inform the Officer of the Watch (OOW) on the bridge of the situation and any "
        f"potential impact on vessel manoeuvrability or speed."
    )
    steps.append(
        f"Step 6: Create a non-conformity report in the vessel's Safety Management System as "
        f"required by ISM Code Section 9. Attach the AMS trend screenshot and any photos taken "
        f"during the inspection."
    )
    steps.append(
        f"Step 7: If the anomaly is resolved, reset the alarm, record the resolution in the log "
        f"book, and brief the oncoming watch. If unresolved, hand over with a clear status update "
        f"and any interim risk mitigation measures in place."
    )
    return " ".join(steps)


# ---------------------------------------------------------------------------
# Core document builder
# ---------------------------------------------------------------------------


def build_document(sensor_type, category, variation_index=0):
    """
    Build a single knowledge-base document for a given sensor type and
    anomaly category.  variation_index > 0 produces alternative scenarios.

    Each document targets 250-400 words to keep content dense but concise.
    """
    meta = SENSOR_META[sensor_type]
    equip = random.choice(EQUIPMENT)
    location = random.choice(meta["locations"])
    severity = random.choice(SEVERITIES)
    cause_key = random.choice(list(ROOT_CAUSES.keys()))
    cause_text = random.choice(ROOT_CAUSES[cause_key])
    reg = random.choice(REGULATIONS)
    resolution = random.choice(HISTORICAL_RESOLUTIONS)

    # --- Title ---
    category_labels = {
        "spike": "Sudden Spike",
        "drift": "Gradual Drift",
        "flatline": "Sensor Flatline / Dropout",
        "oscillation": "Erratic Oscillation",
        "threshold_breach": "Threshold Breach",
        "cross_sensor": "Cross-Sensor Correlation Anomaly",
    }
    var_suffix = f" (Variant {variation_index})" if variation_index > 0 else ""
    title = (
        f"{category_labels[category]} in {sensor_type.replace('_', ' ').title()} "
        f"Sensor at {location}{var_suffix}"
    )

    # --- Telemetry snippet ---
    if category == "spike":
        vals = [_rand_value(meta, "normal")] * 3 + [_rand_value(meta, "spike_high")] + [_rand_value(meta, "normal")] * 2
    elif category == "drift":
        base = meta["normal_range"][0]
        step = (meta["warning"] - base) / 5
        vals = [round(base + step * i + random.uniform(-0.5, 0.5), 1) for i in range(6)]
    elif category == "flatline":
        stuck = _rand_value(meta, "flatline")
        vals = [stuck] * 6
    elif category == "oscillation":
        vals = [
            _rand_value(meta, random.choice(["oscillation_low", "oscillation_high"]))
            for _ in range(6)
        ]
    elif category == "threshold_breach":
        vals = [_rand_value(meta, "normal")] * 4 + [_rand_value(meta, "spike_high")] + [_rand_value(meta, "normal")]
    else:  # cross_sensor
        vals = [_rand_value(meta, "drift")] * 6

    telemetry_str = ", ".join(f"{v} {meta['unit']}" for v in vals)

    # --- Pick affected equipment ---
    affected = _pick(EQUIPMENT, random.randint(2, 3))
    affected_str = ", ".join(affected)

    # --- Severity justification (compact) ---
    sev_just = _severity_justification(severity, sensor_type.replace("_", " "), equip)

    # --- Build compact operator action steps ---
    action_steps = _operator_actions(category, sensor_type.replace("_", " "), equip, meta)

    # --- Compose the document as a single rich paragraph ---
    text = (
        f"A {category_labels[category].lower()} anomaly has been detected on the "
        f"{sensor_type.replace('_', ' ')} sensor installed at the {location} on the {equip}. "
        f"Telemetry over a 30-minute window: [{telemetry_str}]. Normal operating band: "
        f"{meta['normal_range'][0]}-{meta['normal_range'][1]} {meta['unit']}; warning at "
        f"{meta['warning']} {meta['unit']}; critical at {meta['critical']} {meta['unit']}. "
        f"Probable root cause ({cause_key.replace('_', ' ')}): {cause_text}. "
        f"{sev_just} "
        f"Equipment affected: {equip} (primary), with potential secondary impact on {affected_str}. "
        f"Regulatory reference: {reg}. "
        f"{action_steps} "
        f"Historical precedent ({resolution[0]}): {resolution[1]}"
    )

    return {
        "sensor_type": sensor_type,
        "category": category,
        "title": title,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def generate_knowledge_base(target_word_count=72000):
    """
    Generate documents until the target word count is exceeded.

    Returns a list of document dicts.
    """
    random.seed(42)  # reproducible output
    documents = []
    total_words = 0

    # Phase 1: Generate the 42 base documents (7 sensors x 6 categories)
    print("Phase 1: Generating 42 base documents (7 sensors x 6 categories)...")
    for sensor in SENSOR_TYPES:
        for cat in CATEGORIES:
            doc = build_document(sensor, cat, variation_index=0)
            documents.append(doc)
            wc = len(doc["text"].split())
            total_words += wc
        print(f"  Completed sensor: {sensor:15s} | Documents: {len(documents):4d} | Words: {total_words:,}")

    print(f"\nPhase 1 complete: {len(documents)} documents, {total_words:,} words.\n")

    # Phase 2: Generate variation documents until we exceed the target
    print(f"Phase 2: Generating variation documents to reach {target_word_count:,} words...")
    variation_round = 1
    while total_words < target_word_count:
        for sensor in SENSOR_TYPES:
            for cat in CATEGORIES:
                if total_words >= target_word_count:
                    break
                doc = build_document(sensor, cat, variation_index=variation_round)
                documents.append(doc)
                wc = len(doc["text"].split())
                total_words += wc
            if total_words >= target_word_count:
                break

        print(
            f"  Variation round {variation_round} complete | "
            f"Documents: {len(documents):4d} | Words: {total_words:,}"
        )
        variation_round += 1

    print(f"\nPhase 2 complete: {len(documents)} documents, {total_words:,} words.\n")
    return documents


def save_knowledge_base(documents):
    """
    Save the generated documents to the knowledge_base/ directory
    (relative to this script's location).
    """
    # Determine output directory relative to this script
    script_dir = Path(__file__).resolve().parent
    kb_dir = script_dir / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)

    # Save raw_documents.json
    output_path = kb_dir / "raw_documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(documents)} documents to {output_path}")

    # Print summary statistics
    word_count = sum(len(d["text"].split()) for d in documents)
    print(f"Total word count: {word_count:,}")
    print(f"Average words per document: {word_count // len(documents)}")

    # Per-sensor breakdown
    print("\nPer-sensor breakdown:")
    for sensor in SENSOR_TYPES:
        sensor_docs = [d for d in documents if d["sensor_type"] == sensor]
        sensor_words = sum(len(d["text"].split()) for d in sensor_docs)
        print(f"  {sensor:15s}: {len(sensor_docs):4d} docs, {sensor_words:,} words")

    # Per-category breakdown
    print("\nPer-category breakdown:")
    for cat in CATEGORIES:
        cat_docs = [d for d in documents if d["category"] == cat]
        cat_words = sum(len(d["text"].split()) for d in cat_docs)
        print(f"  {cat:20s}: {len(cat_docs):4d} docs, {cat_words:,} words")

    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Marine / Offshore IoT Anomaly Knowledge Base Generator")
    print("=" * 70)
    print()

    # Generate documents
    docs = generate_knowledge_base(target_word_count=72000)

    # Save to disk
    output = save_knowledge_base(docs)

    print()
    print("=" * 70)
    print("Generation complete.")
    print(f"Output file: {output}")
    print("=" * 70)
