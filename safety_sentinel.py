import tempfile
from pathlib import Path
import base64
import cv2
import html
import math
import numpy as np
import pandas as pd
import streamlit as st
import struct
from ultralytics import YOLO
import time
import wave
from io import BytesIO

# --- CONFIGURATION ---
# Path to the trained YOLO model weights used by the PPE detector.
MODEL_PATH = "runs/detect/train9/weights/best.pt"

# Supported upload types for the batch-analysis tab.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# Minimum time between repeated audio alerts for the same critical class.
ALERT_COOLDOWN_SECONDS = 4.0

# Critical detection classes that should trigger a visible and audible alert.
# Each alert has a user-facing message and a short tone sequence:
# (frequency_in_hz, duration_in_seconds). A frequency of 0 means silence.
CRITICAL_ALERTS = {
    "no-hardhat": {
        "message": "Critical: worker without hardhat detected",
        "sequence": [(1046, 0.14), (0, 0.06), (1046, 0.14), (0, 0.06), (1318, 0.18)],
    },
    "no-vest": {
        "message": "Critical: worker without safety vest detected",
        "sequence": [(587, 0.22), (0, 0.07), (784, 0.22), (0, 0.07), (587, 0.22)],
    },
}

# Inline SVG used in the custom Streamlit header.
LOGO_SVG = """
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield-check">
    <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>
    <path d="m9 12 2 2 4-4"/>
</svg>
"""

# Configure the Streamlit page before rendering any app content.
st.set_page_config(
    page_title="Sentinel / Industrial PPE Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_modern_css():
    """Inject app-wide CSS so the default Streamlit UI looks like a custom dashboard."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
            
            :root {{
                --brand-bg: #0b0f1a;
                --card-bg: #161b22;
                --border: #30363d;
                --text-primary: #e6edf3;
                --text-secondary: #8b949e;
                --teal: #2ea043;
                --amber: #d29922;
                --red: #f85149;
            }}

            /* Main App Styling */
            .stApp {{
                background-color: var(--brand-bg);
                color: var(--text-primary);
                font-family: 'Inter', sans-serif;
            }}

            /* Custom Header */
            .header-container {{
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 1.5rem 0;
                border-bottom: 1px solid var(--border);
                margin-bottom: 2rem;
            }}

            .header-title {{
                font-size: 1.75rem;
                font-weight: 700;
                letter-spacing: -0.025em;
                margin: 0;
                color: var(--text-primary);
            }}

            .header-badge {{
                background: rgba(46, 160, 67, 0.15);
                color: var(--teal);
                border: 1px solid var(--teal);
                padding: 2px 10px;
                border-radius: 99px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}

            /* Metric Cards */
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 16px;
                margin-bottom: 2rem;
            }}

            .metric-card {{
                background: var(--card-bg);
                border: 1px solid var(--border);
                padding: 1.25rem;
                border-radius: 12px;
                transition: transform 0.2s ease;
            }}

            .metric-card:hover {{
                border-color: #58a6ff;
                transform: translateY(-2px);
            }}

            .metric-label {{
                font-size: 0.75rem;
                text-transform: uppercase;
                font-weight: 600;
                color: var(--text-secondary);
                letter-spacing: 0.05em;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 6px;
            }}

            .metric-value {{
                font-size: 2.25rem;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
                line-height: 1;
            }}

            /* Alerts */
            .alert-container {{
                background: rgba(248, 81, 73, 0.1);
                border-left: 4px solid var(--red);
                padding: 1rem;
                border-radius: 4px;
                margin: 1rem 0;
            }}

            .alert-title {{
                color: var(--red);
                font-weight: 700;
                font-size: 0.9rem;
                margin-bottom: 4px;
            }}

            .alert-body {{
                color: var(--text-primary);
                font-size: 0.85rem;
            }}

            /* Control Panel */
            .stSidebar {{
                background-color: var(--card-bg);
                border-right: 1px solid var(--border);
            }}

            .stSidebar h2 {{
                color: var(--text-primary);
                font-size: 1.2rem;
                padding-bottom: 1rem;
            }}

            /* Dataframes */
            [data-testid="stDataFrame"] {{
                border: 1px solid var(--border);
                border-radius: 8px;
                overflow: hidden;
            }}

            /* Hide Streamlit components to look more like a custom platform */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            header {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_modern_css()

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Load the YOLO model once and reuse it across Streamlit reruns."""
    if not Path(model_path).exists():
        return None
    return YOLO(model_path)

def normalize_class_name(name):
    """Convert model class names into stable lowercase keys such as 'no-hardhat'."""
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")

# --- DETECTION HELPERS ---
def get_detections(result):
    """Convert one YOLO result object into rows that can be displayed in the UI."""
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            # YOLO stores class id, confidence, and bounding box as tensor values.
            # Convert them to normal Python values for tables, counters, and JSON-like use.
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            detections.append({
                "Object": result.names[cls_id],
                "ClassKey": normalize_class_name(result.names[cls_id]),
                "Confidence": f"{conf:.2%}",
                "BBox": [int(x) for x in box.xyxy[0].tolist()]
            })
    return detections

def analyze_safety(detections):
    """Translate violation-oriented detection classes into readable PPE warnings."""
    violations = []

    # The trained model already has explicit violation classes, so this UI layer maps
    # those class names to messages instead of trying to infer PPE relationships itself.
    violation_pairs = {
        "no-hardhat": "Head protection missing",
        "no-vest": "High-visibility apparel missing",
        "no-mask": "Respiratory protection required",
        "no-gloves": "Hand protection missing"
    }
    
    for det in detections:
        cls = normalize_class_name(det["Object"])
        if cls in violation_pairs:
            violations.append(violation_pairs[cls])

    # Deduplicate messages because multiple detections of the same issue can appear
    # in a single frame or image.
    return sorted(list(set(violations)))

def get_critical_alert_types(detections):
    """Return the critical detection keys that should trigger audio alerts."""
    alert_types = []
    for det in detections:
        cls = normalize_class_name(det["Object"])
        if cls in CRITICAL_ALERTS:
            alert_types.append(cls)
    return sorted(set(alert_types))

def count_classes(detections):
    """Count normalized model classes for KPI cards and summary reporting."""
    counts = {}
    for det in detections:
        cls = normalize_class_name(det["Object"])
        counts[cls] = counts.get(cls, 0) + 1
    return counts

def process_frame(model, frame, conf_threshold):
    """Run YOLO on a single OpenCV frame and return annotated output plus summaries."""
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    result = results[0]

    # result.plot() returns an image with bounding boxes and labels drawn on it.
    annotated = result.plot()
    detections = get_detections(result)
    violations = analyze_safety(detections)
    return annotated, detections, violations

def open_camera(camera_index):
    """Try multiple Windows camera backends and return the first working capture device."""
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Default", None),
        ("Media Foundation", cv2.CAP_MSMF),
    ]

    for backend_name, backend in backends:
        cap = cv2.VideoCapture(camera_index) if backend is None else cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            # Request HD frames; the camera may still return a different supported size.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            return cap, backend_name
        cap.release()

    return None, None

@st.cache_data
def build_alert_audio(alert_type):
    """Generate a base64 WAV tone for one critical alert type."""
    sample_rate = 44100
    amplitude = 0.45
    audio = BytesIO()
    sequence = CRITICAL_ALERTS[alert_type]["sequence"]

    with wave.open(audio, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for frequency, duration in sequence:
            frame_count = int(sample_rate * duration)
            for i in range(frame_count):
                if frequency:
                    # Fade the start and end of each tone to avoid harsh clicks.
                    envelope = min(i / 400, 1.0, (frame_count - i) / 400)
                    sample = amplitude * envelope * math.sin(2 * math.pi * frequency * i / sample_rate)
                else:
                    sample = 0
                wav_file.writeframes(struct.pack("<h", int(sample * 32767)))

    return base64.b64encode(audio.getvalue()).decode("ascii")

def render_audio_alerts(alert_types, audio_placeholder):
    """Play hidden HTML audio elements for critical alerts while respecting cooldowns."""
    if "last_audio_alert_times" not in st.session_state:
        st.session_state.last_audio_alert_times = {}

    now = time.monotonic()
    audio_tags = []

    for alert_type in alert_types:
        last_alert = st.session_state.last_audio_alert_times.get(alert_type, 0)
        if now - last_alert < ALERT_COOLDOWN_SECONDS:
            # Skip repeated alerts that happen too soon after the previous one.
            continue

        st.session_state.last_audio_alert_times[alert_type] = now
        audio_b64 = build_alert_audio(alert_type)
        label = html.escape(CRITICAL_ALERTS[alert_type]["message"])
        audio_tags.append(
            f"""
            <audio autoplay aria-label="{label}">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
            """
        )

    if audio_tags:
        # Render audio tags in a zero-height container so they play without changing layout.
        audio_placeholder.markdown(
            f'<div style="height:0; overflow:hidden;">{"".join(audio_tags)}</div>',
            unsafe_allow_html=True,
        )

# --- UI COMPONENTS ---
def render_header():
    """Render the custom top header with logo, title, status badge, and node metadata."""
    st.markdown(
        f"""
        <div class="header-container">
            <div style="color: #58a6ff;">{LOGO_SVG}</div>
            <h1 class="header-title">SENTINEL_AI</h1>
            <span class="header-badge">Compliance Engine Live</span>
            <div style="margin-left: auto; font-family: 'JetBrains Mono'; font-size: 0.7rem; color: var(--text-secondary);">
                NODE_ID: AIS-LONDON-01 <br/>
                UPTIME: 14:02:44
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_kpis(counts, violation_count):
    """Render dashboard KPI cards from current detection counts."""
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Detected Personnel</div>
                <div class="metric-value" style="color: #58a6ff;">{counts.get('person', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">PPE Compliant</div>
                <div class="metric-value" style="color: var(--teal);">{counts.get('hardhat', 0) + counts.get('vest', 0)}</div>
            </div>
            <div class="metric-card" style="border-color: {'var(--red)' if violation_count > 0 else 'var(--border)'}">
                <div class="metric-label">Active Violations</div>
                <div class="metric-value" style="color: var(--red);">{violation_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Detection Confidence</div>
                <div class="metric-value">0.94</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_alert_panel(violations, critical_alert_types):
    """Render the current visible safety-alert panel."""
    st.markdown("### Safety Alerts")
    if not violations:
        st.success("No active PPE violations detected.")
        return

    alert_lines = []

    # Critical alerts are tracked by class key; add their configured messages to
    # the same panel as the general PPE violation messages.
    critical_messages = {
        CRITICAL_ALERTS[alert_type]["message"]
        for alert_type in critical_alert_types
    }

    for message in sorted(set(violations).union(critical_messages)):
        alert_lines.append(f"<div>{html.escape(message)}</div>")

    st.markdown(
        f"""
        <div class="alert-container">
            <div class="alert-title">CRITICAL PPE VIOLATION</div>
            <div class="alert-body">{"".join(alert_lines)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_detection_table(detections):
    """Render a detection dataframe, or an empty-state message if no objects were found."""
    st.markdown("### Detection Log")
    if detections:
        st.dataframe(pd.DataFrame(detections), use_container_width=True)
    else:
        st.info("No objects detected.")

def process_uploaded_video(model, video_path, conf_threshold):
    """Analyze every frame of an uploaded video and write an annotated MP4 result."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, pd.DataFrame(), pd.DataFrame(), {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        return None, pd.DataFrame(), pd.DataFrame(), {}

    if fps <= 0:
        # Some files do not report FPS; use a practical playback default.
        fps = 20

    # Write the annotated video to a temporary file so Streamlit can play it back.
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    progress = st.progress(0, text="Analyzing uploaded video...")
    all_detections = []
    all_violations = []
    summary_counts = {}
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze this frame, draw boxes, and append it to the output video.
        annotated, detections, violations = process_frame(model, frame, conf_threshold)
        writer.write(annotated)

        for det in detections:
            # Add frame numbers so users can trace each detection back to the video.
            row = det.copy()
            row["Frame"] = frame_number
            all_detections.append(row)
            cls = normalize_class_name(det["Object"])
            summary_counts[cls] = summary_counts.get(cls, 0) + 1

        # Store critical alert entries separately so they can be displayed in the
        # violation log and used for audio playback after processing completes.
        for alert_type in get_critical_alert_types(detections):
            all_violations.append(
                {
                    "Frame": frame_number,
                    "Type": alert_type,
                    "Alert": CRITICAL_ALERTS[alert_type]["message"],
                }
            )

        # Store readable PPE violation messages for the same frame.
        for violation in violations:
            all_violations.append(
                {
                    "Frame": frame_number,
                    "Type": "ppe",
                    "Alert": violation,
                }
            )

        frame_number += 1
        if total_frames > 0:
            progress.progress(
                min(frame_number / total_frames, 1.0),
                text=f"Analyzing uploaded video... frame {frame_number}/{total_frames}",
            )

    cap.release()
    writer.release()
    progress.empty()

    # Drop duplicate rows because a critical class can also map to a general violation.
    violations_df = pd.DataFrame(all_violations).drop_duplicates()
    return output_path, pd.DataFrame(all_detections), violations_df, summary_counts

def render_uploaded_image(model, uploaded_file, conf_threshold, audio_alerts_enabled):
    """Decode an uploaded image, run detection, and render its analysis view."""
    media_bytes = uploaded_file.getvalue()

    # Convert the uploaded bytes into the BGR image format expected by OpenCV.
    file_bytes = np.frombuffer(media_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not read this image file.")
        return

    annotated, detections, violations = process_frame(model, image, conf_threshold)
    critical_alert_types = get_critical_alert_types(detections)
    counts = count_classes(detections)

    result_col, detail_col = st.columns([1.4, 1])
    with result_col:
        st.markdown("### Analyzed Image")
        # Streamlit expects RGB images, while OpenCV and YOLO annotations are BGR.
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    with detail_col:
        render_kpis(counts, len(violations))
        render_alert_panel(violations, critical_alert_types)
        if audio_alerts_enabled and critical_alert_types:
            render_audio_alerts(critical_alert_types, st.empty())

    render_detection_table(detections)

def render_uploaded_video(model, uploaded_file, conf_threshold, audio_alerts_enabled):
    """Save an uploaded video temporarily, analyze it, and render video-level results."""
    media_bytes = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"

    # OpenCV needs a filesystem path, so persist the uploaded bytes to a temp file.
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_video.write(media_bytes)
    temp_video.close()

    output_path, detections_df, violations_df, counts = process_uploaded_video(
        model,
        temp_video.name,
        conf_threshold,
    )

    if output_path is None:
        st.error("Could not process this video file.")
        return

    result_col, detail_col = st.columns([1.4, 1])
    with result_col:
        st.markdown("### Analyzed Video")
        st.video(output_path)

    with detail_col:
        render_kpis(counts, len(violations_df))

        # Pull the critical alert class names back out of the violation dataframe.
        critical_alert_types = sorted(
            {
                normalize_class_name(alert_type)
                for alert_type in violations_df.get("Type", pd.Series(dtype=str)).tolist()
                if normalize_class_name(alert_type) in CRITICAL_ALERTS
            }
        )
        render_alert_panel(violations_df["Alert"].dropna().unique().tolist() if not violations_df.empty else [], critical_alert_types)
        if audio_alerts_enabled and critical_alert_types:
            render_audio_alerts(critical_alert_types, st.empty())

    st.markdown("### Detection Log")
    if detections_df.empty:
        st.info("No objects detected in this video.")
    else:
        st.dataframe(detections_df, use_container_width=True)

    st.markdown("### Violation Log")
    if violations_df.empty:
        st.success("No PPE violations detected in this video.")
    else:
        st.dataframe(violations_df, use_container_width=True)

# --- MAIN APPLICATION ---
# Draw static chrome and load the model before building interactive controls.
render_header()

model = load_model(MODEL_PATH)

# Settings Sidebar
with st.sidebar:
    # User-tunable runtime controls shared by live and uploaded-media analysis.
    st.markdown("## Control Panel")
    conf_threshold = st.slider("Detection confidence", 0.1, 0.9, 0.45, 0.05)
    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    audio_alerts_enabled = st.checkbox("Enable audio alerts", value=True)
    st.divider()
    st.markdown("### System Diagnostics")
    if model:
        st.success("YOLO Engine: Optimized")
        st.caption(f"Weights: {Path(MODEL_PATH).name}")
    else:
        st.error("Engine Fault: Model Missing")
        st.info("Please verify the model path in configuration.")

# Navigation Tabs
# Live feed processes webcam frames; batch analysis handles uploaded images/videos.
tab_live, tab_media = st.tabs(["[ 1 ] LIVE FEED", "[ 2 ] BATCH ANALYSIS"])

with tab_live:
    col_vid, col_logs = st.columns([2, 1])
    
    with col_vid:
        # Left side of the live tab: the camera feed or an idle placeholder.
        st.markdown("### Primary Camera Feed")
        frame_placeholder = st.empty()
        run_live = st.toggle("Enable Optical Engine", value=False, key="run_live_detection")
        
        if run_live and model:
            st.info("Initializing camera stream...")
        else:
            frame_placeholder.image("https://images.unsplash.com/photo-1590644365607-1c5a519a7a37?auto=format&fit=crop&q=80&w=1200", caption="Engine Idle - Waiting for Input")

    with col_logs:
        # Right side of the live tab: KPI cards, alerts, audio, and detection table.
        st.markdown("### Telemetry Stream")
        metrics_placeholder = st.empty()
        alerts_placeholder = st.empty()
        audio_placeholder = st.empty()
        table_placeholder = st.empty()

        if not run_live:
            st.caption("No active signals detected.")
        elif not model:
            st.error("Model unavailable. Live detection cannot start.")
        else:
            st.caption("Reading frames from the active camera.")

    if run_live and model:
        cap, backend_name = open_camera(int(camera_index))

        if cap is None:
            st.error(
                "Could not open webcam. Try camera index 1, close other apps using the camera, "
                "or allow camera access for desktop apps in Windows privacy settings."
            )
        else:
            st.caption(f"Camera backend: {backend_name}")
            frames_this_run = 0

            # Streamlit reruns are single-threaded. Process a bounded batch of frames,
            # then rerun while the toggle remains on to keep the app responsive.
            max_frames_per_run = 90

            while frames_this_run < max_frames_per_run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Could not read a frame from the webcam.")
                    break

                # Analyze the current webcam frame and update every live placeholder.
                annotated, detections, violations = process_frame(model, frame, conf_threshold)
                critical_alert_types = get_critical_alert_types(detections)
                counts = count_classes(detections)

                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                )

                with metrics_placeholder.container():
                    render_kpis(counts, len(violations))

                if violations:
                    # Escape alert text because the panel is rendered with raw HTML.
                    critical_messages = {
                        CRITICAL_ALERTS[alert_type]["message"]
                        for alert_type in critical_alert_types
                    }
                    all_messages = sorted(set(violations).union(critical_messages))
                    safe_messages = "".join(f"<div>{html.escape(message)}</div>" for message in all_messages)
                    alerts_placeholder.markdown(
                        f"""
                        <div class="alert-container">
                            <div class="alert-title">CRITICAL PPE VIOLATION</div>
                            <div class="alert-body">{safe_messages}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    alerts_placeholder.success("No active PPE violations detected.")

                if audio_alerts_enabled and critical_alert_types:
                    render_audio_alerts(critical_alert_types, audio_placeholder)

                if detections:
                    table_placeholder.dataframe(pd.DataFrame(detections), use_container_width=True)
                else:
                    table_placeholder.info("No detections found in the current frame.")

                frames_this_run += 1
                time.sleep(0.03)

            cap.release()

            if st.session_state.get("run_live_detection", False):
                # Continue the live loop by asking Streamlit to run the script again.
                st.rerun()

with tab_media:
    # Batch mode accepts one image or video file and routes it by extension.
    uploaded_file = st.file_uploader(
        "Upload Inspection Media (Image/Video)",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
    )
    
    if uploaded_file is None:
        st.info("Upload an image or video to run PPE analysis.")
    elif not model:
        st.error("Model unavailable. Uploaded media cannot be analyzed.")
    else:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            render_uploaded_image(model, uploaded_file, conf_threshold, audio_alerts_enabled)
        elif suffix in VIDEO_EXTENSIONS:
            render_uploaded_video(model, uploaded_file, conf_threshold, audio_alerts_enabled)
        else:
            st.error("Unsupported file type. Please upload an image or video file.")

# Footer Technical Info
# Static technical footer to complete the dashboard visual treatment.
st.markdown("---")
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono'; font-size: 0.65rem; color: var(--text-secondary);">
        <span>ENGINE: ULTRALYTICS YOLOv8</span>
        <span>STATUS: SYSTEM_READY</span>
        <span>LATENCY: 12ms</span>
    </div>
    """,
    unsafe_allow_html=True
)
