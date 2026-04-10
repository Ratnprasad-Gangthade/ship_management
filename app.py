import hashlib
import os
import subprocess

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from allocation import allocate_oil_to_ships
from delay_penalty_service import run_penalty_calculation
from delay_prediction import predict_allocated_ship_delays
from db import (
    get_allocated_ship_context,
    get_overview,
    get_previous_run_allocations,
    get_recent_allocations,
    initialize_database,
    load_oils_and_ships,
    replace_allocations,
    replace_oils_data,
    replace_ships_data,
)

load_dotenv(override=True)

st.set_page_config(page_title="ShipSync AI", page_icon="🚢", layout="wide")


REQUIRED_OIL_COLUMNS = [
    "oil_id",
    "oil_type",
    "delivery_deadline",
    "origin_port",
    "destination_port",
    "quantity_mt",
]

REQUIRED_SHIP_COLUMNS = [
    "ship_id",
    "capacity_mt",
    "last_oil_type",
    "available_date",
    "available_port",
]

DATA_DIR = "data"
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
RAW_DIR = os.path.join(DATA_DIR, "raw")
NOTEBOOK_PATH = os.path.join("notebooks", "data_cleaning.ipynb")
HISTORICAL_DELAY_CSV_PATH = os.getenv("HISTORICAL_DELAY_CSV_PATH", "historical_delays_updated.csv")


def _inject_theme_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ocean-950: #04111f;
            --ocean-900: #071a31;
            --ocean-850: #0a2140;
            --ocean-800: #0d2c52;
            --ocean-700: #13406f;
            --ocean-500: #2a78c8;
            --aqua-400: #4dd9ff;
            --foam-200: #d8f7ff;
            --gold-300: #ffd36b;
            --panel: rgba(7, 26, 49, 0.68);
            --panel-strong: rgba(7, 26, 49, 0.86);
            --border: rgba(168, 218, 255, 0.18);
            --shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
        }

        .stApp {
            background:
                radial-gradient(circle at top, rgba(77, 217, 255, 0.22), transparent 28%),
                radial-gradient(circle at 15% 20%, rgba(255, 255, 255, 0.08), transparent 18%),
                linear-gradient(180deg, var(--ocean-950) 0%, var(--ocean-900) 35%, var(--ocean-800) 100%);
            color: #ecfeff;
            background-attachment: fixed;
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                radial-gradient(rgba(255, 255, 255, 0.08) 1px, transparent 1px),
                radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px);
            background-size: 56px 56px, 110px 110px;
            background-position: 0 0, 24px 24px;
            opacity: 0.16;
        }

        .main .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2.2rem;
        }

        .shipsync-header {
            text-align: center;
            padding: 1.25rem 1rem 0.85rem;
            margin-bottom: 1rem;
        }

        .shipsync-header h1 {
            margin: 0;
            font-size: clamp(2.25rem, 4vw, 3.9rem);
            font-weight: 900;
            letter-spacing: 0.02em;
            line-height: 1.04;
            background: linear-gradient(90deg, #f5fbff 0%, #5ddcff 28%, #8cf3ff 55%, #ffd36b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 10px 35px rgba(0, 0, 0, 0.25);
        }

        .shipsync-header p {
            margin: 0.45rem auto 0;
            max-width: 980px;
            color: rgba(224, 242, 255, 0.88);
            font-size: 1.03rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .section-heading {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0.35rem 0 0.85rem;
            color: #f8fcff;
            font-weight: 800;
            letter-spacing: 0.01em;
        }

        .section-heading span {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(77, 217, 255, 0.28), rgba(255, 211, 107, 0.22));
            border: 1px solid rgba(255, 255, 255, 0.14);
        }

        .glass-panel,
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }

        .glass-panel {
            border-radius: 24px;
            padding: 1.5rem 1.5rem 1.35rem;
        }

        .login-shell {
            max-width: 540px;
            margin: 0 auto;
            padding: 1.55rem 1.55rem 1.25rem;
            border-radius: 30px;
            background: linear-gradient(180deg, rgba(8, 28, 53, 0.78), rgba(7, 26, 49, 0.58));
            border: 1px solid rgba(167, 220, 255, 0.2);
            box-shadow: 0 30px 90px rgba(2, 10, 22, 0.55);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            animation: floatPanel 6.5s ease-in-out infinite;
        }

        .login-title {
            text-align: center;
            font-size: 1.75rem;
            font-weight: 900;
            margin: 0 0 0.3rem;
            color: #f5fbff;
        }

        .login-subtitle {
            text-align: center;
            color: rgba(221, 242, 255, 0.86);
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
        }

        .field-label {
            font-size: 1rem;
            font-weight: 800;
            color: #e2f5ff;
            margin-bottom: 0.35rem;
        }

        .stTextInput > div > div > input {
            font-size: 1.03rem;
            font-weight: 700;
            color: #f8fbff;
            background: rgba(5, 18, 35, 0.55);
        }

        .stTextInput > div > div > input::placeholder {
            color: rgba(219, 234, 254, 0.55);
        }

        .stButton > button,
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border-radius: 999px;
            border: 1px solid rgba(155, 225, 255, 0.26);
            background: linear-gradient(90deg, #1580d8 0%, #1d67c1 50%, #13406f 100%);
            color: white;
            font-size: 1.03rem;
            font-weight: 800;
            transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
        }

        .stButton > button:hover,
        section[data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 14px 32px rgba(24, 122, 204, 0.32);
            filter: brightness(1.06);
        }

        .ship-card,
        .result-card,
        .stat-card {
            position: relative;
            overflow: hidden;
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem;
            background: linear-gradient(180deg, rgba(10, 34, 63, 0.92), rgba(7, 26, 49, 0.7));
            border: 1px solid rgba(157, 214, 255, 0.15);
            box-shadow: 0 22px 50px rgba(0, 0, 0, 0.24);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            animation: floatPanel 7s ease-in-out infinite;
        }

        .ship-card:hover,
        .result-card:hover,
        .stat-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 28px 56px rgba(0, 0, 0, 0.34);
            border-color: rgba(95, 223, 255, 0.35);
        }

        .ship-card::before,
        .result-card::before,
        .stat-card::before {
            content: "";
            position: absolute;
            inset: auto -15% -55% -15%;
            height: 110px;
            background: radial-gradient(circle, rgba(77, 217, 255, 0.17), transparent 65%);
            pointer-events: none;
        }

        .stat-icon,
        .card-icon {
            width: 2.25rem;
            height: 2.25rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 16px;
            font-size: 1.2rem;
            margin-bottom: 0.7rem;
            background: linear-gradient(135deg, rgba(77, 217, 255, 0.22), rgba(255, 211, 107, 0.22));
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .stat-label,
        .card-title {
            color: rgba(224, 242, 255, 0.9);
            font-size: 0.88rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        .stat-value,
        .card-value {
            margin: 0.2rem 0 0.25rem;
            color: #ffffff;
            font-size: clamp(1.45rem, 2vw, 2rem);
            font-weight: 950;
            line-height: 1.05;
        }

        .card-subtext,
        .stat-detail {
            color: rgba(203, 236, 255, 0.82);
            font-size: 0.88rem;
            font-weight: 600;
        }

        .wave-stage {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            height: 150px;
            overflow: hidden;
            pointer-events: none;
            z-index: 0;
        }

        .wave-stage::after {
            content: "";
            position: absolute;
            inset: 48px 0 0;
            background: linear-gradient(180deg, rgba(0, 0, 0, 0), rgba(4, 17, 31, 0.8));
        }

        .wave-layer {
            position: absolute;
            left: -25%;
            width: 150%;
            border-radius: 44%;
            opacity: 0.92;
        }

        .wave-layer.layer-1 {
            bottom: -62px;
            height: 145px;
            background: rgba(30, 120, 210, 0.5);
            animation: waveDriftA 10s linear infinite;
        }

        .wave-layer.layer-2 {
            bottom: -76px;
            height: 165px;
            background: rgba(18, 82, 150, 0.62);
            animation: waveDriftB 13s linear infinite;
        }

        .wave-layer.layer-3 {
            bottom: -92px;
            height: 185px;
            background: rgba(4, 37, 74, 0.74);
            animation: waveDriftA 18s linear infinite;
        }

        .wave-ship {
            position: absolute;
            bottom: 46px;
            left: -10%;
            font-size: 1.7rem;
            filter: drop-shadow(0 8px 12px rgba(0, 0, 0, 0.28));
            animation: shipSail 20s linear infinite, shipFloat 2.9s ease-in-out infinite;
        }

        .sidebar-title {
            margin: 0.2rem 0 0.3rem;
            font-size: 1.05rem;
            font-weight: 900;
            color: #f4fbff;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(5, 18, 35, 0.95), rgba(7, 26, 49, 0.88));
            border-right: 1px solid rgba(159, 220, 255, 0.14);
        }

        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.25rem;
        }

        section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
            gap: 0.35rem;
        }

        section[data-testid="stSidebar"] .stRadio label {
            border-radius: 16px;
            padding: 0.15rem 0.35rem;
            transition: transform 160ms ease, background 160ms ease;
        }

        section[data-testid="stSidebar"] .stRadio label:hover {
            transform: translateX(2px);
            background: rgba(77, 217, 255, 0.1);
        }

        section[data-testid="stSidebar"] .stRadio [aria-checked="true"] {
            background: linear-gradient(90deg, rgba(77, 217, 255, 0.16), rgba(21, 128, 216, 0.18));
            border: 1px solid rgba(95, 223, 255, 0.2);
        }

        .fade-in {
            animation: fadeInUp 520ms ease both;
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes floatPanel {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-4px); }
        }

        @keyframes waveDriftA {
            0% { transform: translateX(0) translateY(0); }
            50% { transform: translateX(7%) translateY(-4px); }
            100% { transform: translateX(0) translateY(0); }
        }

        @keyframes waveDriftB {
            0% { transform: translateX(0) translateY(0); }
            50% { transform: translateX(-7%) translateY(-5px); }
            100% { transform: translateX(0) translateY(0); }
        }

        @keyframes shipSail {
            0% { left: -10%; }
            100% { left: 106%; }
        }

        @keyframes shipFloat {
            0%, 100% { transform: translateY(0) rotate(-1deg); }
            50% { transform: translateY(-4px) rotate(1deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_app_header() -> None:
    st.markdown(
        """
        <div class="shipsync-header fade-in">
            <h1>🚢 ShipSync AI</h1>
            <p>AI-Based Crude Oil Classification & Intelligent Ship Assignment</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_card(title: str, value: str, icon: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="stat-card fade-in">
            <div class="stat-icon">{icon}</div>
            <div class="stat-label">{title}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_card(title: str, value: str, icon: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="result-card fade-in">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-subtext">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def ensure_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "delay_predictions" not in st.session_state:
        st.session_state.delay_predictions = None
    if "penalty_results" not in st.session_state:
        st.session_state.penalty_results = None
    if "penalty_rules" not in st.session_state:
        st.session_state.penalty_rules = None


def ensure_storage_dirs() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)


def show_login_page() -> None:
    _inject_theme_css()
    _render_app_header()

    st.markdown(
        """
        <div class="login-shell fade-in">
            <div class="login-title">🔐 Manager Login</div>
            <div class="login-subtitle">Please sign in to access the dashboard features</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, center, right = st.columns([1.15, 1.75, 1.15])
    with center:
        with st.container(border=True):
            st.markdown('<div class="field-label">Username</div>', unsafe_allow_html=True)
            username = st.text_input(
                "Username",
                key="login_username",
                label_visibility="collapsed",
                placeholder="Enter username",
            )

            st.markdown('<div class="field-label">Password</div>', unsafe_allow_html=True)
            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
                label_visibility="collapsed",
                placeholder="Enter password",
            )

            if st.button("Login", use_container_width=True):
                expected_username = os.getenv("MANAGER_USERNAME", "manager")
                expected_password_hash = os.getenv("MANAGER_PASSWORD_HASH")

                if username == expected_username and hash_password(password) == expected_password_hash:
                    st.success("Login successful")
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            st.markdown(
                '<div class="card-subtext" style="text-align:center; margin-top: 0.55rem;">Credentials are configured via MANAGER_USERNAME and MANAGER_PASSWORD_HASH.</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="wave-stage">
            <div class="wave-layer layer-1"></div>
            <div class="wave-layer layer-2"></div>
            <div class="wave-layer layer-3"></div>
            <div class="wave-ship">🚢</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> tuple[bool, str]:
    normalized = set(df.columns)

    # Keep backward compatibility for existing files that still use origin_country.
    if "origin_port" in required_columns and "origin_port" not in normalized and "origin_country" in normalized:
        normalized.add("origin_port")

    missing = set(required_columns) - normalized
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    return True, "OK"


def run_cleaning_notebook() -> None:
    notebook_abspath = os.path.abspath(NOTEBOOK_PATH)
    command = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.allow_errors=True",
        "--ExecutePreprocessor.timeout=600",
        notebook_abspath,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr if stderr else stdout
        raise RuntimeError(f"Notebook cleaning failed: {details}")


def load_cleaned_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    oil_path = os.path.join(CLEAN_DIR, "oil_data_cleaned.csv")
    ship_path = os.path.join(CLEAN_DIR, "ship_data_cleaned.csv")

    if not os.path.exists(oil_path) or not os.path.exists(ship_path):
        raise RuntimeError("Notebook did not produce cleaned CSV outputs")

    return pd.read_csv(oil_path), pd.read_csv(ship_path)


def show_upload_page() -> None:
    st.markdown("## 📤 Upload Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Oil Data")
        oil_file = st.file_uploader("Upload Oil CSV", type="csv", key="oil_upload")
    
    with col2:
        st.subheader("Ship Data")
        ship_file = st.file_uploader("Upload Ship CSV", type="csv", key="ship_upload")
    
    if st.button("Proceed", use_container_width=True):
        if oil_file is None or ship_file is None:
            st.error("Please upload both files")
            return
        
        try:
            oil_df = pd.read_csv(oil_file)
            ship_df = pd.read_csv(ship_file)

            if "origin_port" not in oil_df.columns and "origin_country" in oil_df.columns:
                oil_df["origin_port"] = oil_df["origin_country"]

            # Notebook pipeline still expects origin_country; mirror origin_port when needed.
            if "origin_country" not in oil_df.columns and "origin_port" in oil_df.columns:
                oil_df["origin_country"] = oil_df["origin_port"]
            
            oil_valid, oil_message = validate_columns(oil_df, REQUIRED_OIL_COLUMNS)
            ship_valid, ship_message = validate_columns(ship_df, REQUIRED_SHIP_COLUMNS)
            
            if not oil_valid:
                st.error(oil_message)
                return
            if not ship_valid:
                st.error(ship_message)
                return
            
            ensure_storage_dirs()

            raw_oil_path = os.path.join(RAW_DIR, "oil_data_raw.csv")
            raw_ship_path = os.path.join(RAW_DIR, "ship_data_raw.csv")
            oil_df.to_csv(raw_oil_path, index=False)
            ship_df.to_csv(raw_ship_path, index=False)

            run_cleaning_notebook()
            cleaned_oil, cleaned_ship = load_cleaned_outputs()

            if "origin_port" not in cleaned_oil.columns and "origin_country" in cleaned_oil.columns:
                cleaned_oil["origin_port"] = cleaned_oil["origin_country"]

            if cleaned_oil.empty or cleaned_ship.empty:
                st.error("Cleaning removed all rows. Please verify raw input quality.")
                return

            cleaned_oil_path = os.path.join(CLEAN_DIR, "oil_data_cleaned.csv")
            cleaned_ship_path = os.path.join(CLEAN_DIR, "ship_data_cleaned.csv")

            cleaned_oil_db = cleaned_oil.copy()
            cleaned_oil_db["delivery_deadline"] = pd.to_datetime(
                cleaned_oil_db["delivery_deadline"],
                format="%d-%m-%Y",
                errors="coerce",
            ).dt.date

            if "origin_port" not in cleaned_oil_db.columns and "origin_country" in cleaned_oil_db.columns:
                cleaned_oil_db["origin_port"] = cleaned_oil_db["origin_country"]

            cleaned_ship_db = cleaned_ship.copy()
            cleaned_ship_db["available_date"] = pd.to_datetime(
                cleaned_ship_db["available_date"],
                format="%d-%m-%Y",
                errors="coerce",
            ).dt.date

            cleaned_oil_db = cleaned_oil_db.dropna(subset=["delivery_deadline"]).copy()
            cleaned_ship_db = cleaned_ship_db.dropna(subset=["available_date"]).copy()

            if cleaned_oil_db.empty or cleaned_ship_db.empty:
                st.error("Date normalization failed for all rows after cleaning.")
                return
            
            oil_inserted = replace_oils_data(cleaned_oil_db)
            ship_inserted = replace_ships_data(cleaned_ship_db)
            
            st.success(
                f"✅ Upload + cleaning successful!\n"
                f"- Raw oils: {len(oil_df)} → Cleaned oils saved: {oil_inserted}\n"
                f"- Raw ships: {len(ship_df)} → Cleaned ships saved: {ship_inserted}"
            )
            st.caption(f"Cleaned files: {cleaned_oil_path} | {cleaned_ship_path}")
        
        except Exception as error:
            st.error(f"Upload failed: {error}")


def show_dashboard_page() -> None:
    st.markdown('<div class="section-heading fade-in"><span>📊</span><div>Dashboard Overview</div></div>', unsafe_allow_html=True)

    try:
        overview = get_overview()
        oils, ships = load_oils_and_ships()

        cards = [
            ("Total Ships", str(overview["total_ships"]), "🚢", "Fleet capacity across active vessels"),
            ("Allocated Ships", str(overview["allocated_ships"]), "✅", "Ships already assigned to orders"),
            ("Total Orders", str(overview["total_orders"]), "📦", "Oil orders currently in the system"),
            ("Allocated Orders", str(overview["allocated_orders"]), "⚙️", "Orders matched to a vessel"),
            ("Unallocated", str(overview["unallocated_orders"]), "⏳", "Orders still awaiting assignment"),
        ]

        metric_cols = st.columns(5)
        for column, card in zip(metric_cols, cards):
            with column:
                _render_card(*card)

    except Exception as error:
        st.error(f"Failed to load dashboard: {error}")


def show_allocation_page() -> None:
    st.markdown('<div class="section-heading fade-in"><span>🚀</span><div>Allocation Results</div></div>', unsafe_allow_html=True)

    if st.button("🔄 Recalculate Allocation", use_container_width=True):
        try:
            with st.spinner("Calculating optimal ship assignments..."):
                oils, ships = load_oils_and_ships()
                allocation_rows = allocate_oil_to_ships(oils, ships)

                allocated_count = sum(1 for row in allocation_rows if row.get("status") == "allocated")
                unallocated_count = sum(1 for row in allocation_rows if row.get("status") != "allocated")

                replace_allocations(allocation_rows)

            st.success(f"✅ Allocation complete!\n- Allocated: {allocated_count}\n- Unallocated: {unallocated_count}")
            summary_cols = st.columns(3)
            with summary_cols[0]:
                _render_result_card("Allocated", str(allocated_count), "✅", "Orders matched to ships")
            with summary_cols[1]:
                _render_result_card("Unallocated", str(unallocated_count), "⏳", "Orders still pending")
            with summary_cols[2]:
                _render_result_card("Snapshot", str(len(allocation_rows)), "📋", "Rows written to the current allocation set")

        except Exception as error:
            st.error(f"Allocation failed: {error}")

    st.markdown('<div class="section-heading fade-in" style="margin-top:1.1rem;"><span>🧭</span><div>Recent Allocation Results</div></div>', unsafe_allow_html=True)
    try:
        recent = get_recent_allocations()
        if recent.empty:
            st.info("No allocation records yet")
        else:
            display_recent = recent[["oil_id", "ship_id", "status", "final_score", "decision_reason", "allocation_time"]].copy()
            st.dataframe(display_recent, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error(f"Failed to load recent allocations: {error}")

    st.markdown('<div class="section-heading fade-in" style="margin-top:1.1rem;"><span>🗂️</span><div>Allocation History</div></div>', unsafe_allow_html=True)
    try:
        previous_allocations = get_previous_run_allocations()
        if previous_allocations.empty:
            st.info("No previous run allocation history available")
        else:
            display_history = previous_allocations[["oil_id", "ship_id", "status", "final_score", "decision_reason", "allocation_time"]].copy()
            st.dataframe(display_history, use_container_width=True, hide_index=True)

    except Exception as error:
        st.error(f"Failed to load history: {error}")


def show_delay_page() -> None:
    st.markdown('<div class="section-heading fade-in"><span>⏱️</span><div>Delay Prediction</div></div>', unsafe_allow_html=True)

    if st.button("⏱️ Calculate Delay", use_container_width=True):
        try:
            with st.spinner("Forecasting route delays and news risk..."):
                allocated_df = get_allocated_ship_context()
                if allocated_df.empty:
                    st.info("No allocated ships found in current allocation snapshot.")
                    st.session_state.delay_predictions = []
                    return

                predictions = predict_allocated_ship_delays(
                    allocated_df=allocated_df,
                    historical_csv_path=HISTORICAL_DELAY_CSV_PATH,
                )

                if not predictions:
                    st.info("No delay predictions could be generated.")
                    st.session_state.delay_predictions = []
                    return

                st.session_state.delay_predictions = predictions

            st.success(f"Delay prediction complete for {len(predictions)} allocated ships.")

        except Exception as error:
            st.error(f"Delay prediction failed: {error}")

    if st.session_state.delay_predictions is not None:
        stored_predictions = st.session_state.delay_predictions
        if stored_predictions:
            prediction_df = pd.DataFrame(stored_predictions)
            st.markdown('<div class="section-heading fade-in" style="margin-top:1.1rem;"><span>📈</span><div>Delay Prediction Results</div></div>', unsafe_allow_html=True)

            st.dataframe(prediction_df, use_container_width=True, hide_index=True)


def show_penalty_page() -> None:
    st.markdown('<div class="section-heading fade-in"><span>💰</span><div>Delay Penalty</div></div>', unsafe_allow_html=True)

    contract_pdf = st.file_uploader(
        "Upload Contract PDF",
        type=["pdf"],
        key="penalty_contract_pdf",
    )

    if st.button("💰 Calculate Penalty", use_container_width=True):
        if contract_pdf is None:
            st.error("Please upload contract PDF before calculating penalty.")
        else:
            try:
                with st.spinner("Reading contract terms and calculating penalties..."):
                    penalties_df, rules = run_penalty_calculation(
                        contract_pdf.getvalue(),
                        HISTORICAL_DELAY_CSV_PATH,
                    )

                    st.session_state.penalty_results = penalties_df
                    st.session_state.penalty_rules = rules

                st.success("Penalty calculation completed.")
                if penalties_df.empty:
                    st.info("No allocated ships found for penalty calculation.")
            except Exception as error:
                st.error(f"Penalty calculation failed: {error}")

    if st.session_state.penalty_results is not None:
        stored_penalty_df = st.session_state.penalty_results
        stored_rules = st.session_state.penalty_rules
        if stored_rules:
            st.markdown('<div class="section-heading fade-in" style="margin-top:1.1rem;"><span>📜</span><div>Extracted Contract Rules</div></div>', unsafe_allow_html=True)
            st.json(stored_rules)

        if isinstance(stored_penalty_df, pd.DataFrame) and not stored_penalty_df.empty:
            st.markdown('<div class="section-heading fade-in" style="margin-top:1.1rem;"><span>🧾</span><div>Penalty Results</div></div>', unsafe_allow_html=True)
            display_cols = ["ship_id", "delay_days", "penalty", "reason"]
            st.dataframe(
                stored_penalty_df[display_cols],
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    initialize_database()
    ensure_state()
    ensure_storage_dirs()
    _inject_theme_css()
    
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    st.sidebar.markdown('<div class="sidebar-title">⚓ Navigation</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    menu_labels = {
        "Dashboard": "📊 Dashboard",
        "Upload CSV": "📤 Upload CSV",
        "Allocation": "🚀 Allocation",
        "Delay": "⏱️ Delay",
        "Penalty": "💰 Penalty",
    }
    reverse_menu = {value: key for key, value in menu_labels.items()}
    menu_choice = st.sidebar.radio("Go to", list(menu_labels.values()))
    menu_choice = reverse_menu[menu_choice]
    
    st.sidebar.success("✅ Manager session active")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

    _render_app_header()
    
    if menu_choice == "Dashboard":
        show_dashboard_page()
    elif menu_choice == "Upload CSV":
        show_upload_page()
    elif menu_choice == "Allocation":
        show_allocation_page()
    elif menu_choice == "Delay":
        show_delay_page()
    elif menu_choice == "Penalty":
        show_penalty_page()


if __name__ == "__main__":
    main()
