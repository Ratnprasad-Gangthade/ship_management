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

st.set_page_config(page_title="Oil Ship Allocation", layout="wide")


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
    st.markdown("## 🔐 Manager Login")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", use_container_width=True):
            expected_username = os.getenv("MANAGER_USERNAME", "manager")
            expected_password_hash = os.getenv(
                "MANAGER_PASSWORD_HASH"
            )
            
            if username == expected_username and hash_password(password) == expected_password_hash:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    with col2:
        st.info("**Default Credentials (from .env):**\n\nUse MANAGER_USERNAME and MANAGER_PASSWORD_HASH")


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
    st.markdown("## 📊 Dashboard")
    
    try:
        overview = get_overview()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🚢 Total Ships", overview["total_ships"])
        
        with col2:
            st.metric("✅ Allocated Ships", overview["allocated_ships"])
        
        with col3:
            st.metric("📦 Total Orders", overview["total_orders"])
        
        with col4:
            st.metric("⚙️ Allocated Orders", overview["allocated_orders"])
        
        with col5:
            st.metric("⏳ Unallocated", overview["unallocated_orders"])
    
    except Exception as error:
        st.error(f"Failed to load dashboard: {error}")


def show_allocation_page() -> None:
    st.markdown("## 🚀 Allocation Results")
    
    if st.button("🔄 Recalculate Allocation", use_container_width=True):
        try:
            oils, ships = load_oils_and_ships()
            allocation_rows = allocate_oil_to_ships(oils, ships)

            allocated_count = sum(1 for row in allocation_rows if row.get("status") == "allocated")
            unallocated_count = sum(1 for row in allocation_rows if row.get("status") != "allocated")
            
            replace_allocations(allocation_rows)
            st.success(f"✅ Allocation complete!\n- Allocated: {allocated_count}\n- Unallocated: {unallocated_count}")
        
        except Exception as error:
            st.error(f"Allocation failed: {error}")
    
    st.markdown("### Recent Allocation Results")
    try:
        recent = get_recent_allocations()
        if recent.empty:
            st.info("No allocation records yet")
        else:
            display_recent = recent[["oil_id", "ship_id", "status", "final_score", "decision_reason", "allocation_time"]].copy()
            st.dataframe(display_recent, use_container_width=True, hide_index=True)
    
    except Exception as error:
        st.error(f"Failed to load recent allocations: {error}")
    
    st.markdown("### Allocation History")
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
    st.markdown("## ⏱️ Delay Prediction")

    if st.button("⏱️ Calculate Delay", use_container_width=True):
        try:
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
            st.markdown("**Delay Prediction Results**")
            st.dataframe(pd.DataFrame(stored_predictions), use_container_width=True, hide_index=True)


def show_penalty_page() -> None:
    st.markdown("## 💰 Delay Penalty")

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
            st.markdown("**Extracted Contract Rules**")
            st.json(stored_rules)

        if isinstance(stored_penalty_df, pd.DataFrame) and not stored_penalty_df.empty:
            st.markdown("**Penalty Results**")
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
    
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    st.sidebar.markdown("## ⚓ Navigation")
    menu_choice = st.sidebar.radio("Go to", ["Dashboard", "Upload CSV", "Allocation", "Delay", "Penalty"])
    
    st.sidebar.success("✅ Manager session active")
    
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    
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
