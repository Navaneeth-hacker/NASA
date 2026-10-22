"""
app.py
------
Climate Intelligence & Multi-Dataset Studio.
Full-featured Streamlit application supporting single-dataset deep dive,
multi-dataset comparison, validated time-series combinations, dataset library management,
and live NASA POWER API ingestion.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adapters.nasa_power import NasaPowerAdapter
from src.combiner import DatasetCombiner
from src.load import detect_weather_columns, read_universal_csv
from src.registry import DatasetMetadata, DatasetRegistry, get_registry
from src.validation import IngestionService

# Page configuration
st.set_page_config(
    page_title="Climate Intelligence Studio",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌦️ Climate Intelligence & Multi-Dataset Studio")

# Initialize Registry and Services
registry = get_registry(base_dir="data")
ingestion_service = IngestionService(registry)
combiner = DatasetCombiner(registry)


# ---------------------------------------------------------
# Sidebar: Active Dataset Selection & Ingestion
# ---------------------------------------------------------
st.sidebar.header("🗂️ Dataset Selection")

all_datasets = registry.list_datasets()
if not all_datasets:
    registry.seed_initial_datasets()
    all_datasets = registry.list_datasets()

dataset_options = {d.dataset_id: f"{d.name} ({d.record_count:,} rows)" for d in all_datasets}

# Preserve active selection across reruns
active_id = st.session_state.get("active_dataset_id", all_datasets[0].dataset_id if all_datasets else None)
if active_id not in dataset_options and all_datasets:
    active_id = all_datasets[0].dataset_id

selected_id = st.sidebar.selectbox(
    "Active Dataset:",
    options=list(dataset_options.keys()),
    format_func=lambda x: dataset_options.get(x, x),
    index=list(dataset_options.keys()).index(active_id) if active_id in dataset_options else 0,
    help="Select which dataset to view and analyze in the single-dataset dashboard.",
)
st.session_state["active_dataset_id"] = selected_id

active_meta: Optional[DatasetMetadata] = registry.get(selected_id)
df_active: Optional[pd.DataFrame] = registry.load_clean_dataframe(selected_id) if selected_id else None

# Quick Upload in Sidebar
st.sidebar.divider()
st.sidebar.subheader("📥 Quick CSV Upload")
quick_file = st.sidebar.file_uploader(
    "Throw any CSV file here",
    type=["csv", "txt", "tsv"],
    key="sidebar_quick_upload",
    help="Instantly registers and activates any CSV dropped here.",
)

if quick_file is not None:
    # Check if newly uploaded
    last_uploaded = st.session_state.get("_last_quick_file_name")
    if last_uploaded != quick_file.name:
        st.session_state["_last_quick_file_name"] = quick_file.name
        with st.sidebar.status(f"Ingesting {quick_file.name}..."):
            import_res = ingestion_service.ingest_bytes(
                raw_bytes=quick_file.getvalue(),
                filename=quick_file.name,
                source_type="file_upload",
            )
            if import_res.success and import_res.dataset_id:
                st.session_state["active_dataset_id"] = import_res.dataset_id
                st.sidebar.success(f"Registered: **{import_res.name}**")
                st.rerun()
            else:
                st.sidebar.error(f"Failed to ingest: {import_res.error_message}")

# Sidebar Column Mapping Override
if df_active is not None and not df_active.empty:
    all_numeric_cols = [
        c for c in df_active.select_dtypes(include=[np.number]).columns
        if c not in ["YEAR", "month", "day", "day_of_year", "day_of_week", "is_weekend", "quarter"]
    ]
    detected_cols = detect_weather_columns(df_active)

    with st.sidebar.expander("⚙️ Column Mapping & Detection", expanded=False):
        st.caption("Auto-detected variables from dataset schema:")
        temp_default = detected_cols["temperature"] if detected_cols["temperature"] in df_active.columns else None
        temp_idx = ([None] + all_numeric_cols).index(temp_default) if temp_default in all_numeric_cols else 0
        override_temp = st.selectbox("Temperature (°C)", [None] + all_numeric_cols, index=temp_idx, key="override_temp")

        rain_default = detected_cols["rainfall"] if detected_cols["rainfall"] in df_active.columns else None
        rain_idx = ([None] + all_numeric_cols).index(rain_default) if rain_default in all_numeric_cols else 0
        override_rain = st.selectbox("Rainfall (mm)", [None] + all_numeric_cols, index=rain_idx, key="override_rain")

        rh_default = detected_cols["humidity"] if detected_cols["humidity"] in df_active.columns else None
        rh_idx = ([None] + all_numeric_cols).index(rh_default) if rh_default in all_numeric_cols else 0
        override_rh = st.selectbox("Relative Humidity (%)", [None] + all_numeric_cols, index=rh_idx, key="override_rh")
else:
    override_temp, override_rain, override_rh = None, None, None


# ---------------------------------------------------------
# Primary Navigation Tabs
# ---------------------------------------------------------
tab_dash, tab_combined, tab_library, tab_nasa = st.tabs([
    "📊 Dataset Dashboard",
    "🔗 Multi-Dataset Studio",
    "🗃️ Dataset Library",
    "🛰️ Live NASA Ingestion",
])


# =========================================================
# TAB 1: Dataset Dashboard (Single Dataset Deep Dive)
# =========================================================
with tab_dash:
    if df_active is None or df_active.empty or active_meta is None:
        st.warning("⚠️ No dataset currently selected or loaded. Use the sidebar or Library tab to select a dataset.")
    else:
        # Header Badge & Metadata
        date_min = str(df_active["date"].min().date()) if "date" in df_active.columns else "N/A"
        date_max = str(df_active["date"].max().date()) if "date" in df_active.columns else "N/A"
        is_inferred = df_active.attrs.get("date_inferred", False)

        st.caption(
            f"**{active_meta.name}** | ID: `{active_meta.dataset_id[:8]}...` | "
            f"Observations: **{len(df_active):,}** | Range: **{date_min}** to **{date_max}**"
            + (" *(synthetic timeline)*" if is_inferred else "")
            + (" | 🔗 *Combined Dataset*" if active_meta.is_combined else "")
        )

        # Filters
        years = sorted(df_active["YEAR"].dropna().unique().tolist()) if "YEAR" in df_active.columns else []
        if len(years) > 1:
            col_f1, col_f2 = st.columns([3, 1])
            with col_f1:
                selected_years = st.multiselect("Filter Years:", years, default=years, key="dash_year_filter")
            with col_f2:
                st.write("")
                st.write("")
                reset_y = st.button("Reset All Years", key="dash_reset_years")
                if reset_y:
                    selected_years = years
            df_filtered = df_active[df_active["YEAR"].isin(selected_years)].copy() if selected_years else df_active.copy()
        else:
            df_filtered = df_active.copy()

        # Top KPI Summary Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Records", f"{len(df_filtered):,}")
        kpi2.metric("Date Span", f"{df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")

        if override_temp and override_temp in df_filtered.columns:
            avg_t = df_filtered[override_temp].mean()
            kpi3.metric("Avg Temperature", f"{avg_t:.1f} °C" if pd.notna(avg_t) else "N/A")
        elif all_numeric_cols:
            kpi3.metric(f"Mean ({all_numeric_cols[0]})", f"{df_filtered[all_numeric_cols[0]].mean():.2f}")
        else:
            kpi3.metric("Metrics", "0")

        if override_rain and override_rain in df_filtered.columns:
            tot_rain = df_filtered[override_rain].sum()
            kpi4.metric("Total Rainfall", f"{tot_rain:.1f} mm" if pd.notna(tot_rain) else "N/A")
        elif len(all_numeric_cols) > 1:
            kpi4.metric(f"Mean ({all_numeric_cols[1]})", f"{df_filtered[all_numeric_cols[1]].mean():.2f}")
        else:
            kpi4.metric("Columns", f"{len(df_filtered.columns)}")

        st.divider()

        # Visualizations
        st.subheader("📈 Time-Series Visualizations")
        freq_opt = st.radio("Frequency:", ["Monthly", "Weekly", "Daily"], index=0, horizontal=True, key="dash_freq")
        freq_map = {"Monthly": "MS", "Weekly": "W", "Daily": "D"}
        freq_code = freq_map[freq_opt]

        has_weather = bool(override_rain and override_temp and override_rain in df_filtered.columns and override_temp in df_filtered.columns)

        if has_weather:
            agg_spec = {override_temp: "mean", override_rain: "sum"}
            if override_rh and override_rh in df_filtered.columns:
                agg_spec[override_rh] = "mean"

            try:
                resampled = (
                    df_filtered.set_index("date")
                    .resample(freq_code)
                    .agg(agg_spec)
                    .reset_index()
                    .dropna(subset=[override_temp, override_rain], how="all")
                )
            except Exception:
                resampled = df_filtered.copy()

            vcol1, vcol2 = st.columns(2)
            fig_r = px.bar(
                resampled,
                x="date",
                y=override_rain,
                title=f"{freq_opt} Rainfall ({override_rain}) [mm]",
                labels={"date": "Date", override_rain: "Rainfall (mm)"},
                color_discrete_sequence=["#1f77b4"],
            )
            vcol1.plotly_chart(fig_r, use_container_width=True)

            fig_t = px.line(
                resampled,
                x="date",
                y=override_temp,
                title=f"{freq_opt} Avg Temperature ({override_temp}) [°C]",
                labels={"date": "Date", override_temp: "Temperature (°C)"},
                color_discrete_sequence=["#ff7f0e"],
            )
            vcol2.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("💡 General multi-metric viewer for detected numeric variables.")
            if all_numeric_cols:
                picked = st.multiselect("Select columns:", all_numeric_cols, default=all_numeric_cols[:min(3, len(all_numeric_cols))], key="dash_picked_cols")
                if picked:
                    try:
                        resampled = df_filtered.set_index("date").resample(freq_code)[picked].mean().reset_index()
                    except Exception:
                        resampled = df_filtered.copy()
                    gcol1, gcol2 = st.columns(2)
                    fig_l = px.line(resampled, x="date", y=picked, title=f"{freq_opt} Average Trends")
                    gcol1.plotly_chart(fig_l, use_container_width=True)
                    fig_b = px.bar(resampled, x="date", y=picked[0], title=f"{freq_opt} Distribution: {picked[0]}")
                    gcol2.plotly_chart(fig_b, use_container_width=True)

        # Machine Learning: Tomorrow's Rain Prediction
        st.divider()
        st.subheader("🤖 Tomorrow's Rain Prediction")

        if override_temp and override_rain and override_temp in df_active.columns and override_rain in df_active.columns:
            try:
                model_path = ROOT_DIR / "model.pkl"
                if model_path.exists():
                    model = joblib.load(model_path)
                    full = df_active.sort_values("date").copy().reset_index(drop=True)
                    full["T2M"] = pd.to_numeric(full[override_temp], errors="coerce").interpolate().bfill().ffill()
                    full["PRECTOTCORR"] = pd.to_numeric(full[override_rain], errors="coerce").fillna(0.0)

                    if override_rh and override_rh in full.columns:
                        full["RH2M"] = pd.to_numeric(full[override_rh], errors="coerce").interpolate().bfill().ffill()
                    else:
                        full["RH2M"] = 70.0

                    full["T2M_yesterday"] = full["T2M"].shift(1).bfill().ffill()
                    full["T2M_avg7"] = full["T2M"].rolling(7, min_periods=1).mean().bfill().ffill()
                    full["rain_avg7"] = full["PRECTOTCORR"].rolling(7, min_periods=1).mean().bfill().ffill()
                    full["rain_avg3"] = full["PRECTOTCORR"].rolling(3, min_periods=1).mean().bfill().ffill()
                    full["rh_avg3"] = full["RH2M"].rolling(3, min_periods=1).mean().bfill().ffill()
                    full["month"] = full["date"].dt.month

                    expected_features = list(getattr(model, "feature_names_in_", [
                        "T2M", "T2M_yesterday", "T2M_avg7", "PRECTOTCORR", "rain_avg7", "RH2M", "month"
                    ]))

                    last_row = full.iloc[[-1]]
                    prob = model.predict_proba(last_row[expected_features])[0][1]
                    last_date_str = str(last_row["date"].iloc[0].date())

                    p1, p2 = st.columns([1, 2])
                    with p1:
                        st.metric(
                            label=f"Chance of rain after {last_date_str}",
                            value=f"{prob:.0%}",
                            delta="Rain Expected (>1mm)" if prob >= 0.5 else "Dry / Low Rain",
                            delta_color="normal" if prob >= 0.5 else "inverse",
                        )
                    with p2:
                        if prob >= 0.5:
                            st.success(f"🌧️ **High Rain Likelihood ({prob:.1%})**: Precipitation (>1mm) is likely on the next day.")
                        elif prob >= 0.25:
                            st.info(f"⛅ **Moderate Rain Likelihood ({prob:.1%})**: Possible light rain or isolated showers.")
                        else:
                            st.write(f"☀️ **Low Rain Likelihood ({prob:.1%})**: Mostly dry weather anticipated.")
                        if not (override_rh and override_rh in df_active.columns):
                            st.caption("ℹ️ *Relative humidity column not found; neutral estimate (70%) was used for inference.*")
                else:
                    st.warning("Model file 'model.pkl' not found. Run 'python model.py' to train the model.")
            except Exception as e:
                st.error(f"Prediction failed on this dataset: {e}")
        else:
            st.info("ℹ️ Rain prediction requires Temperature and Rainfall measurements.")

        # Data Explorer
        st.divider()
        st.subheader("📋 Data Explorer")
        st.dataframe(df_filtered, use_container_width=True)
        st.download_button(
            label="📥 Download Filtered CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name=f"{active_meta.dataset_id[:8]}_{Path(active_meta.name).stem}.csv",
            mime="text/csv",
        )


# =========================================================
# TAB 2: Multi-Dataset Studio & Combined Analysis
# =========================================================
with tab_combined:
    st.subheader("🔗 Multi-Dataset Comparison & Combination Studio")
    st.caption("Select two or more datasets to perform comparative cross-analysis or execute validated dataset combinations.")

    available_datasets = registry.list_datasets()
    if len(available_datasets) < 2:
        st.info("ℹ️ You need at least 2 datasets registered to use the Combination Studio. Ingest more datasets via the 'Dataset Library' or 'Live NASA Ingestion' tabs.")
    else:
        # Multi-dataset selection
        selected_combine_ids = st.multiselect(
            "Select Datasets to Compare / Combine:",
            options=[d.dataset_id for d in available_datasets],
            format_func=lambda x: f"{registry.get(x).name} ({registry.get(x).record_count:,} rows)",
            default=[available_datasets[0].dataset_id, available_datasets[1].dataset_id] if len(available_datasets) >= 2 else [],
            key="combine_multiselect",
        )

        if len(selected_combine_ids) >= 2:
            # Comparative Overview Table
            comp_rows = []
            selected_dfs: Dict[str, pd.DataFrame] = {}
            for did in selected_combine_ids:
                meta = registry.get(did)
                df_item = registry.load_clean_dataframe(did)
                if meta and df_item is not None:
                    selected_dfs[did] = df_item
                    d_min = str(df_item["date"].min().date()) if "date" in df_item.columns else "N/A"
                    d_max = str(df_item["date"].max().date()) if "date" in df_item.columns else "N/A"
                    comp_rows.append({
                        "Dataset ID": did[:8] + "...",
                        "Name": meta.name,
                        "Format": meta.format,
                        "Rows": meta.record_count,
                        "Date Start": d_min,
                        "Date End": d_max,
                        "Columns": len(df_item.columns),
                    })

            st.write("####Selected Datasets Inventory")
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

            # Sub-Section: Visual Cross-Comparison
            st.divider()
            st.write("#### Synchronized Multi-Dataset Visualization")

            # Find common numeric columns across selected datasets
            common_numeric = None
            for df_item in selected_dfs.values():
                nums = set(df_item.select_dtypes(include=[np.number]).columns) - {"YEAR", "month", "day", "is_weekend"}
                if common_numeric is None:
                    common_numeric = nums
                else:
                    common_numeric = common_numeric.intersection(nums)

            compare_metric = st.selectbox(
                "Select Variable to Compare Across Datasets:",
                options=sorted(list(common_numeric)) if common_numeric else ["(No shared column names)"],
                key="compare_metric_picker",
            )

            if compare_metric and compare_metric != "(No shared column names)":
                fig_comp = go.Figure()
                for did, df_item in selected_dfs.items():
                    meta = registry.get(did)
                    if "date" in df_item.columns and compare_metric in df_item.columns:
                        # Resample to weekly for smooth comparison
                        res = df_item.set_index("date").resample("W")[compare_metric].mean().reset_index()
                        fig_comp.add_trace(go.Scatter(
                            x=res["date"],
                            y=res[compare_metric],
                            mode="lines",
                            name=meta.name[:25],
                        ))

                fig_comp.update_layout(
                    title=f"Weekly Average Comparison: {compare_metric}",
                    xaxis_title="Date",
                    yaxis_title=compare_metric,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            # Sub-Section: Dataset Combination Execution
            st.divider()
            st.write("####  Execute Validated Dataset Combination")

            op_type = st.radio(
                "Combination Strategy:",
                [
                    "Concatenation (Time-Series Union across periods)",
                    "Join (Horizontal Feature Union on Date Key)"
                ],
                key="combination_op_radio",
            )

            if op_type.startswith("Concatenation"):
                st.info(" **Concatenation** merges records along the chronological time axis. Overlapping dates are cleanly deduplicated.")
                # Pairwise compatibility check
                id1, id2 = selected_combine_ids[0], selected_combine_ids[1]
                m1, m2 = registry.get(id1), registry.get(id2)
                d1, d2 = selected_dfs[id1], selected_dfs[id2]
                check = combiner.check_concatenation_compatibility(m1, d1, m2, d2)

                col_c1, col_c2 = st.columns(2)
                col_c1.metric("Compatibility Status", "Compatible" if check.is_compatible else "Incompatible")
                col_c1.metric("Estimated Output Rows", f"{check.estimated_output_rows:,}")

                with col_c2:
                    if check.warnings:
                        st.warning("" + "\n\n ".join(check.warnings))
                    if check.errors:
                        st.error(" " + "\n\n❌ ".join(check.errors))

                combined_name_input = st.text_input(
                    "Combined Dataset Name:",
                    value=f"Combined: {m1.name[:15]} + {m2.name[:15]}",
                    key="concat_name_input",
                )

                if st.button("Execute Concatenation & Register", disabled=(not check.is_compatible), key="btn_run_concat"):
                    try:
                        with st.spinner("Executing time-series concatenation..."):
                            meta_out, _ = combiner.concatenate_datasets(
                                dataset_ids=selected_combine_ids,
                                combined_name=combined_name_input,
                            )
                        st.success(f"Successfully created and registered combined dataset **{meta_out.name}** (ID: `{meta_out.dataset_id}`)!")
                        st.session_state["active_dataset_id"] = meta_out.dataset_id
                        st.rerun()
                    except Exception as e:
                        st.error(f"Concatenation error: {e}")

            else:
                st.info("**Join** merges complementary variables horizontally on matching dates. Variables with duplicate names receive clean suffixes.")
                if len(selected_combine_ids) != 2:
                    st.warning("Please select exactly two datasets to perform a horizontal join.")
                else:
                    id1, id2 = selected_combine_ids[0], selected_combine_ids[1]
                    m1, m2 = registry.get(id1), registry.get(id2)
                    d1, d2 = selected_dfs[id1], selected_dfs[id2]
                    check_j = combiner.check_join_compatibility(m1, d1, m2, d2, join_key="date")

                    jcol1, jcol2 = st.columns(2)
                    jcol1.metric("Join Compatibility", "Compatible" if check_j.is_compatible else "Incompatible")
                    jcol1.metric("Matching Date Observations", f"{check_j.common_date_count:,}")

                    with jcol2:
                        if check_j.warnings:
                            st.warning(" " + "\n\n ".join(check_j.warnings))
                        if check_j.errors:
                            st.error(" " + "\n\n ".join(check_j.errors))

                    join_name_input = st.text_input(
                        "Joined Dataset Name:",
                        value=f"Joined: {m1.name[:15]} & {m2.name[:15]}",
                        key="join_name_input",
                    )

                    if st.button(" Execute Join & Register", disabled=(not check_j.is_compatible), key="btn_run_join"):
                        try:
                            with st.spinner("Executing horizontal merge..."):
                                meta_out, _ = combiner.join_datasets(
                                    dataset_id1=id1,
                                    dataset_id2=id2,
                                    combined_name=join_name_input,
                                    join_key="date",
                                )
                            st.success(f"Successfully created and registered joined dataset **{meta_out.name}** (ID: `{meta_out.dataset_id}`)!")
                            st.session_state["active_dataset_id"] = meta_out.dataset_id
                            st.rerun()
                        except Exception as e:
                            st.error(f"Join error: {e}")
        else:
            st.info("Select at least 2 datasets above to compare and combine.")


# =========================================================
# TAB 3: Dataset Library & Management
# =========================================================
with tab_library:
    st.subheader(" Dataset Catalog & Management")
    st.caption("Manage persistent dataset registry, batch ingest files with failure isolation, and inspect schema provenance.")

    # Search and Filter
    scol1, scol2 = st.columns([3, 1])
    search_q = scol1.text_input(" Search datasets by name, ID, or description:", key="lib_search")
    source_filter = scol2.selectbox("Filter Source:", ["All", "local_file", "file_upload", "nasa_api", "combined"], key="lib_source_filter")

    filtered_catalog = registry.list_datasets(
        search_query=search_q if search_q else None,
        source_type=None if source_filter == "All" else source_filter,
    )

    # Catalog Table
    catalog_display = []
    for d in filtered_catalog:
        catalog_display.append({
            "Dataset ID": d.dataset_id,
            "Name": d.name,
            "Source Type": d.source_type,
            "Format": d.format,
            "Rows": f"{d.record_count:,}",
            "Size (KB)": f"{d.size_bytes / 1024:.1f}",
            "Combined": " Yes" if d.is_combined else "No",
            "Imported At": d.imported_at[:19].replace("T", " "),
        })

    st.dataframe(pd.DataFrame(catalog_display), use_container_width=True)

    # Batch Ingestion Section
    st.divider()
    st.write("#### Batch Dataset Ingestion")
    st.caption("Upload multiple files at once. Ingestion failure isolation guarantees one bad file will not disrupt valid files.")

    batch_files = st.file_uploader(
        "Upload one or more CSV / TSV / JSON / Parquet files:",
        accept_multiple_files=True,
        type=["csv", "txt", "tsv", "json", "parquet"],
        key="batch_file_uploader",
    )

    if batch_files and st.button(" Process Batch Ingestion", key="btn_run_batch"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        batch_results = []

        for idx, file_obj in enumerate(batch_files):
            status_text.text(f"Ingesting file {idx+1}/{len(batch_files)}: {file_obj.name}...")
            res = ingestion_service.ingest_bytes(
                raw_bytes=file_obj.getvalue(),
                filename=file_obj.name,
                source_type="file_upload",
            )
            batch_results.append(res)
            progress_bar.progress((idx + 1) / len(batch_files))

        status_text.text("Batch processing complete.")
        success_count = sum(1 for r in batch_results if r.success)
        st.write(f"**Results:** {success_count}/{len(batch_files)} datasets imported successfully.")

        # Display outcome breakdown
        batch_summary = []
        for r in batch_results:
            batch_summary.append({
                "Filename": r.filename,
                "Status": " Success" if r.success else " Failed",
                "Dataset ID": r.dataset_id[:8] if r.dataset_id else "N/A",
                "Duplicate": "Yes" if r.is_duplicate else "No",
                "Details": r.error_message if not r.success else "Ready",
            })
        st.dataframe(pd.DataFrame(batch_summary), use_container_width=True)
        st.rerun()

    # Individual Dataset Inspection and Management
    st.divider()
    st.write("#### Dataset Inspector & Maintenance")

    inspect_id = st.selectbox(
        "Choose dataset to inspect or delete:",
        options=[d.dataset_id for d in all_datasets],
        format_func=lambda x: f"{registry.get(x).name} ({x[:8]}...)",
        key="inspect_dataset_picker",
    )

    if inspect_id:
        target_meta = registry.get(inspect_id)
        if target_meta:
            icol1, icol2 = st.columns(2)
            with icol1:
                st.write("**Metadata Attributes:**")
                st.json({
                    "dataset_id": target_meta.dataset_id,
                    "name": target_meta.name,
                    "source": target_meta.source,
                    "source_type": target_meta.source_type,
                    "record_count": target_meta.record_count,
                    "checksum": target_meta.checksum,
                    "is_combined": target_meta.is_combined,
                    "parent_ids": target_meta.parent_ids,
                    "clean_path": target_meta.clean_path,
                })
            with icol2:
                st.write("**Schema & Processing History:**")
                st.json({
                    "schema": target_meta.schema,
                    "history": target_meta.processing_history,
                })

            st.write("---")
            # Deletion guard
            del_col1, del_col2 = st.columns([2, 1])
            confirm_del = del_col1.checkbox(f"I confirm that I want to delete dataset '{target_meta.name}'", key="confirm_delete_chk")
            if del_col2.button(" Delete Dataset", disabled=(not confirm_del), key="btn_delete_dataset"):
                registry.delete_dataset(inspect_id, purge_files=True)
                st.success(f"Dataset '{target_meta.name}' successfully deleted.")
                st.rerun()


# =========================================================
# TAB 4: Live NASA Ingestion
# =========================================================
with tab_nasa:
    st.subheader(" Live NASA POWER Data Ingestion")
    st.caption("Directly fetch daily meteorological observations from NASA's MERRA-2 and satellite feeds for any global coordinates.")

    nasa_adapter = NasaPowerAdapter()
    capabilities = nasa_adapter.discover()

    col_geo, col_time = st.columns(2)
    with col_geo:
        st.write("####  Spatial Coordinates")
        preset_locations = {
            "Custom Coordinates": (8.89, 76.61),
            "Kollam, Kerala (Default)": (8.8932, 76.6141),
            "New Delhi, India": (28.6139, 77.2090),
            "Nairobi, Kenya": (-1.2921, 36.8219),
            "London, UK": (51.5074, -0.1278),
            "New York City, USA": (40.7128, -74.0060),
            "Sydney, Australia": (-33.8688, 151.2093),
        }
        loc_choice = st.selectbox("Location Presets:", list(preset_locations.keys()), key="nasa_preset_choice")
        default_lat, default_lon = preset_locations[loc_choice]

        lat_val = st.number_input("Latitude [-90.0 to 90.0]:", value=default_lat, min_value=-90.0, max_value=90.0, format="%.4f", key="nasa_lat")
        lon_val = st.number_input("Longitude [-180.0 to 180.0]:", value=default_lon, min_value=-180.0, max_value=180.0, format="%.4f", key="nasa_lon")

    with col_time:
        st.write("#### Temporal Range")
        today = date.today()
        default_start = today - timedelta(days=365 * 2)
        default_end = today - timedelta(days=60)  # NASA POWER MERRA-2 has ~1-2 month latency

        start_dt = st.date_input("Start Date:", value=default_start, min_value=date(1981, 1, 1), max_value=today, key="nasa_start")
        end_dt = st.date_input("End Date:", value=default_end, min_value=date(1981, 1, 1), max_value=today, key="nasa_end")

        st.caption("Available Parameters:")
        param_options = list(capabilities["supported_parameters"].keys())
        selected_params = st.multiselect(
            "Meteorological Parameters:",
            options=param_options,
            default=["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS2M", "PS"],
            key="nasa_param_picker",
        )

    nasa_dataset_name = st.text_input(
        "Dataset Label:",
        value=f"NASA POWER: {loc_choice.split(' (')[0]} ({start_dt.year}–{end_dt.year})",
        key="nasa_name_input",
    )

    if st.button(" Fetch from NASA & Register Dataset", key="btn_fetch_nasa"):
        if start_dt > end_dt:
            st.error("Start Date must precede End Date.")
        elif not selected_params:
            st.error("Select at least one meteorological parameter.")
        else:
            with st.spinner("Contacting NASA POWER API..."):
                payload, err = nasa_adapter.fetch(
                    latitude=lat_val,
                    longitude=lon_val,
                    start_date=start_dt.strftime("%Y%m%d"),
                    end_date=end_dt.strftime("%Y%m%d"),
                    parameters=selected_params,
                )

                if err or payload is None:
                    st.error(f"NASA POWER API Error: {err.message if err else 'Unknown'}")
                else:
                    df_nasa = nasa_adapter.normalize(payload)
                    nasa_meta = nasa_adapter.metadata(payload)

                    meta_new, _ = registry.register_dataset(
                        name=nasa_dataset_name,
                        df=df_nasa,
                        source=f"NASA POWER Daily Point API (Lat={lat_val}, Lon={lon_val})",
                        source_type="nasa_api",
                        description=f"Live observation dataset retrieved from NASA POWER for Lat {lat_val}, Lon {lon_val} between {start_dt} and {end_dt}.",
                        weather_mapping=detect_weather_columns(df_nasa),
                    )

                    st.success(f"Successfully fetched and registered **{meta_new.name}** ({len(df_nasa):,} observations)!")
                    st.session_state["active_dataset_id"] = meta_new.dataset_id
                    st.rerun()
