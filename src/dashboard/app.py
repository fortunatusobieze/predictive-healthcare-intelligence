from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------
# Paths
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# -----------------------
# Data loaders
# -----------------------

@st.cache_data
def load_model_comparison():
    return pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")


@st.cache_data
def load_risk_band_summary():
    return pd.read_csv(OUTPUTS_DIR / "readmission_risk_band_summary_xgb_val.csv")


@st.cache_data
def load_risk_scores():
    return pd.read_csv(OUTPUTS_DIR / "readmission_risk_scores_xgb_val.csv")


@st.cache_data
def load_ed_summary():
    return pd.read_csv(OUTPUTS_DIR / "ed_forecast_summary_naive_ets.csv")


@st.cache_data
def load_ed_forecast_for_hospital(hospital_safe_name: str):
    path = OUTPUTS_DIR / f"ed_forecast_{hospital_safe_name}.csv"
    return pd.read_csv(path)


# -----------------------
# Readmission page
# -----------------------


def show_readmission_page():
    st.title("Readmission risk intelligence")
    st.caption(
        "Predicts 30-day readmission risk and highlights high-risk patients "
        "to support proactive post-discharge planning."
    )

    with st.expander("How to use this page"):
        st.markdown(
            "- **Who is this for?** Project managers, clinical leads, and ward "
            "teams who need to understand and act on readmission risk.\n"
            "- **Step 1 – Check performance:** Use the *Model performance "
            "overview* and *Readmission prediction performance* sections to "
            "see whether the model is performing well enough (ROC-AUC, PR-AUC, "
            "and how concentrated readmissions are in the high-risk bands).\n"
            "- **Step 2 – Understand risk segmentation:** Use the "
            "*High-risk patient identification* section to see how readmission "
            "rates differ across low/medium/high bands.\n"
            "- **Step 3 – Act on high-risk patients:** Use the *High-risk "
            "patients* table to identify specific patients for enhanced "
            "follow-up. Filter by predicted risk, age band, and whether "
            "they actually readmitted to review cases.\n"
            "- **Step 4 – Review drivers of risk:** Use the SHAP plots in "
            "*Model explanations* to understand which features are driving "
            "higher risk scores overall."
        )

    comp = load_model_comparison()
    band_summary = load_risk_band_summary()
    scores = load_risk_scores()

    # Ensure consistent band ordering
    band_order = ["low", "medium", "high"]
    band_summary["risk_band"] = pd.Categorical(
        band_summary["risk_band"], categories=band_order, ordered=True
    )
    band_summary = band_summary.sort_values("risk_band")

    # ---------------- Performance KPIs ----------------
    st.subheader("Model performance overview")

    primary_model = "logistic_regression"
    primary_label = "Logistic regression (primary)"

    row = comp[comp["model"] == primary_model].iloc[0]
    high_band = band_summary[band_summary["risk_band"] == "high"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Primary model", primary_label)
    c2.metric("ROC-AUC", f"{row['roc_auc']:.3f}")
    c3.metric("PR-AUC", f"{row['pr_auc']:.3f}")
    c4.metric(
        "High-risk band readmission rate",
        f"{high_band['readmission_rate']*100:.1f}%",
    )

    st.caption(
        "Logistic regression was selected as the primary model because it "
        "slightly outperforms the other models and is most interpretable."
    )

    st.markdown("---")

    # ---------------- 1. Readmission prediction performance ----------------
    st.subheader("1. Readmission prediction performance")

    col_left, col_right = st.columns((2, 1))

    with col_left:
        st.markdown("**Discrimination (ROC-AUC, similar across models)**")
        roc_df = comp.set_index("model")[["roc_auc"]]
        st.bar_chart(roc_df, use_container_width=True)

    with col_right:
        st.markdown("**Top-risk band metrics**")
        band_metrics = comp[
            [
                "model",
                "prec_top_10",
                "recall_top_10",
                "prec_top_20",
                "recall_top_20",
            ]
        ].copy()
        band_metrics = band_metrics.set_index("model")
        st.dataframe(band_metrics, use_container_width=True)

    st.caption(
        "Top 10% and 20% metrics show how concentrated readmissions are "
        "among the highest-risk patients ranked by the model."
    )

    st.markdown("---")

    # ---------------- 2. High-risk patient identification ----------------
    st.subheader("2. High-risk patient identification")

    col1, col2 = st.columns((2, 1))

    with col1:
        st.markdown("**Readmission rate by risk band**")
        st.bar_chart(
            band_summary.set_index("risk_band")[["readmission_rate"]],
            use_container_width=True,
        )

    with col2:
        st.markdown("**Risk band summary**")
        st.dataframe(
            band_summary.set_index("risk_band"),
            use_container_width=True,
        )

    low_rate = float(
        band_summary.loc[band_summary["risk_band"] == "low", "readmission_rate"]
    )
    high_rate = float(
        band_summary.loc[band_summary["risk_band"] == "high", "readmission_rate"]
    )
    lift = high_rate / low_rate if low_rate > 0 else float("nan")

    st.caption(
        f"In this validation set, the high-risk band has a readmission rate of "
        f"{high_rate*100:.1f}% versus {low_rate*100:.1f}% in the low-risk band "
        f"(~{lift:.1f}× higher)."
    )

    st.markdown("#### High-risk patients (validation set, XGBoost)")

    # Threshold slider
    min_risk = float(scores["predicted_risk"].min())
    max_risk = float(scores["predicted_risk"].max())
    default_thr = float(scores["predicted_risk"].quantile(0.9))

    thr = st.slider(
        "Minimum predicted risk",
        min_value=round(min_risk, 2),
        max_value=round(max_risk, 2),
        value=round(default_thr, 2),
        step=0.01,
    )

    # Extra filters
    colf1, colf2 = st.columns(2)
    only_readmitted = colf1.checkbox(
        "Show only patients who were readmitted",
        value=False,
    )
    age_band_options = ["All"] + sorted(
        scores["age_band"].dropna().unique().tolist()
    )
    age_filter = colf2.selectbox("Age band", age_band_options)

    subset = scores[scores["predicted_risk"] >= thr].copy()
    if only_readmitted:
        subset = subset[subset["actual_unplanned_readmission_30d"] == 1]
    if age_filter != "All":
        subset = subset[subset["age_band"] == age_filter]

    st.caption(
        f"Showing {len(subset)} patients with predicted risk ≥ {thr:.2f} "
        f"(XGBoost validation set)."
    )

    display_cols = [
        "admission_id",
        "patient_id",
        "predicted_risk",
        "actual_unplanned_readmission_30d",
        "risk_decile",
        "risk_band",
        "age_band",
        "prior_admissions_365d",
        "length_of_stay_days",
        "prior_ed_visits_365d",
        "max_news2",
        "num_abnormal_labs",
    ]
    existing_cols = [c for c in display_cols if c in subset.columns]
    subset = subset[existing_cols]

    st.dataframe(subset, use_container_width=True)

    # ---------------- 3. Model explanations (XGBoost SHAP) ----------------
    st.markdown("---")
    st.subheader("3. Model explanations (XGBoost)")

    st.caption(
        "These plots show which features contribute most to predicted "
        "readmission risk across the validation set."
    )

    shap_bar_path = OUTPUTS_DIR / "xgb_shap_summary_bar.png"
    shap_beeswarm_path = OUTPUTS_DIR / "xgb_shap_summary_beeswarm.png"

    col1, col2 = st.columns(2)

    with col1:
        if shap_bar_path.exists():
            st.markdown("**Top drivers of readmission risk (importance)**")
            st.image(str(shap_bar_path), use_column_width=True)
        else:
            st.info("SHAP summary bar plot not found in outputs/.")

    with col2:
        if shap_beeswarm_path.exists():
            st.markdown("**Feature effects across patients (beeswarm)**")
            st.image(str(shap_beeswarm_path), use_column_width=True)
        else:
            st.info("SHAP beeswarm plot not found in outputs/.")

# -----------------------
# ED forecast page
# -----------------------

def show_ed_forecast_page():
    st.title("ED demand forecasting")
    st.caption(
        "Forecasts short-term emergency department arrivals to support "
        "staffing and capacity planning across hospitals."
    )

    with st.expander("How to use this page"):
        st.markdown(
            "- **Who is this for?** Operational leads, site managers, and PMs "
            "planning ED staffing and downstream capacity.\n"
            "- **Step 1 – Select a hospital:** Use the hospital dropdown in "
            "the sidebar to focus on a specific site.\n"
            "- **Step 2 – Check forecast accuracy:** Review the top metrics "
            "(ETS MAE/RMSE/MAPE) and the *Naive vs ETS* table to see how "
            "much better ETS performs than the simple seasonal naive model.\n"
            "- **Step 3 – Inspect forecast vs actual:** Use the *Forecast vs "
            "actual (test window)* chart to see how well forecasts follow "
            "recent demand patterns for that hospital.\n"
            "- **Step 4 – Compare across hospitals:** Use the cross-hospital "
            "charts to see which sites are hardest to forecast and where ETS "
            "delivers the biggest reduction in error compared with naive."
        )

    ed_summary = load_ed_summary()

    # Hospital selector in sidebar
    hospitals = ed_summary["hospital"].tolist()
    hospital = st.sidebar.selectbox("Hospital", hospitals, key="ed_hospital")

    # Metrics for selected hospital
    row = ed_summary[ed_summary["hospital"] == hospital].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hospital", hospital)
    c2.metric("ETS MAE (patients/day)", f"{row['ets_mae']:.2f}")
    c3.metric("ETS RMSE (patients/day)", f"{row['ets_rmse']:.2f}")
    c4.metric("ETS MAPE", f"{row['ets_mape']*100:.1f}%")

    mae_improve = row["naive_mae"] - row["ets_mae"]
    mape_improve = row["naive_mape"] - row["ets_mape"]

    st.caption(
        f"Compared with seasonal naive (same weekday last week), ETS reduces "
        f"MAE by {mae_improve:.2f} patients/day and MAPE by "
        f"{mape_improve*100:.1f} percentage points for {hospital}."
    )

    # Side-by-side comparison of naive vs ETS for selected hospital
    comp_table = pd.DataFrame(
        [
            {
                "model": "Seasonal naive",
                "MAE (patients/day)": row["naive_mae"],
                "RMSE (patients/day)": row["naive_rmse"],
                "MAPE": row["naive_mape"],
            },
            {
                "model": "ETS (Holt-Winters)",
                "MAE (patients/day)": row["ets_mae"],
                "RMSE (patients/day)": row["ets_rmse"],
                "MAPE": row["ets_mape"],
            },
        ]
    )

    st.markdown("**Naive vs ETS accuracy for this hospital**")
    st.dataframe(
        comp_table.set_index("model").style.format(
            {
                "MAE (patients/day)": "{:.2f}",
                "RMSE (patients/day)": "{:.2f}",
                "MAPE": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("---")

    # Load forecast detail for selected hospital
    safe_name = hospital.replace(" ", "_").replace("/", "_")
    df_h = load_ed_forecast_for_hospital(safe_name)
    df_h["date"] = pd.to_datetime(df_h["date"])

    df_test = df_h[df_h["set"] == "test"].copy()

    st.subheader("1. Forecast vs actual (test window)")

    plot_df = df_test[["date", "ed_arrivals"]].rename(
        columns={"ed_arrivals": "Actual arrivals"}
    )
    if "ed_arrivals_pred_naive" in df_test.columns:
        plot_df["Naive (same weekday last week)"] = df_test[
            "ed_arrivals_pred_naive"
        ].values
    if "ed_arrivals_pred_ets" in df_test.columns:
        plot_df["ETS forecast"] = df_test["ed_arrivals_pred_ets"].values

    plot_df = plot_df.set_index("date")
    st.line_chart(plot_df, use_container_width=True)

    st.caption(
        "The test window shows how closely each model tracks observed ED arrivals "
        "over the most recent days."
    )

    st.subheader("2. Cross-hospital ETS accuracy")

    col_a, col_b = st.columns(2)

    # Raw ETS MAE
    with col_a:
        st.markdown("**ETS MAE by hospital (lower is better)**")
        chart_df = ed_summary.set_index("hospital")[["ets_mae"]]
        st.bar_chart(chart_df, use_container_width=True)
        st.caption("Hospitals with shorter bars have more accurate ETS forecasts.")

    # Improvement vs naive (MAE reduction)
    with col_b:
        st.markdown("**MAE reduction vs naive (ETS – naive)**")
        improvement_df = ed_summary.copy()
        improvement_df["mae_reduction"] = (
            improvement_df["naive_mae"] - improvement_df["ets_mae"]
        )
        chart_imp = improvement_df.set_index("hospital")[["mae_reduction"]]
        st.bar_chart(chart_imp, use_container_width=True)
        st.caption(
            "Higher bars mean a larger reduction in error when using ETS "
            "instead of the naive baseline."
        )

    st.dataframe(ed_summary.set_index("hospital"), use_container_width=True)


# -----------------------
# Main app
# -----------------------

def main():
    st.set_page_config(
        page_title="Predictive Healthcare Intelligence",
        layout="wide",
    )

    st.sidebar.title("Predictive Healthcare Intelligence")
    module = st.sidebar.radio(
        "Select module",
        ["Readmission risk", "ED demand forecast"],
    )

    if module == "Readmission risk":
        show_readmission_page()
    else:
        show_ed_forecast_page()


if __name__ == "__main__":
    main()