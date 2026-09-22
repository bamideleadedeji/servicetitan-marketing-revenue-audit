import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ML & Statistical Forecasting Libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import SimpleExpSmoothing, Holt

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Marketing Attribution & Revenue Forecasting Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Marketing Performance, Forensic Audit & Predictive Forecasting Engine")
st.markdown("""
**Executive Summary:** Reconciles multi-channel ad spend, call tracking logs, and ServiceTitan CRM records. 
It performs forensic audit checks to detect vendor over-attribution ("ghost leads"), evaluates unit economics 
(**CAC**, **LTV**, **Margin ROAS**), and incorporates **time-series predictive modeling** to forecast revenue trajectories and future CAC.
""")

st.sidebar.header("⚙️ Audit & Forecast Control Panel")

# -----------------------------------------------------------------------------
# STEP 1: SYNTHETIC DATA GENERATOR
# -----------------------------------------------------------------------------
@st.cache_data
def generate_audit_data():
    np.random.seed(42)
    n_days = 120  # 120 days of historical data for statistical time-series forecasting
    dates = pd.date_range(start="2026-05-01", periods=n_days, freq="D")
    
    channels = ["Meta Ads", "Google Search", "External Lead Vendor A", "External Lead Vendor B", "Organic Search"]
    
    spend_data = []
    servicetitan_jobs = []
    customer_id_counter = 1000
    
    for i, date in enumerate(dates):
        # Adding a slight upward growth trend over time
        growth_factor = 1 + (i * 0.003)
        
        for channel in channels:
            if channel == "Google Search":
                spend = np.random.uniform(400, 700) * growth_factor
                reported_leads = int(spend / np.random.uniform(35, 50))
            elif channel == "Meta Ads":
                spend = np.random.uniform(300, 500) * growth_factor
                reported_leads = int(spend / np.random.uniform(25, 40))
            elif channel == "External Lead Vendor A":
                spend = np.random.uniform(500, 800)
                reported_leads = int(spend / np.random.uniform(20, 30))  # High ghost lead risk
            elif channel == "External Lead Vendor B":
                spend = np.random.uniform(400, 600) * growth_factor
                reported_leads = int(spend / np.random.uniform(30, 45))
            else:  # Organic Search
                spend = 0.0
                reported_leads = int(np.random.randint(15, 30) * growth_factor)
                
            spend_data.append({
                "Date": date,
                "Channel": channel,
                "Spend": spend,
                "Vendor_Reported_Leads": reported_leads
            })
            
            # Conversion rates to verified jobs
            if channel in ["Google Search", "Organic Search"]:
                conversion_rate = np.random.uniform(0.65, 0.85)
            elif channel in ["Meta Ads", "External Lead Vendor B"]:
                conversion_rate = np.random.uniform(0.40, 0.60)
            else:
                conversion_rate = np.random.uniform(0.18, 0.28)  # Low conversion yield
                
            actual_jobs = int(reported_leads * conversion_rate)
            
            for _ in range(actual_jobs):
                customer_id = f"CUST-{customer_id_counter}"
                customer_id_counter += 1
                
                revenue = np.random.normal(loc=1250 * growth_factor, scale=300)
                revenue = max(150, round(revenue, 2))
                
                servicetitan_jobs.append({
                    "Date": date,
                    "Customer_ID": customer_id,
                    "Lead_Source": channel,
                    "Job_Status": "Completed",
                    "ServiceTitan_Verified_Revenue": revenue
                })

    df_spend = pd.DataFrame(spend_data)
    df_jobs = pd.DataFrame(servicetitan_jobs)
    
    return df_spend, df_jobs

df_spend, df_jobs = generate_audit_data()

# Sidebar Controls for Financial Inputs
st.sidebar.subheader("Financial Unit Parameters")
gross_margin_pct = st.sidebar.slider("Gross Profit Margin (%)", min_value=10, max_value=80, value=45) / 100.0
repeat_purchase_rate = st.sidebar.slider("Estimated LTV Multiplier", min_value=1.0, max_value=3.0, value=1.8, step=0.1)

st.sidebar.subheader("Predictive Forecast Parameters")
forecast_horizon = st.sidebar.selectbox("Forecast Horizon (Days Ahead)", [15, 30, 60, 90], index=1)
model_choice = st.sidebar.radio("Forecasting Algorithm", ["Holt's Exponential Smoothing (Statsmodels)", "Polynomial Regression (Scikit-Learn)"])

# -----------------------------------------------------------------------------
# STEP 2: DATA AGGREGATION & FORENSIC RECONCILIATION
# -----------------------------------------------------------------------------
st_summary = df_jobs.groupby("Lead_Source").agg(
    ServiceTitan_Completed_Jobs=("Job_Status", "count"),
    Verified_Revenue=("ServiceTitan_Verified_Revenue", "sum")
).reset_index().rename(columns={"Lead_Source": "Channel"})

audit_summary = df_spend.groupby("Channel").agg(
    Total_Ad_Spend=("Spend", "sum"),
    Vendor_Reported_Leads=("Vendor_Reported_Leads", "sum")
).reset_index()

audit_df = pd.merge(audit_summary, st_summary, on="Channel", how="left").fillna(0)

# Metrics Calculations
audit_df["Ghost_Leads_Discrepancy"] = audit_df["Vendor_Reported_Leads"] - audit_df["ServiceTitan_Completed_Jobs"]
audit_df["Lead_Realization_Rate (%)"] = (audit_df["ServiceTitan_Completed_Jobs"] / audit_df["Vendor_Reported_Leads"]) * 100
audit_df["Reported_CPL"] = np.where(audit_df["Vendor_Reported_Leads"] > 0, audit_df["Total_Ad_Spend"] / audit_df["Vendor_Reported_Leads"], 0)
audit_df["True_CAC"] = np.where(audit_df["ServiceTitan_Completed_Jobs"] > 0, audit_df["Total_Ad_Spend"] / audit_df["ServiceTitan_Completed_Jobs"], 0)
audit_df["Gross_Profit"] = audit_df["Verified_Revenue"] * gross_margin_pct
audit_df["Margin_Adjusted_ROAS"] = np.where(audit_df["Total_Ad_Spend"] > 0, audit_df["Gross_Profit"] / audit_df["Total_Ad_Spend"], 0)
audit_df["Estimated_LTV"] = (audit_df["Verified_Revenue"] / audit_df["ServiceTitan_Completed_Jobs"]) * gross_margin_pct * repeat_purchase_rate
audit_df["LTV_to_CAC_Ratio"] = np.where(audit_df["True_CAC"] > 0, audit_df["Estimated_LTV"] / audit_df["True_CAC"], 0)

# -----------------------------------------------------------------------------
# STEP 3: DASHBOARD TOP METRICS OVERVIEW
# -----------------------------------------------------------------------------
total_spend = audit_df["Total_Ad_Spend"].sum()
total_verified_rev = audit_df["Verified_Revenue"].sum()
overall_roas = (total_verified_rev * gross_margin_pct) / total_spend if total_spend > 0 else 0
total_ghost_leads = audit_df["Ghost_Leads_Discrepancy"].sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Marketing Spend", f"${total_spend:,.2f}")
col2.metric("Verified ServiceTitan Revenue", f"${total_verified_rev:,.2f}")
col3.metric("Overall Margin ROAS", f"{overall_roas:.2f}x")
col4.metric("Unmatched / Ghost Leads", f"{int(total_ghost_leads):,} leads", delta="-Audit Discrepancy", delta_color="inverse")

st.markdown("---")

# -----------------------------------------------------------------------------
# STEP 4: TABS SETUP INCLUDING TIME-SERIES FORECASTING
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🕵️ Forensic Audit & Attribution", 
    "📈 Unit Economics (CAC vs LTV)", 
    "💡 Budget Optimization Scenario", 
    "🔮 Time-Series Revenue Forecasting"
])

# TAB 1: FORENSIC ATTRIBUTION
with tab1:
    st.subheader("Vendor Reported Leads vs. ServiceTitan Verified Completed Jobs")
    fig_leads = go.Figure()
    fig_leads.add_trace(go.Bar(
        x=audit_df["Channel"], y=audit_df["Vendor_Reported_Leads"],
        name="Vendor Reported Leads", marker_color="#94A3B8"
    ))
    fig_leads.add_trace(go.Bar(
        x=audit_df["Channel"], y=audit_df["ServiceTitan_Completed_Jobs"],
        name="ServiceTitan Verified Jobs", marker_color="#0EA5E9"
    ))
    fig_leads.update_layout(barmode="group", title="Lead Attribution Discrepancy Analysis", xaxis_title="Channel", yaxis_title="Count")
    st.plotly_chart(fig_leads, use_container_width=True)
    
    st.dataframe(audit_df[[
        "Channel", "Total_Ad_Spend", "Vendor_Reported_Leads", 
        "ServiceTitan_Completed_Jobs", "Ghost_Leads_Discrepancy", "Lead_Realization_Rate (%)"
    ]].style.format({
        "Total_Ad_Spend": "${:,.2f}", "Vendor_Reported_Leads": "{:,.0f}", 
        "ServiceTitan_Completed_Jobs": "{:,.0f}", "Ghost_Leads_Discrepancy": "{:,.0f}", 
        "Lead_Realization_Rate (%)": "{:.1f}%"
    }), use_container_width=True)

# TAB 2: UNIT ECONOMICS
with tab2:
    st.subheader("True Customer Acquisition Cost (CAC) vs Estimated LTV")
    fig_cac = px.bar(
        audit_df[audit_df["Total_Ad_Spend"] > 0],
        x="Channel", y=["True_CAC", "Estimated_LTV"], barmode="group",
        title="Unit Economics Evaluation per Channel",
        labels={"value": "USD ($)", "variable": "Metric"},
        color_discrete_map={"True_CAC": "#EF4444", "Estimated_LTV": "#10B981"}
    )
    st.plotly_chart(fig_cac, use_container_width=True)
    
    st.dataframe(audit_df[[
        "Channel", "Reported_CPL", "True_CAC", "Estimated_LTV", "LTV_to_CAC_Ratio", "Margin_Adjusted_ROAS"
    ]].style.format({
        "Reported_CPL": "${:,.2f}", "True_CAC": "${:,.2f}", "Estimated_LTV": "${:,.2f}", 
        "LTV_to_CAC_Ratio": "{:.2f}x", "Margin_Adjusted_ROAS": "{:.2f}x"
    }), use_container_width=True)

# TAB 3: BUDGET OPTIMIZER
with tab3:
    st.subheader("Dynamic Budget Re-allocation Optimizer")
    allocations = {}
    col_alloc1, col_alloc2 = st.columns(2)
    with col_alloc1:
        allocations["Google Search"] = st.slider("Google Search Share (%)", 0, 100, 40)
        allocations["Meta Ads"] = st.slider("Meta Ads Share (%)", 0, 100, 30)
    with col_alloc2:
        allocations["External Lead Vendor A"] = st.slider("External Lead Vendor A Share (%)", 0, 100, 5)
        allocations["External Lead Vendor B"] = st.slider("External Lead Vendor B Share (%)", 0, 100, 25)
        
    total_alloc = sum(allocations.values())
    if total_alloc != 100:
        st.warning(f"Total budget allocation must equal 100%. Current total: **{total_alloc}%**")
    else:
        st.success("Budget allocation balanced at 100%!")
        projected_rev = sum([(pct / 100.0) * total_spend * audit_df.loc[audit_df["Channel"] == ch, "Margin_Adjusted_ROAS"].values[0] for ch, pct in allocations.items()])
        current_gross_profit = audit_df["Gross_Profit"].sum()
        st.metric("Projected Gross Profit", f"${projected_rev:,.2f}", delta=f"${projected_rev - current_gross_profit:,.2f} Profit Delta")

# TAB 4: TIME-SERIES FORECASTING WITH 95% CONFIDENCE INTERVALS
with tab4:
    st.subheader(" Predictive Time-Series Revenue Forecasting with 95% Confidence Intervals")
    st.markdown("""
    This forecasting engine projects future ServiceTitan revenue streams while modeling **uncertainty bounds (95% Confidence Interval)** 
    to help leadership plan for best-case, expected, and conservative financial outcomes.
    """)
    
    # Prepare Daily Aggregated Time-Series Data
    daily_rev = df_jobs.groupby("Date")["ServiceTitan_Verified_Revenue"].sum().reset_index()
    daily_spend = df_spend.groupby("Date")["Spend"].sum().reset_index()
    daily_ts = pd.merge(daily_rev, daily_spend, on="Date", how="outer").fillna(0)
    daily_ts = daily_ts.sort_values("Date").reset_index(drop=True)
    
    # Generating Future Date Range
    last_date = daily_ts["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    
    if "Holt" in model_choice:
        # STATSMODELS: Holt's Linear Exponential Smoothing with Prediction Variance
        model_rev = Holt(daily_ts["ServiceTitan_Verified_Revenue"], initialization_method="estimated").fit()
        forecast_rev = model_rev.forecast(forecast_horizon)
        
        # Calculate residual variance for prediction interval (95% CI -> 1.96 * std_error)
        residuals = model_rev.resid
        sigma = np.std(residuals)
        
        # Standard error scales over forecast horizon step t
        horizon_steps = np.arange(1, forecast_horizon + 1)
        stderr = sigma * np.sqrt(horizon_steps)
        
        upper_bound = forecast_rev + (1.96 * stderr)
        lower_bound = np.maximum(0, forecast_rev - (1.96 * stderr))  # Prevent negative revenue predictions
        
        # Spend Forecast
        model_spend = Holt(daily_ts["Spend"], initialization_method="estimated").fit()
        forecast_spend = model_spend.forecast(forecast_horizon)

    else:
        # SCIKIT-LEARN: Polynomial Regression with Residual Uncertainty Bounds
        daily_ts["Time_Index"] = np.arange(len(daily_ts))
        X = daily_ts[["Time_Index"]]
        y_rev = daily_ts["ServiceTitan_Verified_Revenue"]
        y_spend = daily_ts["Spend"]
        
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        
        # Train SKLearn Models
        model_rev = LinearRegression().fit(X_poly, y_rev)
        model_spend = LinearRegression().fit(X_poly, y_spend)
        
        # Future Indices
        future_indices = np.arange(len(daily_ts), len(daily_ts) + forecast_horizon).reshape(-1, 1)
        future_poly = poly.transform(future_indices)
        
        forecast_rev = model_rev.predict(future_poly)
        forecast_spend = model_spend.predict(future_poly)
        
        # Residual variance for SKLearn bounds
        residuals = y_rev - model_rev.predict(X_poly)
        sigma = np.std(residuals)
        
        horizon_steps = np.arange(1, forecast_horizon + 1)
        stderr = sigma * np.sqrt(1 + (horizon_steps / len(daily_ts)))
        
        upper_bound = forecast_rev + (1.96 * stderr)
        lower_bound = np.maximum(0, forecast_rev - (1.96 * stderr))

    # Combine Forecast and Bounds into DataFrame
    df_future = pd.DataFrame({
        "Date": future_dates,
        "ServiceTitan_Verified_Revenue": forecast_rev,
        "Upper_Bound_95": upper_bound,
        "Lower_Bound_95": lower_bound,
        "Spend": forecast_spend,
        "Type": "Forecast"
    })
    
    daily_ts["Type"] = "Historical"
    daily_ts["Upper_Bound_95"] = daily_ts["ServiceTitan_Verified_Revenue"]
    daily_ts["Lower_Bound_95"] = daily_ts["ServiceTitan_Verified_Revenue"]
    
    combined_ts = pd.concat([
        daily_ts[["Date", "ServiceTitan_Verified_Revenue", "Upper_Bound_95", "Lower_Bound_95", "Spend", "Type"]], 
        df_future
    ], axis=0).reset_index(drop=True)
    
    # Plotting Forecast Chart with Plotly Shaded Area
    fig_forecast = go.Figure()
    
    # Historical Revenue Line
    hist_mask = combined_ts["Type"] == "Historical"
    fig_forecast.add_trace(go.Scatter(
        x=combined_ts.loc[hist_mask, "Date"], 
        y=combined_ts.loc[hist_mask, "ServiceTitan_Verified_Revenue"],
        mode="lines", name="Historical Revenue", line=dict(color="#0EA5E9", width=2)
    ))
    
    # Continuous forecast dates
    fc_mask = combined_ts["Type"] == "Forecast"
    fc_dates = pd.concat([combined_ts.loc[hist_mask, "Date"].tail(1), combined_ts.loc[fc_mask, "Date"]])
    fc_rev = pd.concat([combined_ts.loc[hist_mask, "ServiceTitan_Verified_Revenue"].tail(1), combined_ts.loc[fc_mask, "ServiceTitan_Verified_Revenue"]])
    fc_upper = pd.concat([combined_ts.loc[hist_mask, "Upper_Bound_95"].tail(1), combined_ts.loc[fc_mask, "Upper_Bound_95"]])
    fc_lower = pd.concat([combined_ts.loc[hist_mask, "Lower_Bound_95"].tail(1), combined_ts.loc[fc_mask, "Lower_Bound_95"]])
    
    # 95% Confidence Interval Band (Upper Limit)
    fig_forecast.add_trace(go.Scatter(
        x=fc_dates, y=fc_upper,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"
    ))
    
    # 95% Confidence Interval Band (Lower Limit + Shading)
    fig_forecast.add_trace(go.Scatter(
        x=fc_dates, y=fc_lower,
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(16, 185, 129, 0.18)",
        name="95% Confidence Interval", hoverinfo="skip"
    ))
    
    # Expected Forecast Mean Line
    fig_forecast.add_trace(go.Scatter(
        x=fc_dates, y=fc_rev,
        mode="lines+markers", name=f"{forecast_horizon}-Day Expected Forecast", 
        line=dict(color="#10B981", width=3, dash="dash")
    ))
    
    fig_forecast.update_layout(
        title=f"ServiceTitan Revenue Projection with 95% Confidence Interval ({forecast_horizon} Days Ahead)",
        xaxis_title="Date", yaxis_title="Daily Revenue ($)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Executive Scenario Metrics (Expected vs Best Case vs Conservative)
    projected_add_rev = df_future["ServiceTitan_Verified_Revenue"].sum()
    projected_upper_rev = df_future["Upper_Bound_95"].sum()
    projected_lower_rev = df_future["Lower_Bound_95"].sum()
    projected_add_spend = df_future["Spend"].sum()
    
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    fcol1.metric("Expected Projected Revenue", f"${projected_add_rev:,.2f}")
    fcol2.metric("Conservative Bound (95% Lower)", f"${projected_lower_rev:,.2f}", delta=f"-${projected_add_rev - projected_lower_rev:,.2f}", delta_color="inverse")
    fcol3.metric("Optimistic Bound (95% Upper)", f"${projected_upper_rev:,.2f}", delta=f"+${projected_upper_rev - projected_add_rev:,.2f}")
    fcol4.metric("Required Ad Spend", f"${projected_add_spend:,.2f}")
