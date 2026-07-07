"""
Stock Predictor Dashboard
ML-powered directional predictions with XGBoost and LSTM
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import io
from contextlib import redirect_stdout
from pandas.tseries.offsets import BDay

from src.data.loader import DataLoader
from src.data.preprocessor import FeatureEngineer
from src.models.backtester import BackTester
from src.models.lstm_model import StockLSTM, LSTMTrainer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

torch.manual_seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════
# CONSTANTS  (fix #9 — more Indian companies)
# ═══════════════════════════════════════════════════════════
TICKERS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "SBI": "SBIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Wipro": "WIPRO.NS",
    "ITC": "ITC.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "L&T": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "HCL Tech": "HCLTECH.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Titan": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
}

PERIOD_MAP = {
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "10Y": "10y",
}

LINE_COLORS = [
    "#58A6FF", "#F97583", "#3FB950", "#D2A8FF", "#F0883E",
    "#56D4DD", "#DBAB79", "#7EE787", "#A5D6FF", "#FF7B72",
]

# ═══════════════════════════════════════════════════════════
# CUSTOM CSS  (fixes #3 boundaries, #6 chart borders, #8 dropdown, #10 heading)
# ═══════════════════════════════════════════════════════════
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* hide streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }

    /* reduce top padding so title is near top (#10) */
    .block-container { padding-top: 1.5rem !important; }

    /* metric cards */
    [data-testid="stMetric"] {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { color: #8B949E !important; }
    [data-testid="stMetricValue"] { color: #E6EDF3 !important; }

    hr { border-color: #21262D !important; }

    /* chart containers — subtle border (#6) */
    [data-testid="stPlotlyChart"] {
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 4px;
    }

    /* close multiselect dropdown after selection (#8) */
    .stMultiSelect [data-baseweb="popover"] { display: none; }
    .stMultiSelect:focus-within [data-baseweb="popover"] { display: block; }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════
# HELPERS  (fix #1 legend, #2 spacing — via layout tweaks)
# ═══════════════════════════════════════════════════════════
def _chart_layout(height: int = 400, title: str = "") -> dict:
    """Reusable dark Plotly layout with proper spacing."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8B949E", family="Inter", size=11),
        xaxis=dict(gridcolor="#21262D", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#21262D", showgrid=True, zeroline=False),
        # fix #1 & #2: more top margin so legend + range‑selector don't overlap title
        margin=dict(l=10, r=10, t=80 if title else 10, b=10),
        height=height,
        legend=dict(
            orientation="h", yanchor="top", y=-0.12,
            xanchor="center", x=0.5, font=dict(size=10),
        ),
        hovermode="x unified",
    )
    if title:
        layout["title"] = dict(
            text=title, font=dict(size=13, color="#E6EDF3"), x=0.02, y=0.97,
        )
    return layout


def directional_accuracy(y_true, y_pred):
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


# ═══════════════════════════════════════════════════════════
# CACHED DATA LOADING
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=300)
def load_ticker_data(symbol: str, period: str) -> pd.DataFrame:
    loader = DataLoader(symbol, period=period)
    df = loader.yf_cleaned()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def get_featured(symbol: str, period: str) -> pd.DataFrame:
    df = load_ticker_data(symbol, period)
    fe = FeatureEngineer(df.copy())
    fe.to_returns()
    return fe.build()


# ═══════════════════════════════════════════════════════════
# FORECAST FUNCTIONS
# ═══════════════════════════════════════════════════════════
def _feature_cols(featured_df: pd.DataFrame) -> list:
    return [
        c for c in featured_df.columns
        if c not in ["Date", "Close", "Open", "High", "Low", "Returns"]
    ]


def forecast_xgb(df_raw, featured_df, feat_cols, forecast_days):
    x = featured_df[feat_cols].values
    y = featured_df["Returns"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    model = XGBRegressor(n_estimators=200, random_state=42, colsample_bytree=0.5)
    model.fit(X_scaled, y)

    df = df_raw.copy()
    predicted_prices, predicted_dates = [], []
    for _ in range(forecast_days):
        fe = FeatureEngineer(df.copy()); fe.to_returns(); feat = fe.build()
        last_row = feat[feat_cols].iloc[-1:].values
        pred_ret = model.predict(scaler.transform(last_row))[0]
        last_close = float(df["Close"].iloc[-1])
        new_close = last_close * (1 + pred_ret)
        next_date = pd.Timestamp(df["Date"].iloc[-1]) + BDay(1)
        predicted_prices.append(new_close); predicted_dates.append(next_date)
        new_row = pd.DataFrame({"Date":[next_date],"Open":[new_close],"High":[new_close*1.005],
                                 "Low":[new_close*0.995],"Close":[new_close],"Volume":[df["Volume"].iloc[-5:].mean()]})
        df = pd.concat([df, new_row], ignore_index=True)

    daily_vol = df_raw["Close"].pct_change().std()
    upper = [p*(1+1.96*daily_vol*np.sqrt(i+1)) for i,p in enumerate(predicted_prices)]
    lower = [p*(1-1.96*daily_vol*np.sqrt(i+1)) for i,p in enumerate(predicted_prices)]
    return predicted_dates, predicted_prices, upper, lower, model


def forecast_lstm(df_raw, featured_df, feat_cols, forecast_days, seq_len=30):
    x = featured_df[feat_cols].values; y = featured_df["Returns"].values
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(x)
    x_seq, y_seq = [], []
    for i in range(len(X_scaled)-seq_len):
        x_seq.append(X_scaled[i:i+seq_len]); y_seq.append(y[i+seq_len])
    x_seq, y_seq = np.array(x_seq), np.array(y_seq)
    input_size = len(feat_cols)
    lstm_model = StockLSTM(input_size=input_size)
    trainer = LSTMTrainer(model=lstm_model, lr=0.01)
    with redirect_stdout(io.StringIO()):
        trainer.train(X_train=x_seq, y_train=y_seq, epochs=100)
    df = df_raw.copy(); predicted_prices, predicted_dates = [], []
    for _ in range(forecast_days):
        fe = FeatureEngineer(df.copy()); fe.to_returns(); feat = fe.build()
        last_seq = feat[feat_cols].iloc[-seq_len:].values
        last_scaled = scaler.transform(last_seq)
        tensor = torch.FloatTensor(last_scaled).unsqueeze(0)
        lstm_model.eval()
        with torch.no_grad(): pred_ret = lstm_model(tensor).item()
        last_close = float(df["Close"].iloc[-1]); new_close = last_close*(1+pred_ret)
        next_date = pd.Timestamp(df["Date"].iloc[-1]) + BDay(1)
        predicted_prices.append(new_close); predicted_dates.append(next_date)
        new_row = pd.DataFrame({"Date":[next_date],"Open":[new_close],"High":[new_close*1.005],
                                 "Low":[new_close*0.995],"Close":[new_close],"Volume":[df["Volume"].iloc[-5:].mean()]})
        df = pd.concat([df, new_row], ignore_index=True)
    daily_vol = df_raw["Close"].pct_change().std()
    upper = [p*(1+1.96*daily_vol*np.sqrt(i+1)) for i,p in enumerate(predicted_prices)]
    lower = [p*(1-1.96*daily_vol*np.sqrt(i+1)) for i,p in enumerate(predicted_prices)]
    return predicted_dates, predicted_prices, upper, lower


# ═══════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════
def run_backtest(featured_df, feat_cols, model_choice, seq_len=30):
    x = featured_df[feat_cols].values; y = featured_df["Returns"].values
    split = int(len(x)*0.8)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x[:split]); X_test = scaler.transform(x[split:])
    y_train, y_test = y[:split], y[split:]
    if model_choice == "XGBoost":
        model = XGBRegressor(n_estimators=200, random_state=42, colsample_bytree=0.5)
        model.fit(X_train, y_train); preds = model.predict(X_test)
    else:
        X_all = np.vstack([X_train, X_test]); xs, ys = [], []
        for i in range(len(X_all)-seq_len):
            xs.append(X_all[i:i+seq_len]); ys.append(y[i+seq_len])
        xs, ys = np.array(xs), np.array(ys)
        seq_split = split - seq_len
        xt, xv = xs[:seq_split], xs[seq_split:]
        yt, yv = ys[:seq_split], ys[seq_split:]
        m = StockLSTM(input_size=len(feat_cols)); t = LSTMTrainer(model=m, lr=0.01)
        with redirect_stdout(io.StringIO()):
            t.train(X_train=xt, y_train=yt, epochs=100)
        preds = t.predict(xv); y_test = yv
    mae = mean_absolute_error(y_test, preds); d_acc = directional_accuracy(y_test, preds)
    bt = BackTester(initial_capital=10000); res = bt.run(y_test, preds)
    res["mae"]=mae; res["dir_acc"]=d_acc
    res["y_test"]=list(y_test); res["preds"]=list(preds) if isinstance(preds, np.ndarray) else preds
    return res


# ═══════════════════════════════════════════════════════════
#  L A Y O U T
# ═══════════════════════════════════════════════════════════

# ── Title  (fix #10 — bigger, centered, no emoji, near top) ──
st.markdown(
    '<h1 style="text-align:center; margin-top:-0.3rem; margin-bottom:0; font-size:2.4rem; '
    'letter-spacing:-0.5px; color:#E6EDF3;">Stock Predictor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="text-align:center; color:#8B949E; margin-top:0; margin-bottom:1.2rem;">'
    'ML-powered directional predictions · Indian &amp; global equities</p>',
    unsafe_allow_html=True,
)

# ── 1. Top row — controls (left) + chart (right)  (#3 proper alignment) ──
ctrl, chart_area = st.columns([1, 2.8], gap="medium")

with ctrl:
    st.markdown("**Stock Tickers**")
    selected = st.multiselect(
        "sel", list(TICKERS.keys()),
        default=["Reliance", "TCS", "Infosys"],
        label_visibility="collapsed",
    )

    st.markdown("**Time Horizon**")
    period_label = st.radio(
        "p", list(PERIOD_MAP.keys()),
        horizontal=True, index=3, label_visibility="collapsed",
    )
    period = PERIOD_MAP[period_label]

    st.markdown("**Model**")
    model_choice = st.radio(
        "m", ["XGBoost", "LSTM"],
        horizontal=True, label_visibility="collapsed",
    )

    st.markdown("**Forecast Days**")
    forecast_days = st.slider("fd", 1, 30, 10, label_visibility="collapsed")

    run_btn = st.button("🚀  Predict", use_container_width=True, type="primary")

# ── 2. Load ticker data (cached) ────────────────────────
comparison = {}
if selected:
    for name in selected:
        try:
            comparison[name] = load_ticker_data(TICKERS[name], period)
        except Exception:
            st.toast(f"⚠️ Could not load {name}", icon="⚠️")

# ── 3. Performance cards (left col) ─────────────────────
with ctrl:
    if len(comparison) >= 2:
        perfs = {n:(d["Close"].iloc[-1]/d["Close"].iloc[0]-1)*100 for n,d in comparison.items()}
        best = max(perfs, key=perfs.get); worst = min(perfs, key=perfs.get)
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Best", best, f"{perfs[best]:+.1f}%")
        c2.metric("Worst", worst, f"{perfs[worst]:+.1f}%")

# ── 4. Process "Run Analysis" ───────────────────────────
if run_btn and selected:
    primary_name = selected[0]; primary_sym = TICKERS[primary_name]
    with st.spinner(f"Engineering features for {primary_name}…"):
        raw_df = load_ticker_data(primary_sym, period)
        feat_df = get_featured(primary_sym, period); fcols = _feature_cols(feat_df)
    if model_choice == "XGBoost":
        with st.spinner("Training XGBoost & forecasting…"):
            dates, prices, upper, lower, _ = forecast_xgb(raw_df, feat_df, fcols, forecast_days)
    else:
        with st.spinner("Training LSTM — this may take up to 60 s…"):
            dates, prices, upper, lower = forecast_lstm(raw_df, feat_df, fcols, forecast_days)
    st.session_state["fc"] = dict(dates=dates, prices=prices, upper=upper, lower=lower)
    st.session_state["fc_ticker"] = primary_name; st.session_state["fc_model"] = model_choice
    with st.spinner("Running backtest…"):
        bt_res = run_backtest(feat_df, fcols, model_choice)
    st.session_state["bt"] = bt_res; st.session_state["bt_model"] = model_choice
    st.rerun()

# ── 5. Main chart (right col)  (fix #2 — rangeselector pushed below title) ──
with chart_area:
    primary = selected[0] if selected else None
    if primary and primary in comparison:
        df_p = comparison[primary]; fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_p["Date"], y=df_p["Close"], name=primary,
                                  line=dict(color="#58A6FF", width=2)))
        # overlay forecast
        fc = st.session_state.get("fc")
        if fc and st.session_state.get("fc_ticker") == primary:
            fc_x = [df_p["Date"].iloc[-1]] + fc["dates"]
            fc_y = [float(df_p["Close"].iloc[-1])] + fc["prices"]
            fc_u = [float(df_p["Close"].iloc[-1])] + fc["upper"]
            fc_l = [float(df_p["Close"].iloc[-1])] + fc["lower"]
            fig.add_trace(go.Scatter(x=fc_x, y=fc_y, name="Forecast",
                                      line=dict(color="#3FB950", width=2, dash="dot")))
            fig.add_trace(go.Scatter(x=fc_x, y=fc_u, line=dict(width=0),
                                      showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fc_x, y=fc_l, fill="tonexty",
                                      fillcolor="rgba(63,185,80,0.12)",
                                      line=dict(width=0), name="95% Confidence", hoverinfo="skip"))
            fig.add_vline(x=df_p["Date"].iloc[-1], line_dash="dash", line_color="#484F58",
                          annotation_text="Today", annotation_font_color="#8B949E")
        layout = _chart_layout(height=460, title=f"{primary} — Close Price")
        fig.update_layout(**layout)
        # fix #2 — push range selector below the title with y offset
        fig.update_xaxes(rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#161B22", activecolor="#8B5CF6", font=dict(color="#E6EDF3"),
            y=1.06, x=0.01, xanchor="left",
        ))
        st.plotly_chart(fig, use_container_width=True)
    elif not selected:
        st.info("← Select at least one stock ticker to get started.")

# ── 6. Comparison chart (if 2+ tickers) — with description (#7) ──
if len(comparison) >= 2:
    st.markdown("---")
    st.markdown("#### Normalized Price Comparison")
    st.caption("All prices rebased to 1.0 at the start of the selected period for easy cross‑stock comparison.")
    fig = go.Figure()
    for i, (name, d) in enumerate(comparison.items()):
        norm = d["Close"] / d["Close"].iloc[0]
        fig.add_trace(go.Scatter(x=d["Date"], y=norm, name=name,
                                  line=dict(color=LINE_COLORS[i%len(LINE_COLORS)], width=2)))
    fig.update_layout(**_chart_layout(height=340, title="Normalized Price (base = 1.0)"))
    fig.update_yaxes(title_text="Normalized Price")
    st.plotly_chart(fig, use_container_width=True)

# ── 7. Backtest Results  (fix #5 — moved ABOVE indicators) ──
if "bt" in st.session_state:
    st.markdown("---")
    bt_model = st.session_state.get("bt_model", "XGBoost")
    st.markdown(f"#### Backtest Results — {bt_model}")
    st.caption("Historical walk-forward test: model trained on 80 % of data, evaluated on the remaining 20 %.")
    bt = st.session_state["bt"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{bt['mae']:.4f}", help="Mean Absolute Error of predicted daily returns vs actual.")
    m2.metric("Directional Accuracy", f"{bt['dir_acc']:.1%}", help="Percentage of days the model correctly predicted up or down.")
    m3.metric("Sharpe Ratio", f"{bt['sharpe']:.2f}", help="Risk-adjusted return. Above 1.0 is good, above 2.0 is excellent.")
    m4.metric("Max Drawdown", f"{bt['max_drawdown']:.1%}", help="Largest peak-to-trough portfolio drop during the test period.")

    bc1, bc2 = st.columns(2)
    with bc1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=bt["portfolio"], name="Portfolio", line=dict(color="#3FB950", width=2)))
        fig.add_hline(y=10000, line_dash="dash", line_color="#484F58")
        fig.update_layout(**_chart_layout(height=300, title="Portfolio Equity Curve ($10,000 start)"))
        fig.update_yaxes(title_text="Value ($)")
        st.plotly_chart(fig, use_container_width=True)
    with bc2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=bt["y_test"], name="Actual", line=dict(color="#8B949E", width=1)))
        fig.add_trace(go.Scatter(y=bt["preds"], name="Predicted", line=dict(color="#58A6FF", width=1.5)))
        fig.update_layout(**_chart_layout(height=300, title="Actual vs Predicted Returns"))
        st.plotly_chart(fig, use_container_width=True)

# ── 8. Technical Indicators  (fix #4 — 2×2 grid, better colors) ──
if selected and selected[0] in comparison:
    st.markdown("---")
    primary = selected[0]
    st.markdown(f"#### Technical Indicators — {primary}")
    st.caption("Key momentum, volatility, and volume signals derived from the stock's price history.")

    try:
        feat_df = get_featured(TICKERS[primary], period)
        dates = feat_df["Date"]

        # Row 1 — RSI + MACD
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=feat_df["RSI14"], name="RSI(14)",
                                      line=dict(color="#D2A8FF", width=2)))
            fig.add_hline(y=70, line_dash="dash", line_color="#FF7B72",
                          annotation_text="Overbought (70)", annotation_font_color="#FF7B72")
            fig.add_hline(y=30, line_dash="dash", line_color="#7EE787",
                          annotation_text="Oversold (30)", annotation_font_color="#7EE787")
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(139,92,246,0.06)", line_width=0)
            fig.update_layout(**_chart_layout(height=300, title="RSI (14) — Relative Strength Index"))
            st.plotly_chart(fig, use_container_width=True)

        with r1c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=feat_df["MACD"], name="MACD",
                                      line=dict(color="#58A6FF", width=2)))
            fig.add_trace(go.Scatter(x=dates, y=feat_df["Signal"], name="Signal",
                                      line=dict(color="#F0883E", width=2)))
            hist = feat_df["MACD"] - feat_df["Signal"]
            h_colors = ["#3FB950" if v >= 0 else "#F85149" for v in hist]
            fig.add_trace(go.Bar(x=dates, y=hist, name="Histogram",
                                  marker_color=h_colors, opacity=0.5))
            fig.update_layout(**_chart_layout(height=300, title="MACD — Moving Avg Convergence / Divergence"))
            st.plotly_chart(fig, use_container_width=True)

        # Row 2 — Bollinger + Volume
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            close = feat_df["Close"]; rm = close.rolling(20).mean(); rs = close.rolling(20).std()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates, y=rm+2*rs, line=dict(width=0),
                                      showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=dates, y=rm-2*rs, fill="tonexty",
                                      fillcolor="rgba(88,166,255,0.12)",
                                      line=dict(width=0), name="±2σ Band"))
            fig.add_trace(go.Scatter(x=dates, y=close, name="Close",
                                      line=dict(color="#58A6FF", width=2)))
            fig.add_trace(go.Scatter(x=dates, y=rm, name="SMA(20)",
                                      line=dict(color="#F0883E", width=1.5, dash="dot")))
            fig.update_layout(**_chart_layout(height=300, title="Bollinger Bands — Volatility Envelope"))
            st.plotly_chart(fig, use_container_width=True)

        with r2c2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dates, y=feat_df["Volume"], name="Volume",
                                  marker_color="#56D4DD", opacity=0.6))
            fig.update_layout(**_chart_layout(height=300, title="Daily Trading Volume"))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render indicators: {e}")
