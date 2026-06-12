import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from kiteconnect import KiteConnect
import datetime

# 1. ENFORCE INDUSTRIAL MINIMALIST DESIGN WITH EXTENDED CHART HEIGHTS
st.set_page_config(page_title="ALPHA QUANT TERMINAL", layout="wide")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F0F11 !important;
        color: #FFFFFF !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    h1, h2, h3, p, span {
        color: #FFFFFF !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #16161A;
        border-bottom: 1px solid #333333;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888888 !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #FFFFFF !important;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-family: monospace !important;
    }
    table {
        border-collapse: collapse !important;
        width: 100%;
        background-color: #0F0F11;
    }
    th, td {
        border: 1px solid #333333 !important;
        padding: 8px !important;
        text-align: center !important;
        color: #FFFFFF !important;
    }
    iframe {
        border: 1px solid #333333 !important;
        border-radius: 4px;
        background-color: #16161A;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MATHEMATICAL MODULE (BLACK-SCHOLES ENGINE)
# ==============================================================================
def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return {"delta": 1.0 if S > K else 0.0, "gamma": 0.0, "theta": 0.0}
        else:
            return {"delta": -1.0 if S < K else 0.0, "gamma": 0.0, "theta": 0.0}
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = -norm.cdf(-d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
    return {"delta": round(delta, 3), "gamma": round(gamma, 6), "theta": round(theta, 2)}

# ==============================================================================
# 3. LIVE DATA EXTRACTOR (SPOT, FUTURE, VIX, CHAIN)
# ==============================================================================
def fetch_live_option_chain(underlying="NIFTY"):
    if not st.session_state.authenticated or st.session_state.kite is None:
        mock_strikes = [23100, 23150, 23200, 23250, 23300, 23350, 23400]
        rows = []
        for strike in mock_strikes:
            c_gree = calculate_greeks(23240.50, strike, 4/365, 0.07, 0.155, "call")
            p_gree = calculate_greeks(23240.50, strike, 4/365, 0.07, 0.155, "put")
            rows.append({
                "Put_OI_M": 4.5 if strike < 23250 else 1.2,
                "Put_Chg": "+120K" if strike == 23200 else "+15K",
                "Put_Delta": p_gree["delta"],
                "Strike": strike,
                "Call_Delta": c_gree["delta"],
                "Call_Chg": "+4.1M" if strike == 23300 else "+450K",
                "Call_OI_M": 18.2 if strike > 23250 else 0.8
            })
        return pd.DataFrame(rows), 23240.50, 23255.20, 15.50

    try:
        kite = st.session_state.kite
        vix_ticker = "NSE:INDIA_VIX"
        spot_ticker = "NSE:NIFTY_50" if underlying == "NIFTY" else "NSE:NIFTY_BANK"
        
        all_inst = pd.DataFrame(kite.instruments("NFO"))
        fut_ins = all_inst[(all_inst["name"] == underlying) & (all_inst["instrument_type"] == "FUT")]
        fut_ins["expiry"] = pd.to_datetime(fut_ins["expiry"])
        closest_fut_symbol = fut_ins.loc[fut_ins["expiry"].idxmin()]["tradingsymbol"]
        fut_ticker = f"NFO:{closest_fut_symbol}"
        
        quotes = kite.quote([spot_ticker, fut_ticker, vix_ticker])
        
        spot_price = quotes[spot_ticker]["last_price"]
        future_price = quotes[fut_ticker]["last_price"]
        vix_value = quotes[vix_ticker]["last_price"]
        
        opt_ins = all_inst[(all_inst["name"] == underlying) & (all_inst["instrument_type"].
