"""
scores.py
=========
Computes Altman Z-score, Beneish M-score, and Piotroski F-score
from live financial data pulled via yfinance.

Each score answers a different question:
  - Altman Z:    Is this company at risk of bankruptcy?
  - Beneish M:   Is this company manipulating its earnings?
  - Piotroski F: Is this company financially strong?
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Optional
import logging

log = logging.getLogger(__name__)


@dataclass
class QuantitativeScores:
    ticker: str
    company_name: str
    exchange: str
    altman_z: Optional[float]
    altman_zone: str          # Distress / Grey / Safe
    beneish_m: Optional[float]
    beneish_flag: str         # Manipulator / Grey / Non-manipulator
    piotroski_f: Optional[int]
    piotroski_tier: str       # Weak / Moderate / Strong
    risk_tier: str            # High / Medium / Low
    risk_drivers: list
    validity_flags: list      # Valimetrica layer
    raw: dict                 # Raw financials for audit trail


def fetch_financials(ticker: str) -> dict:
    """Pull financial statements from yfinance."""
    t = yf.Ticker(ticker)
    try:
        info        = t.info or {}
        income      = t.financials
        balance     = t.balance_sheet
        cashflow    = t.cashflow
        return {
            "info":     info,
            "income":   income,
            "balance":  balance,
            "cashflow": cashflow,
        }
    except Exception as e:
        log.warning(f"Failed to fetch {ticker}: {e}")
        return {}


def safe_get(df, row, col=0):
    """Safely extract a value from a financial dataframe."""
    try:
        if df is None or df.empty:
            return np.nan
        if row not in df.index:
            return np.nan
        val = df.loc[row].iloc[col]
        return float(val) if pd.notna(val) else np.nan
    except Exception:
        return np.nan


def altman_z(raw: dict) -> tuple[Optional[float], str, list]:
    """
    Altman Z-score for public companies.
    Z > 2.99: Safe zone
    1.81 < Z < 2.99: Grey zone
    Z < 1.81: Distress zone
    """
    bs = raw.get("balance")
    inc = raw.get("income")
    info = raw.get("info", {})
    drivers = []

    total_assets     = safe_get(bs, "Total Assets")
    total_liab       = safe_get(bs, "Total Liabilities Net Minority Interest")
    current_assets   = safe_get(bs, "Current Assets")
    current_liab     = safe_get(bs, "Current Liabilities")
    retained_earn    = safe_get(bs, "Retained Earnings")
    ebit             = safe_get(inc, "EBIT")
    revenue          = safe_get(inc, "Total Revenue")
    market_cap       = info.get("marketCap", np.nan)

    if any(np.isnan(x) for x in [total_assets, total_liab, current_assets,
                                   current_liab, retained_earn, ebit, revenue]):
        return None, "Insufficient data", drivers

    if total_assets == 0:
        return None, "Insufficient data", drivers

    working_capital  = current_assets - current_liab
    book_equity      = total_assets - total_liab

    x1 = working_capital / total_assets
    x2 = retained_earn / total_assets
    x3 = ebit / total_assets
    x4 = (market_cap if not np.isnan(market_cap) else book_equity) / total_liab
    x5 = revenue / total_assets

    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5

    if x3 < 0:
        drivers.append("Negative EBIT / total assets")
    if x1 < 0:
        drivers.append("Negative working capital")
    if x2 < 0.1:
        drivers.append("Low retained earnings ratio")

    if z > 2.99:
        zone = "Safe"
    elif z > 1.81:
        zone = "Grey"
    else:
        zone = "Distress"

    return round(z, 3), zone, drivers


def beneish_m(raw: dict) -> tuple[Optional[float], str, list]:
    """
    Beneish M-score (8-variable model).
    M > -1.78: Likely manipulator
    M < -2.22: Non-manipulator
    Between: Grey zone

    Note: Calibrated on US data (1990s). Validity flag raised for
    NZ/AU companies — treat as indicative, not definitive.
    """
    bs  = raw.get("balance")
    inc = raw.get("income")
    cf  = raw.get("cashflow")
    drivers = []

    if bs is None or inc is None or bs.shape[1] < 2:
        return None, "Insufficient data", drivers

    # Current year (col 0) and prior year (col 1)
    rev_t      = safe_get(inc, "Total Revenue", 0)
    rev_t1     = safe_get(inc, "Total Revenue", 1)
    cogs_t     = safe_get(inc, "Cost Of Revenue", 0)
    cogs_t1    = safe_get(inc, "Cost Of Revenue", 1)
    ar_t       = safe_get(bs, "Accounts Receivable", 0)
    ar_t1      = safe_get(bs, "Accounts Receivable", 1)
    ta_t       = safe_get(bs, "Total Assets", 0)
    ta_t1      = safe_get(bs, "Total Assets", 1)
    ppe_t      = safe_get(bs, "Net PPE", 0)
    ppe_t1     = safe_get(bs, "Net PPE", 1)
    sga_t      = safe_get(inc, "Selling General And Administration", 0)
    sga_t1     = safe_get(inc, "Selling General And Administration", 1)
    dep_t      = safe_get(cf,  "Depreciation And Amortization", 0)
    dep_t1     = safe_get(cf,  "Depreciation And Amortization", 1)
    ltd_t      = safe_get(bs, "Long Term Debt", 0)
    ltd_t1     = safe_get(bs, "Long Term Debt", 1)
    ca_t       = safe_get(bs, "Current Assets", 0)
    ca_t1      = safe_get(bs, "Current Assets", 1)
    cl_t       = safe_get(bs, "Current Liabilities", 0)
    cl_t1      = safe_get(bs, "Current Liabilities", 1)
    cfo_t      = safe_get(cf,  "Operating Cash Flow", 0)
    ni_t       = safe_get(inc, "Net Income", 0)

    required = [rev_t, rev_t1, ar_t, ar_t1, ta_t, ta_t1, cogs_t, cogs_t1]
    if any(np.isnan(x) or x == 0 for x in required):
        return None, "Insufficient data", drivers

    dsri = (ar_t/rev_t) / (ar_t1/rev_t1) if rev_t1 != 0 else np.nan
    gmi  = ((rev_t1-cogs_t1)/rev_t1) / ((rev_t-cogs_t)/rev_t) if (rev_t != 0 and rev_t1 != 0) else np.nan
    aqi  = (1 - (ca_t + ppe_t)/ta_t) / (1 - (ca_t1 + ppe_t1)/ta_t1) if ta_t1 != 0 else np.nan
    sgi  = rev_t / rev_t1 if rev_t1 != 0 else np.nan
    depi = (dep_t1/(dep_t1+ppe_t1)) / (dep_t/(dep_t+ppe_t)) if (dep_t+ppe_t) != 0 else np.nan
    sgai = (sga_t/rev_t) / (sga_t1/rev_t1) if (sga_t1 != np.nan and rev_t1 != 0) else np.nan
    lvgi = ((ltd_t+cl_t)/ta_t) / ((ltd_t1+cl_t1)/ta_t1) if ta_t1 != 0 else np.nan
    tata = (ni_t - cfo_t) / ta_t if (not np.isnan(ni_t) and not np.isnan(cfo_t)) else np.nan

    components = [dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata]
    if sum(np.isnan(x) for x in components) > 3:
        return None, "Insufficient data", drivers

    def s(x): return x if not np.isnan(x) else 0

    m = (-4.84
         + 0.920*s(dsri)
         + 0.528*s(gmi)
         + 0.404*s(aqi)
         + 0.892*s(sgi)
         + 0.115*s(depi)
         - 0.172*s(sgai)
         + 4.679*s(tata)
         - 0.327*s(lvgi))

    if not np.isnan(s(dsri)) and dsri > 1.465:
        drivers.append("High days sales in receivables index (DSRI)")
    if not np.isnan(s(tata)) and tata > 0.031:
        drivers.append("High total accruals to total assets")
    if not np.isnan(s(sgi)) and sgi > 1.607:
        drivers.append("High sales growth index")

    if m > -1.78:
        flag = "Likely manipulator"
    elif m > -2.22:
        flag = "Grey zone"
    else:
        flag = "Non-manipulator"

    return round(m, 3), flag, drivers


def piotroski_f(raw: dict) -> tuple[Optional[int], str, list]:
    """
    Piotroski F-score (0-9).
    0-2: Weak
    3-6: Moderate
    7-9: Strong
    """
    bs  = raw.get("balance")
    inc = raw.get("income")
    cf  = raw.get("cashflow")
    drivers = []
    score = 0

    if bs is None or inc is None:
        return None, "Insufficient data", drivers

    roa      = safe_get(inc, "Net Income", 0) / (safe_get(bs, "Total Assets", 0) or 1)
    cfo      = safe_get(cf,  "Operating Cash Flow", 0)
    ta       = safe_get(bs,  "Total Assets", 0)
    roa_t1   = safe_get(inc, "Net Income", 1) / (safe_get(bs, "Total Assets", 1) or 1)
    ltd_t    = safe_get(bs,  "Long Term Debt", 0)
    ltd_t1   = safe_get(bs,  "Long Term Debt", 1)
    cr_t     = (safe_get(bs, "Current Assets", 0) /
                (safe_get(bs, "Current Liabilities", 0) or 1))
    cr_t1    = (safe_get(bs, "Current Assets", 1) /
                (safe_get(bs, "Current Liabilities", 1) or 1))
    shares_t = safe_get(bs, "Ordinary Shares Number", 0)
    shares_t1= safe_get(bs, "Ordinary Shares Number", 1)
    gm_t     = ((safe_get(inc, "Total Revenue", 0) - safe_get(inc, "Cost Of Revenue", 0)) /
                (safe_get(inc, "Total Revenue", 0) or 1))
    gm_t1    = ((safe_get(inc, "Total Revenue", 1) - safe_get(inc, "Cost Of Revenue", 1)) /
                (safe_get(inc, "Total Revenue", 1) or 1))
    at_t     = safe_get(inc, "Total Revenue", 0) / (ta or 1)
    at_t1    = safe_get(inc, "Total Revenue", 1) / (safe_get(bs, "Total Assets", 1) or 1)

    # Profitability
    if roa > 0:       score += 1; drivers.append("✓ Positive ROA")
    if cfo > 0:       score += 1; drivers.append("✓ Positive CFO")
    if roa > roa_t1:  score += 1; drivers.append("✓ Improving ROA")
    if ta != 0 and cfo/ta > roa: score += 1; drivers.append("✓ CFO > ROA (accrual quality)")

    # Leverage & liquidity
    if not np.isnan(ltd_t) and not np.isnan(ltd_t1) and ltd_t < ltd_t1:
        score += 1; drivers.append("✓ Declining long-term debt")
    if cr_t > cr_t1:  score += 1; drivers.append("✓ Improving current ratio")
    if not np.isnan(shares_t) and not np.isnan(shares_t1) and shares_t <= shares_t1:
        score += 1; drivers.append("✓ No share dilution")

    # Operating efficiency
    if gm_t > gm_t1:  score += 1; drivers.append("✓ Improving gross margin")
    if at_t > at_t1:  score += 1; drivers.append("✓ Improving asset turnover")

    if score >= 7:    tier = "Strong"
    elif score >= 3:  tier = "Moderate"
    else:             tier = "Weak"

    return score, tier, drivers


def composite_risk_tier(z_zone: str, m_flag: str, f_tier: str) -> tuple[str, list]:
    """Combine three scores into a single risk tier with reasoning."""
    flags = []
    high_signals = 0

    if z_zone == "Distress":   high_signals += 2; flags.append("Altman Z in distress zone")
    elif z_zone == "Grey":     high_signals += 1; flags.append("Altman Z in grey zone")

    if m_flag == "Likely manipulator": high_signals += 2; flags.append("Beneish M flags manipulation risk")
    elif m_flag == "Grey zone":        high_signals += 1; flags.append("Beneish M in grey zone")

    if f_tier == "Weak":       high_signals += 1; flags.append("Piotroski F score weak")

    if high_signals >= 3:   tier = "High"
    elif high_signals >= 1: tier = "Medium"
    else:                   tier = "Low"

    return tier, flags


def valimetrica_flags(exchange: str, z: Optional[float], m: Optional[float]) -> list:
    """
    Valimetrica validity layer.
    Flag where measurement instruments may not transfer to NZ/AU context.
    """
    flags = []
    if exchange in ["NZX", "ASX"]:
        flags.append(
            "Beneish M-score calibrated on US data (1990s Compustat). "
            "Construct validity unconfirmed for NZ/AU markets. Treat as indicative."
        )
    if exchange == "NZX":
        flags.append(
            "Altman Z-score developed on US manufacturing firms. "
            "NZX market structure differs — financial sector and small-cap firms "
            "may show systematic score bias."
        )
    if z is not None and z > 5:
        flags.append(
            "Altman Z > 5 may reflect market cap inflation in growth stocks "
            "rather than genuine financial safety."
        )
    return flags


def score_company(ticker: str, exchange: str = "ASX") -> QuantitativeScores:
    """Main entry point — score a single company."""
    raw = fetch_financials(ticker)
    if not raw:
        return QuantitativeScores(
            ticker=ticker, company_name=ticker, exchange=exchange,
            altman_z=None, altman_zone="No data",
            beneish_m=None, beneish_flag="No data",
            piotroski_f=None, piotroski_tier="No data",
            risk_tier="Unknown", risk_drivers=[], validity_flags=[], raw={}
        )

    company_name = raw.get("info", {}).get("longName", ticker)

    z_score, z_zone, z_drivers   = altman_z(raw)
    m_score, m_flag, m_drivers   = beneish_m(raw)
    f_score, f_tier, f_drivers   = piotroski_f(raw)

    risk_tier, risk_flags = composite_risk_tier(z_zone, m_flag, f_tier)
    all_drivers = z_drivers + m_drivers + risk_flags
    validity    = valimetrica_flags(exchange, z_score, m_score)

    return QuantitativeScores(
        ticker=ticker,
        company_name=company_name,
        exchange=exchange,
        altman_z=z_score,
        altman_zone=z_zone,
        beneish_m=m_score,
        beneish_flag=m_flag,
        piotroski_f=f_score,
        piotroski_tier=f_tier,
        risk_tier=risk_tier,
        risk_drivers=all_drivers,
        validity_flags=validity,
        raw={"info": raw.get("info", {})}
    )
