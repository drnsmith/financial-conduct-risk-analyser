"""
screener.py
===========
Screens a list of NZX/ASX companies and returns a ranked risk table.
"""

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.quantitative.scores import score_company, QuantitativeScores
import logging

log = logging.getLogger(__name__)

# Default watchlist — major NZX and ASX listed companies
NZX_TICKERS = [
    "AIR.NZ", "ANZ.NZ", "AIA.NZ", "ATM.NZ", "CEN.NZ",
    "EBO.NZ", "FBU.NZ", "FPH.NZ", "GMT.NZ", "HLG.NZ",
    "IFT.NZ", "MCY.NZ", "MEL.NZ", "MFT.NZ", "MPG.NZ",
    "NZR.NZ", "OCA.NZ", "PCT.NZ", "PFI.NZ", "POT.NZ",
    "RYM.NZ", "SKC.NZ", "SKT.NZ", "SUM.NZ", "TPW.NZ",
    "THL.NZ", "VCT.NZ", "WBC.NZ", "WHS.NZ",
]

ASX_TICKERS = [
    "CBA.AX", "BHP.AX", "CSL.AX", "NAB.AX", "WBC.AX",
    "ANZ.AX", "WOW.AX", "WES.AX", "MQG.AX", "RIO.AX",
    "TLS.AX", "FMG.AX", "TCL.AX", "AMC.AX", "ALL.AX",
    "APA.AX", "ASX.AX", "BXB.AX", "COL.AX", "GMG.AX",
    "IAG.AX", "JHX.AX", "MIN.AX", "NCM.AX", "ORG.AX",
    "QBE.AX", "REA.AX", "RMD.AX", "SEK.AX", "SHL.AX",
    "STO.AX", "SUN.AX", "TAH.AX", "TWE.AX", "WPL.AX",
]


def screen_companies(
    tickers: list[str],
    exchange: str,
    max_workers: int = 5
) -> pd.DataFrame:
    """Score a list of companies in parallel and return ranked DataFrame."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_company, t, exchange): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                r: QuantitativeScores = future.result()
                results.append({
                    "ticker":          r.ticker,
                    "company":         r.company_name,
                    "exchange":        r.exchange,
                    "altman_z":        r.altman_z,
                    "altman_zone":     r.altman_zone,
                    "beneish_m":       r.beneish_m,
                    "beneish_flag":    r.beneish_flag,
                    "piotroski_f":     r.piotroski_f,
                    "piotroski_tier":  r.piotroski_tier,
                    "risk_tier":       r.risk_tier,
                    "risk_drivers":    "; ".join(r.risk_drivers),
                    "validity_flags":  len(r.validity_flags),
                })
            except Exception as e:
                log.warning(f"Failed {ticker}: {e}")

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Risk tier ordering
    tier_order = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}
    df["tier_rank"] = df["risk_tier"].map(tier_order)
    df = df.sort_values(["tier_rank", "altman_z"], ascending=[True, True])
    df = df.drop(columns=["tier_rank"])

    return df


def run_full_screen() -> pd.DataFrame:
    """Screen both NZX and ASX."""
    import logging
    logging.basicConfig(level=logging.WARNING)

    print("Screening NZX companies...")
    nzx = screen_companies(NZX_TICKERS, "NZX")

    print("Screening ASX companies...")
    asx = screen_companies(ASX_TICKERS, "ASX")

    combined = pd.concat([nzx, asx], ignore_index=True)
    combined.to_parquet("data/processed/screen_results.parquet", index=False)
    combined.to_csv("data/processed/screen_results.csv", index=False)
    print(f"✓ Screened {len(combined)} companies")
    print(combined[["ticker","company","risk_tier","altman_zone","beneish_flag","piotroski_tier"]].to_string(index=False))
    return combined


if __name__ == "__main__":
    run_full_screen()
