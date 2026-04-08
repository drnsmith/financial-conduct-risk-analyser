# Financial Conduct Risk Analyser

**Live demo:** https://huggingface.co/spaces/drnsmith/financial-conduct-risk-analyser

---

## Why I built this

Financial regulators face a problem that is structurally similar to the one I work on in my research: the instruments used to assess risk are often not validated for the context in which they are applied. The Beneish M-score was calibrated on 1990s US data. The Altman Z-score was developed on 66 US manufacturing firms in 1968. Both are routinely applied to NZX and ASX listed companies as though the population validity problem doesn't exist.

I built this project because I wanted to work through what a rigorous, validity-aware financial conduct risk assessment actually looks like in practice — combining quantitative financial scoring with qualitative document analysis, and being explicit about where the measurement instruments are and aren't trustworthy.

The FMA's mandate is to promote fair, efficient, and transparent financial markets. That mandate requires the ability to identify companies showing early indicators of financial stress, earnings manipulation, or governance failure — and to do so with appropriate epistemic humility about what the scores can and cannot tell you. This platform is my attempt to build that capability properly.

---

## What it does

**Quantitative layer** — screens NZX and ASX listed companies using three complementary financial risk scores:

- **Altman Z-score** — bankruptcy risk. Z < 1.81 = distress zone, Z > 2.99 = safe zone. Developed on US manufacturing firms; validity flags raised for NZ/AU financial and infrastructure companies.
- **Beneish M-score** — earnings manipulation probability. M > -1.78 = likely manipulator. Calibrated on US GAAP data; treated as indicative rather than definitive for NZ IFRS companies.
- **Piotroski F-score** — financial strength (0-9). The most transferable instrument across accounting regimes; primary signal recommended for NZ/AU screening.

**Qualitative layer** — ingests financial regulatory documents (FMA enforcement notices, RBNZ financial stability reports, UNCTAD World Investment Reports, company annual reports) and uses Claude to extract structured risk signals mapped to six categories: conduct risk, model risk, operational risk, financial risk, regulatory risk, and systemic risk. Every signal includes a direct evidence quote and confidence score.

**Valimetrica validity layer** — explicitly flags where the quantitative instruments may not transfer to the NZ/AU context. Distinguishes between screening instruments (appropriate for prioritising further investigation) and decision instruments (requiring additional validation before regulatory action).

---

## Key findings from initial screen

Screening 64 NZX and ASX companies identified several patterns worth noting:

**High risk flags:** Ryman Healthcare (RYM.NZ), Oceania Healthcare (OCA.NZ), and Summerset (SUM.NZ) all show Altman Z in distress zone with weak Piotroski scores — consistent with the well-documented debt burden of NZ retirement village operators. Fletcher Building (FBU.NZ) and Infratil (IFT.NZ) also flag high.

**Validity illustration:** Fisher & Paykel Healthcare (FPH.NZ) is flagged as "Likely manipulator" by the Beneish M-score due to high Days Sales in Receivables Index (DSRI). This is a construct validity failure — FPH's long payment cycles reflect its medical device business model (hospital procurement), not earnings manipulation. The Valimetrica layer correctly identifies this as a population validity problem with the Beneish instrument.

**Document analysis:** 62 risk signals extracted from three regulatory documents — the FMA v IAG New Zealand enforcement judgement (2025), FMA DIMS Sector Insights (2024), and UNCTAD World Investment Report (2024). The IAG case alone yielded 29 signals across all six risk categories, with the $19.5M penalty and systemic pricing algorithm failures correctly classified as high-severity conduct and operational risk.

---

## Documents supported

- FMA enforcement notices and judgements
- RBNZ Financial Stability Reports
- UNCTAD World Investment Reports
- Company annual reports (NZX/ASX)
- APRA prudential guidance
- Any financial regulatory PDF

---

## Tech stack

- Python 3.11, yfinance 0.2.37 (live financial data)
- Anthropic Claude Haiku (document risk extraction)
- pdfplumber (PDF text extraction)
- Plotly Dash 4.1.0, dash-bootstrap-components
- pandas, numpy, pydantic

---

## Run locally

```bash
git clone https://github.com/drnsmith/financial-conduct-risk-analyser.git
cd financial-conduct-risk-analyser
pip install -r requirements.txt
cp .env.example .env  # Add your Anthropic API key

# Screen companies
PYTHONPATH=. python src/quantitative/screener.py

# Analyse documents
PYTHONPATH=. python -c "
from src.qualitative.document_analyser import analyse_document
result = analyse_document('path/to/document.pdf')
print(result['risk_breakdown'])
"

# Launch dashboard
PYTHONPATH=. python dashboard/app.py
# → http://127.0.0.1:8053
```

---

## Project structure

```
financial-conduct-risk-analyser/
├── src/
│   ├── quantitative/
│   │   ├── scores.py          # Altman Z, Beneish M, Piotroski F
│   │   └── screener.py        # NZX/ASX batch screener
│   └── qualitative/
│       └── document_analyser.py  # Claude-powered risk extraction
├── dashboard/
│   └── app.py                 # Plotly Dash, 5 tabs
├── data/
│   └── raw/documents/         # FMA, RBNZ, UNCTAD PDFs
└── requirements.txt
```

---

## Methodology note

This platform is designed as a **screening tool**, not a decision instrument. A High risk tier flag should be interpreted as a signal warranting further investigation, not as a definitive finding of financial distress or misconduct. The Valimetrica validity assessment tab makes this distinction explicit for each scoring instrument.

This distinction — between a screening instrument and a decision instrument — is the core methodological contribution of the Valimetrica framework to applied financial risk assessment.

---

## Author

Dr Natalya Smith — [github.com/drnsmith](https://github.com/drnsmith) · [medium.com/@NeverOblivious](https://medium.com/@NeverOblivious)
