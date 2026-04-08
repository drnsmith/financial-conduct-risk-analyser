"""
document_analyser.py
====================
Ingests PDF documents and extracts structured risk signals using Claude.
Handles: World Investment Reports, FMA enforcement notices,
         RBNZ financial stability reports, company annual reports,
         regulatory guidance documents.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional
import anthropic
import pdfplumber
import logging
from dotenv import load_dotenv
load_dotenv(override=True)

log = logging.getLogger(__name__)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

RISK_CATEGORIES = {
    "conduct_risk":      "Mis-selling, conflicts of interest, unfair treatment of customers, disclosure failures",
    "model_risk":        "Model validation failures, inappropriate use, overfitting, measurement invalidity",
    "operational_risk":  "Process failures, system outages, fraud, human error, third-party failures",
    "financial_risk":    "Liquidity, credit, market, capital adequacy",
    "regulatory_risk":   "Compliance failures, enforcement actions, licence conditions, regulatory change",
    "systemic_risk":     "Contagion, market-wide stress, interconnectedness, concentration",
}

EXTRACTION_PROMPT = """You are a senior financial risk analyst at a regulatory body.

Analyse the following document excerpt and extract structured risk information.

Document type: {doc_type}
Document: {doc_name}

Text excerpt:
{text}

Extract and return a JSON object with exactly this structure:
{{
  "risk_signals": [
    {{
      "risk_category": "one of: conduct_risk, model_risk, operational_risk, financial_risk, regulatory_risk, systemic_risk",
      "severity": "one of: High, Medium, Low",
      "description": "clear description of the risk signal in 1-2 sentences",
      "evidence": "direct quote or paraphrase from the text (max 50 words)",
      "page_reference": "page number if available, else null",
      "confidence": 0.0-1.0
    }}
  ],
  "key_entities": ["list of companies, regulators, or markets mentioned"],
  "time_period": "time period covered by this excerpt if mentioned",
  "document_summary": "2-3 sentence summary of what this excerpt covers"
}}

Return only valid JSON. No preamble, no explanation."""


def extract_text_from_pdf(pdf_path: str, max_pages: int = 50) -> list[dict]:
    """Extract text page by page from a PDF."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:max_pages]):
                text = page.extract_text()
                if text and len(text.strip()) > 100:
                    pages.append({
                        "page": i + 1,
                        "text": text.strip()
                    })
    except Exception as e:
        log.error(f"Failed to extract PDF {pdf_path}: {e}")
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 3) -> list[dict]:
    """Group pages into chunks for analysis."""
    chunks = []
    for i in range(0, len(pages), chunk_size):
        group = pages[i:i+chunk_size]
        chunks.append({
            "pages": f"{group[0]['page']}-{group[-1]['page']}",
            "text": "\n\n".join(p["text"] for p in group)
        })
    return chunks


def classify_document(filename: str) -> str:
    """Infer document type from filename."""
    fname = filename.lower()
    if "wir" in fname or "world investment" in fname:
        return "UN World Investment Report"
    elif "fma" in fname or "enforcement" in fname:
        return "FMA Enforcement Notice"
    elif "rbnz" in fname or "financial stability" in fname:
        return "RBNZ Financial Stability Report"
    elif "annual" in fname or "report" in fname:
        return "Company Annual Report"
    elif "apra" in fname:
        return "APRA Prudential Guidance"
    else:
        return "Financial/Regulatory Document"


def analyse_chunk(chunk: dict, doc_name: str, doc_type: str) -> Optional[dict]:
    """Send a chunk to Claude for risk extraction."""
    # Truncate to avoid token limits
    text = chunk["text"][:4000]

    prompt = EXTRACTION_PROMPT.format(
        doc_type=doc_type,
        doc_name=doc_name,
        text=text
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)

        result = json.loads(raw)
        result["pages"] = chunk["pages"]
        return result

    except json.JSONDecodeError as e:
        log.warning(f"JSON parse failed for pages {chunk['pages']}: {e}")
        return None
    except Exception as e:
        log.warning(f"Claude API error for pages {chunk['pages']}: {e}")
        return None


def analyse_document(pdf_path: str, max_chunks: int = 10) -> dict:
    """
    Full pipeline: PDF → text → chunks → Claude extraction → structured output.
    """
    path = Path(pdf_path)
    doc_name = path.stem
    doc_type = classify_document(path.name)

    log.info(f"Analysing: {doc_name} ({doc_type})")

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return {"error": "Could not extract text from PDF", "doc": doc_name}

    chunks = chunk_pages(pages)[:max_chunks]
    log.info(f"  Pages: {len(pages)} | Chunks: {len(chunks)}")

    all_signals = []
    all_entities = set()
    summaries = []

    for i, chunk in enumerate(chunks):
        log.info(f"  Analysing chunk {i+1}/{len(chunks)} (pages {chunk['pages']})")
        result = analyse_chunk(chunk, doc_name, doc_type)
        if result:
            all_signals.extend(result.get("risk_signals", []))
            all_entities.update(result.get("key_entities", []))
            if result.get("document_summary"):
                summaries.append(result["document_summary"])

    # Deduplicate and sort by severity
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    all_signals.sort(key=lambda x: severity_order.get(x.get("severity", "Low"), 2))

    return {
        "document":      doc_name,
        "document_type": doc_type,
        "path":          str(pdf_path),
        "pages_analysed": len(pages),
        "risk_signals":  all_signals,
        "key_entities":  sorted(all_entities),
        "high_risks":    [s for s in all_signals if s.get("severity") == "High"],
        "summary":       summaries[0] if summaries else "No summary extracted",
        "total_signals": len(all_signals),
        "risk_breakdown": {
            cat: len([s for s in all_signals if s.get("risk_category") == cat])
            for cat in RISK_CATEGORIES
        }
    }
