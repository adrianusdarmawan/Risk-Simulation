from __future__ import annotations

import io
import zipfile
from datetime import date, timedelta
from typing import Any

import requests

from .db import upsert_source

CFPB_API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
VCDB_ZIP = "https://raw.githubusercontent.com/vz-risk/VCDB/master/data/csv/vcdb.csv.zip"
FCA_LATEST = "https://www.fca.org.uk/data/complaints-data/firm-level"


def refresh_cfpb() -> dict[str, Any]:
    today = date.today(); start = today - timedelta(days=365)
    params = {"date_received_min": start.isoformat(), "date_received_max": (today + timedelta(days=1)).isoformat(), "size": 1, "no_highlight": "true"}
    r = requests.get(CFPB_API, params=params, timeout=30, headers={"User-Agent":"RiskPulseLab/0.1 public-data research"}); r.raise_for_status()
    js = r.json(); hits = js.get("hits",{}).get("total",{}); count = hits.get("value") if isinstance(hits,dict) else hits; meta = js.get("_meta",{})
    payload = {"window": f"{start.isoformat()} to {today.isoformat()}", "last_updated": meta.get("last_updated"), "total_record_count": meta.get("total_record_count"), "note": "Daily-updating public complaint database. Not a statistical sample; volume must be contextualised by market size/exposure."}
    upsert_source("CFPB Consumer Complaint Database", "ok", int(count) if count is not None else None, payload)
    return {"source":"CFPB Consumer Complaint Database","status":"ok","record_count":count,"payload":payload}


def refresh_vcdb() -> dict[str, Any]:
    r = requests.get(VCDB_ZIP, timeout=60, headers={"User-Agent":"RiskPulseLab/0.1 public-data research"}); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names: raise RuntimeError("VCDB zip contained no CSV")
        with z.open(names[0]) as f:
            header = f.readline().decode("utf-8", errors="replace").rstrip("\n\r"); columns = len(header.split(",")); row_count = sum(1 for _ in f)
    payload = {"columns": int(columns), "note": "VERIS Community Database is useful for incident patterns/causes but is a convenience sample, not an unbiased denominator for company incident probability."}
    upsert_source("VERIS Community Database (VCDB)", "ok", int(row_count), payload)
    return {"source":"VERIS Community Database (VCDB)","status":"ok","record_count":int(row_count),"payload":payload}


def refresh_fca_status() -> dict[str, Any]:
    r = requests.get(FCA_LATEST, timeout=30, headers={"User-Agent":"RiskPulseLab/0.1 public-data research"}); r.raise_for_status()
    payload = {"http_last_modified": r.headers.get("Last-Modified"), "note": "App ships a reproducible 2024H1-2025H2 FCA product panel for backtesting and checks the FCA source page for availability. A future adapter can ingest each newly published workbook after schema validation."}
    upsert_source("FCA complaints data", "ok", None, payload)
    return {"source":"FCA complaints data","status":"ok","record_count":None,"payload":payload}


def refresh_all() -> list[dict[str, Any]]:
    out=[]
    for fn,name in [(refresh_cfpb,"CFPB"),(refresh_vcdb,"VCDB"),(refresh_fca_status,"FCA")]:
        try: out.append(fn())
        except Exception as exc:
            payload={"error":str(exc)[:500],"note":"Last known data remains usable; refresh can be retried later."}; upsert_source(name,"error",None,payload); out.append({"source":name,"status":"error","error":str(exc)})
    return out
