import json
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def offers_to_dataframe(offers: list[dict]) -> pd.DataFrame:
    """Convert list of analyzed offers to a clean DataFrame."""
    rows = []
    for o in offers:
        if o.get("status") == "success":
            rows.append({
                "Titre": o.get("title", "N/A"),
                "Entreprise": o.get("company", "N/A"),
                "Localisation": o.get("location", "N/A"),
                "Contrat": o.get("contract_type", "N/A"),
                "Niveau": o.get("level", "N/A"),
                "Salaire": o.get("salary_range", "N/A"),
                "Expérience": o.get("experience_required", "N/A"),
                "Stack": ", ".join(o.get("tech_stack", [])),
                "Compétences": ", ".join(o.get("required_skills", [])),
                "URL": o.get("url", ""),
            })
    return pd.DataFrame(rows)


def save_results(offers: list[dict], fmt: str = "json") -> str:
    """Save results to data/ folder. Returns file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if fmt == "json":
        path = DATA_DIR / f"offers_{ts}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(offers, f, ensure_ascii=False, indent=2)
    elif fmt == "csv":
        df = offers_to_dataframe(offers)
        path = DATA_DIR / f"offers_{ts}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)
