import pandas as pd


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
                "Stack": ", ".join(s for s in (o.get("tech_stack") or []) if s),
                "Compétences": ", ".join(s for s in (o.get("required_skills") or []) if s),
                "URL": o.get("url", ""),
            })
    return pd.DataFrame(rows)
