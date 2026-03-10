from collections import Counter
from typing import Optional
import yaml
from pathlib import Path
from src.scraper import scrape_job_offer
from src.chains import analyze_offer_chain, gap_analysis_chain

CONFIG_DIR = Path(__file__).parent.parent / "config"

_BLOCKED_HOSTS = {
    "linkedin.com": (
        "LinkedIn bloque le scraping. "
        "Ouvre l'offre dans ton navigateur, sélectionne tout le texte (Ctrl+A) et colle-le dans "
        "la zone « Ou colle du texte d'offre directement »."
    ),
    "indeed.com": (
        "Indeed bloque le scraping. "
        "Colle le texte de l'offre manuellement via la zone de texte."
    ),
}


def load_profile() -> dict:
    """Load user profile from YAML config."""
    with open(CONFIG_DIR / "profile.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_profile(yaml_content: str) -> None:
    """Write raw YAML string back to config/profile.yaml."""
    with open(CONFIG_DIR / "profile.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)


def _blocked_reason(url: str) -> Optional[str]:
    for host, msg in _BLOCKED_HOSTS.items():
        if host in url:
            return msg
    return None


def process_offer(url: str) -> dict:
    """Scrape + analyze a single job offer URL."""
    blocked = _blocked_reason(url)
    if blocked:
        return {"error": blocked, "url": url}

    content = scrape_job_offer(url)
    if not content:
        return {"error": "Scraping échoué : page inaccessible ou vide.", "url": url}

    analysis = analyze_offer_chain(content)
    if not analysis:
        return {"error": "LLM parsing failed", "url": url}

    analysis["url"] = url
    analysis["status"] = "success"
    return analysis


def compute_match_score(offer: dict, profile: dict) -> int:
    """
    Compute the % of offer skills already covered by the profile.
    Uses case-insensitive substring matching (bidirectional).
    """
    profile_skills = {
        s.lower()
        for cat in profile.get("skills", {}).values()
        for s in cat
    }
    offer_skills = [
        s.lower()
        for s in offer.get("required_skills", []) + offer.get("tech_stack", [])
    ]
    if not offer_skills:
        return 0
    matches = sum(
        1 for s in offer_skills
        if any(p in s or s in p for p in profile_skills)
    )
    return round(matches / len(offer_skills) * 100)


def run_gap_analysis(offers: list[dict], profile: dict = None) -> str:
    """Aggregate skills from successful offers and run gap analysis."""
    if profile is None:
        profile = load_profile()

    all_skills = []
    for offer in offers:
        if offer.get("status") == "success":
            all_skills.extend(offer.get("required_skills", []))
            all_skills.extend(offer.get("tech_stack", []))

    if not all_skills:
        return "Aucune offre analysée avec succès pour générer un rapport."

    return gap_analysis_chain(profile, all_skills)


def get_top_skills(offers: list[dict], top_n: int = 15) -> list[tuple]:
    """Return the most requested skills across all analyzed offers."""
    all_skills = []
    for offer in offers:
        if offer.get("status") == "success":
            all_skills.extend(offer.get("required_skills", []))
            all_skills.extend(offer.get("tech_stack", []))
    return Counter(all_skills).most_common(top_n)
