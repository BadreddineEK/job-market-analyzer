from collections import Counter
from typing import Optional
import yaml
from pathlib import Path
from src.scraper import scrape_job_offer
from src.chains import analyze_offer_chain, gap_analysis_chain

CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_profile() -> dict:
    """Load user profile from YAML config."""
    with open(CONFIG_DIR / "profile.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_offer(url: str) -> dict:
    """
    Full pipeline for a single offer:
    1. Scrape the URL
    2. Analyze with LLM
    3. Return structured result
    """
    content = scrape_job_offer(url)
    if not content:
        return {"error": f"Could not scrape: {url}", "url": url}

    analysis = analyze_offer_chain(content)
    if not analysis:
        return {"error": "LLM parsing failed", "url": url}

    analysis["url"] = url
    analysis["status"] = "success"
    return analysis


def process_batch(urls: list[str]) -> list[dict]:
    """Process multiple offers and return all results."""
    results = []
    for url in urls:
        result = process_offer(url)
        results.append(result)
    return results


def run_gap_analysis(offers: list[dict]) -> str:
    """
    Aggregate skills from all successful offers and run gap analysis.
    """
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
