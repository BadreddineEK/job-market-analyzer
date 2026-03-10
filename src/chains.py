import json
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def get_llm(model: str = None, temperature: float = 0.1):
    """Initialize Groq LLM (free tier)."""
    return ChatGroq(
        model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def analyze_offer_chain(job_content: str) -> Optional[dict]:
    """
    Analyze a single job offer and return structured JSON.
    """
    prompt_text = (PROMPTS_DIR / "analyze_offer.txt").read_text(encoding="utf-8")
    prompt = PromptTemplate.from_template(prompt_text)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    raw = chain.invoke({"job_content": job_content})

    try:
        # Clean potential markdown code blocks
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def gap_analysis_chain(profile: dict, aggregated_skills: list[str]) -> str:
    """
    Run a skill gap analysis comparing the user profile to market demand.
    """
    prompt_text = (PROMPTS_DIR / "gap_analysis.txt").read_text(encoding="utf-8")
    prompt = PromptTemplate.from_template(prompt_text)
    llm = get_llm(temperature=0.3)
    chain = prompt | llm | StrOutputParser()

    profile_str = json.dumps(profile, ensure_ascii=False, indent=2)
    skills_str = ", ".join(set(aggregated_skills))

    return chain.invoke({"profile": profile_str, "aggregated_skills": skills_str})
