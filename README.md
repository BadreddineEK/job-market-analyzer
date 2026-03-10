# 🤖 Job Market Analyzer

An intelligent LLM-powered agent that analyzes job offers, detects skill gaps, and generates personalized career insights — powered by **LangChain**, **Groq (Llama 3.3 70B)** and **Streamlit**.

## ✨ Features

- 🔍 **Scrape job offers** from any URL (LinkedIn, WTTJ, Indeed…)
- 📊 **Batch analysis** of multiple offers at once
- 🧠 **Skill gap detection** — compare offers vs your profile
- 📝 **Smart summaries** per offer (missions, stack, salary, level)
- 📈 **Trend report** — most requested skills across all offers
- 💡 **Personalized recommendations** on what to learn next
- 💾 **Export** results as CSV or JSON

## 🛠 Tech Stack

| Tool | Role |
|---|---|
| LangChain | Agent orchestration & chains |
| Groq API (Llama 3.3 70B) | Free, ultra-fast LLM |
| Streamlit | Web UI |
| BeautifulSoup4 | Web scraping |
| ChromaDB | Optional vector store for RAG |
| Pydantic | Data validation |

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/BadreddineEK/job-market-analyzer.git
cd job-market-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file at the root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get your free key at [console.groq.com](https://console.groq.com)

### 4. Configure your profile
Edit `config/profile.yaml` with your skills, experience, and job target.

### 5. Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure

```
job-market-analyzer/
├── app.py                  # Main Streamlit app
├── config/
│   └── profile.yaml        # Your professional profile
├── src/
│   ├── scraper.py          # Job offer scraper
│   ├── agent.py            # LangChain agent
│   ├── chains.py           # LangChain chains
│   └── utils.py            # Helpers
├── prompts/
│   ├── analyze_offer.txt   # Prompt for single offer analysis
│   └── gap_analysis.txt    # Prompt for skill gap
├── data/
│   └── .gitkeep
├── requirements.txt
├── .env.example
└── README.md
```

## 📸 Demo

> Coming soon

## 📄 License

MIT
