# Citizen Navigator AI Portal

A dual-dashboard Flask + LangGraph application for Indian citizen assistance:
- **⚖️ Legal Resolution Dashboard**: upload a notice/document or describe a case and get official-step guidance.
- **🎓 Opportunity Finder Dashboard**: submit a detailed profile and get personalized schemes, jobs, innovation grants, missed opportunities, and a practical roadmap.

The system uses **hybrid retrieval** for profile intelligence:
- **Live web search** (official domains via Tavily)
- **Offline policy RAG** (FAISS over local policy PDFs)

---

## Features

- Two independent pipelines (no cross-contamination of legal/profile workflows)
- Stage-aware legal analysis (`legal_category`, `case_status`)
- Deep profile filtering (`gender`, `income`, `status`, `skills`, `goal`)
- Structured profile outputs:
  - `opportunities_json`
  - `missed_opportunities_json`
  - `guiding_roadmap`
- Dark-mode, dual-tab frontend with async API calls
- OCR fallback for scanned PDF/image documents

---

## Project Structure

```text
legal/
├── README.md
├── .gitignore
├── legal_agentic_system/
│   ├── app.py
│   ├── main_graph.py
│   ├── state.py
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html
│   ├── agents/
│   │   ├── data_agent.py
│   │   ├── legal_pipeline.py
│   │   ├── profile_pipeline.py
│   │   ├── policy_rag.py
│   │   └── search_tool.py
│   └── ...
└── venv/ (local)
```

---

## Architecture Overview

### 1) Legal Graph
`data_extraction -> legal_analysis -> END`

- `data_extraction`: reads raw text / text file / PDF / image OCR
- `legal_analysis`: summarizes issue, performs official procedure search, returns legal action plan

### 2) Profile Graph
`profile_analysis -> END`

- `profile_analysis`: combines policy RAG + live search, ranks opportunities, flags missed windows, builds roadmap

---

## Requirements

- Python 3.10+
- Poppler installed (required by `pdf2image`)
- API keys for Groq and Tavily

Install dependencies:

```bash
cd legal_agentic_system
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

Create `legal_agentic_system/.env`:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
HF_TOKEN=your_huggingface_token
```

Optional policy-RAG corpus:
- Put policy PDFs in: `legal_agentic_system/policy_docs/`
- FAISS index is auto-created in: `legal_agentic_system/policy_faiss_index/`

---

## Run Locally

```bash
cd legal_agentic_system
python app.py
```

Open: `http://localhost:5000`

---

## API Endpoints

- `POST /api/analyze_legal`
  - Inputs: `raw_text` **or** `file`, plus optional `legal_category`, `case_status`
  - Output: `solution`

- `POST /api/analyze_profile`
  - Inputs: profile fields (`user_age`, `user_state`, `user_education`, `user_specialization`, `user_gender`, `annual_income`, `current_status`, `extracurriculars`, `user_goal`)
  - Output: `opportunities`, `missed_opportunities`, `roadmap`, `report`

---

## Why Files May Not Have Appeared on GitHub

This repository previously had `legal_agentic_system` tracked as a **nested git link** instead of normal files. That setup was converted so the folder contents are now tracked directly by this repo.

After pulling these changes, run:

```bash
git add -A
git commit -m "Fix tracking + add README"
git push
```

---

## Notes

- Large/generated directories are intentionally gitignored by default:
  - `legal_agentic_system/data/`
  - `legal_agentic_system/faiss_index/`
  - `legal_agentic_system/policy_faiss_index/`
- If you want these committed, remove those lines from `.gitignore`.

---

## License

Use according to your organization/course requirements.
