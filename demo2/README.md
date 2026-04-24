# Medical RAG Chatbot
Multi-PDF RAG chatbot with **Pydantic structured output** and **validation**.
---
demo2/
│
├── schemas.py        ← Pydantic v2 output models + validators  (READ FIRST)
├── ingest.py         ← One-time: PDF → chunks → Pinecone
├── rag_chain.py      ← LCEL RAG chain with structured Groq output
├── app.py            ← Streamlit chat UI
├── requirements.txt
├── .env
├── pdfs/             ← DROP YOUR PDFs HERE
│   ├── diabetes.pdf
│   ├── renal_failure.pdf
│   └── A-new-threat-to-obesity--vanity-sizing.pdf
└── README.md
```
---

### Run order
```bash
pip install -r requirements.txt
python ingest.py          # ← run once
streamlit run app.py      # ← run every time
```