# Customer Service Chatbot 🤖

A multi-feature AI-powered customer service chatbot built with Google Gemini, LangChain, FAISS, and Streamlit. The app supports six distinct features ranging from RAG-based Q&A to multimodal image chat and multilingual support.

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| Base | Q&A Chatbot | Answers questions from a FAQ knowledge base using RAG |
| 1 | Expand Knowledge Base | Dynamically add new URLs or CSV files to the knowledge base |
| 2 | Multimodal Image Chat | Upload an image and ask questions about it using Gemini Vision |
| 3 | Medical Q&A | Answers medical questions using the MedQuAD dataset with entity recognition |
| 4 | arXiv Expert | Searches and explains research papers from the arXiv dataset |
| 5 | Sentiment-aware Reply | Detects user mood and adjusts reply tone accordingly |
| 6 | Multi-language Reply | Auto-detects the user's language and responds in the same language |

---

## Tech Stack

- **LLM:** Google Gemini 2.5 Flash
- **Embeddings:** Google Gemini Embedding (via REST API)
- **Vector Store:** FAISS
- **Framework:** LangChain + Streamlit
- **Sentiment Analysis:** VADER
- **Language Detection:** langdetect

---

## Project Structure

```
customer_service_chatbot_LLM/
├── src/
│   ├── main.py                    # Streamlit app entry point
│   ├── config.py                  # Model names, paths, API key loading
│   ├── langchain_helper.py        # Core RAG logic and embeddings
│   ├── dataset/
│   │   └── dataset.csv            # Base FAQ dataset
│   └── tasks/
│       ├── task1_knowledge_base.py
│       ├── task2_multimodal.py
│       ├── task3_medical_qa.py
│       ├── task4_arxiv_expert.py
│       ├── task5_sentiment.py
│       └── task6_multilanguage.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY="your_google_api_key_here"
```

Get a free key at: https://aistudio.google.com/app/apikey

---

## Running the App

```bash
cd src
streamlit run main.py
```

---

## First-time Setup for Tasks 3 & 4

Tasks 3 and 4 require building FAISS vector databases from external datasets before use.

### Task 3 — Medical Q&A (MedQuAD)

Download MedQuAD from: https://github.com/abachaa/MedQuAD

```bash
cd src
python -c "from tasks.task3_medical_qa import prepare_medquad_csv, build_medical_db; prepare_medquad_csv('PATH/TO/MedQuAD'); build_medical_db()"
```

### Task 4 — arXiv Expert

Download the arXiv dataset from: https://www.kaggle.com/datasets/Cornell-University/arxiv

```bash
cd src
python -c "from tasks.task4_arxiv_expert import build_arxiv_db; build_arxiv_db('PATH/TO/arxiv-metadata-oai-snapshot.json')"
```

> **Note:** The free tier Google API key has rate limits. The build scripts include automatic retry logic with delays. Allow 10-15 minutes for each database to build.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Your Google Gemini API key |

---

## License

MIT
