# Customer Service Chatbot 🤖

A multi-feature AI-powered customer service chatbot built with Google Gemini, LangChain, FAISS, and Streamlit. The app supports six distinct features ranging from RAG-based Q&A to multimodal image chat and multilingual support.

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| Base | Q&A Chatbot | Answers questions from a FAQ knowledge base using RAG |
| 1 | Expand Knowledge Base | Dynamically add new CSV files to the knowledge base (URL adding not recommended on free tier) |
| 2 | Multimodal Image Chat | Upload an image and ask questions about it using Gemini Vision |
| 3 | Medical Q&A | Answers medical questions using the MedQuAD dataset with entity recognition |
| 4 | arXiv Expert | Searches and explains research papers from the arXiv dataset |
| 5 | Sentiment-aware Reply | Detects user mood and adjusts reply tone accordingly |
| 6 | Multi-language Reply | Auto-detects the user's language and responds in the same language |

---

## Tech Stack

- **LLM:** Google Gemini 2.5 Flash
- **Embeddings:** Google Gemini Embedding (via direct REST API calls to v1 endpoint)
- **Vector Store:** FAISS
- **Framework:** LangChain + Streamlit
- **Sentiment Analysis:** VADER
- **Language Detection:** langdetect

---

## Project Structure

```
Customer-Service-Chatbot/
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
git clone https://github.com/HammadAbrarMatto/Customer-Service-Chatbot.git
cd Customer-Service-Chatbot
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

---

## Known Limitations (Free Tier API Key)

This project uses the Google Gemini free tier API key which has strict rate limits. The following limitations apply:

**Embedding rate limit (~5 requests/minute):**
- Building the Task 3 (Medical) database is capped at **50 documents** to complete within rate limits
- Building the Task 4 (arXiv) database is capped at **50 papers** to complete within rate limits
- Each build takes 15-30 minutes due to 15-second delays between embedding calls
- The build scripts include automatic retry logic (waits 60s on 429 errors)
- Larger datasets will consistently hit rate limits and fail even with retries

**Task 1 — URL ingestion not recommended on free tier:**
- Adding a webpage URL (e.g. a Wikipedia article) splits it into hundreds of chunks
- Each chunk requires one embedding API call
- This exhausts the free tier quota very quickly and the process stalls indefinitely
- **Workaround:** Use the CSV upload option in Task 1 instead (small CSV = few API calls = works fine)

**Chat model rate limit:**
- The free tier also limits `generateContent` calls per day
- If you see a 429 error while chatting, wait a few minutes or create a new API key at https://aistudio.google.com/app/apikey

**Solution for all of the above:** A paid Google AI Studio API key ($5 credit) removes all these limits and allows full dataset ingestion in under a minute.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Your Google Gemini API key |

---

## Project Analysis & Visualizations

See [`analysis.ipynb`](analysis.ipynb) for:
- FAQ dataset analysis and plots
- Sentiment analysis results and confusion matrix
- MedQuAD dataset exploration
- RAG pipeline architecture diagram
- Baseline vs advanced model comparison
- Full list of technical challenges and solutions

## License

MIT
