# 🩺 MedQuAD Medical RAG Assistant (Hybrid Search + Reranking)

A Retrieval-Augmented Generation (RAG) medical information assistant built on the **NIH MedQuAD** knowledge base using **LangChain**, **FAISS**, **BM25**, **Cross-Encoder Reranking**, and **Groq**. This application answers health-related queries using verified NIH passages and provides:

- 🟢 Grounded Answers with Inline Citations (`[1]`, `[2]`)
- 🔴 Strict Anti-Hallucination & Emergency Safety Guardrails

The pipeline combines **hybrid dense-sparse retrieval**, **cross-encoder reranking**, and **multi-turn conversational query rewriting** to deliver accurate, source-backed responses.

---

# 📌 Features

- Hybrid Retrieval (Sparse BM25 + Dense `BAAI/bge-small-en-v1.5`)
- Cross-Encoder Reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Multi-Turn Conversational Memory & Follow-Up Question Condensing
- Inline Source Citations with Expandable NIH Passage Viewer
- Pre-built Local FAISS & BM25 Indexes for Fast Startup
- Strict Grounding & Emergency Triage System Prompts
- Interactive Streamlit Web Chat Interface
- Google Colab & Streamlit Community Cloud Compatible

---

# 📂 Knowledge Base Structure

```
MedQuAD Dataset (NIH Sources)/
│
├── GHR (Genetics Home Reference)
├── GARD (Genetic and Rare Diseases)
├── NIDDK (Diabetes & Digestive/Kidney Diseases)
├── NINDS (Neurological Disorders & Stroke)
├── NHLBI (Heart, Lung, and Blood Institute)
└── Other NIH Medical Institutes
```

---

# 📊 Dataset & Index Information

- **Source:** MedQuAD (Medical Question Answering Dataset — NIH)
- **Total Indexed Passages:** 33,671 cleaned Q&A documents
- **Chunking Strategy:**
  - Chunk Size: 800 characters
  - Chunk Overlap: 120 characters

---

# 🏗️ RAG Pipeline Architecture

- **Sparse Retriever:** Custom stemmed & stop-word-filtered BM25 (`k=20`)
- **Dense Retriever:** Normalized `BAAI/bge-small-en-v1.5` embeddings in FAISS (`k=20`)
- **Ensemble Fusion:** Weighted Hybrid Search (`0.4` BM25 + `0.6` Dense)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (`top_n=5`)
- **Generator:** Groq Cloud LLM (`qwen/qwen3.8-27b`, `temperature=0`)

```
User Query + Chat History
           │
   Standalone Condenser
           │
   ┌───────┴───────┐
   ▼               ▼
BM25 (0.4)    FAISS BGE (0.6)
   └───────┬───────┘
           ▼
 Cross-Encoder Reranker (Top 5)
           │
   Groq LLM + Citations
```

---

# ⚙️ System Configuration

| Parameter | Value |
|-----------|---------|
| Framework | LangChain + Streamlit |
| Embedding Model | `BAAI/bge-small-en-v1.5` |
| Vector Store | FAISS (CPU) |
| Lexical Retriever | BM25 (Custom Stemmer) |
| Hybrid Weights | 0.4 BM25 / 0.6 Dense |
| Reranker Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Candidate Pool (`k`) | 20 |
| Final Context (`top_n`) | 5 |
| LLM Provider & Model | Groq (`qwen/qwen3.8-27b`) |
| Temperature | 0 |

---

# 📈 Retrieval & Safety Guardrails

The system enforces reliability and clinical safety through:

- **Context-Only Answering:** Refuses to guess if the answer is absent from retrieved NIH passages
- **Inline Attribution:** Every claim is tied to numbered context passages (`[1]`, `[2]`)
- **Emergency Detection:** Directs users describing acute emergencies (chest pain, stroke signs, overdose) to local emergency services
- **Non-Diagnostic Stance:** Explicitly avoids diagnosing or prescribing medication

---

# 💬 Chat Interface & Sources

The Streamlit web application provides:

- Persistent multi-turn chat session history
- Automatic standalone search query display for follow-up questions
- Expandable **View NIH Sources** drawer showing original source institute, matched question, and raw passage text
- One-click chat history reset

---

# 💾 Saved Indexes

The pre-computed vector index and serialized document store are saved locally for instant deployment:

```
faiss_index/
├── index.faiss
└── index.pkl

docs.pkl
```

---

# 🔍 Example Interaction

The app allows users to ask medical questions and follow-ups naturally:

```
User: What are the symptoms of glaucoma?

Assistant:
In its early stages, open-angle glaucoma often has no symptoms and causes no pain [1]. 
As the disease progresses, a person may gradually lose peripheral (side) vision, 
which can eventually lead to tunnel vision or blindness if untreated [1][2].

🔍 Searched for: What are the symptoms of glaucoma?
▶ View NIH Sources
```

---

# 📁 Project Structure

```
medquad-rag/
│
├── app.py
├── requirements.txt
├── docs.pkl
├── README.md
│
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/CareerCore/medquad-rag.git
cd medquad-rag
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Groq API key as an environment variable (or in `.streamlit/secrets.toml`):

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

---

# ▶️ Run

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Open the local URL shown in your terminal (typically `http://localhost:8501`).

---

# 📦 Required Libraries

```
torch
streamlit
langchain
langchain-classic
langchain-community
langchain-huggingface
langchain-groq
faiss-cpu
rank_bm25
sentence-transformers
```

Install manually:

```bash
pip install torch streamlit langchain langchain-classic langchain-community langchain-huggingface langchain-groq faiss-cpu rank_bm25 sentence-transformers
```

---

# 🎯 Results & Capabilities

The RAG pipeline delivers:

- High-precision medical passage retrieval via two-stage hybrid + cross-encoder ranking
- Fast sub-second LLM inference powered by Groq
- Accurate multi-turn follow-up resolution
- Transparent, verifiable NIH citations for every response

---

# 📚 Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Transformers (`BGE` & `MS-MARCO Cross-Encoder`)
- Groq API (`Qwen 3.8 27B`)
- Rank-BM25
- PyTorch

---

# 📜 License

This project is developed for educational and research purposes. Information provided by the assistant is sourced from NIH MedQuAD data and does not constitute professional medical advice.

---

# 👨‍💻 Author

**Muhammad Shoaib**

Information Technology Student — Punjab University

AI / Machine Learning | RAG Systems | Full-Stack Development

---

⭐ If you found this project useful, consider giving it a star on GitHub!
