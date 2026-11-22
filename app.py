import os
import re
import pickle
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# --- Page Config ---
st.set_page_config(page_title="MedQuAD Medical Assistant", page_icon="🩺", layout="centered")
st.title("🩺 MedQuAD Medical RAG Assistant")
st.caption("Powered by Hybrid Search (BM25 + BGE), Cross-Encoder Reranking & Groq")

# --- Constants ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "qwen/qwen3.8-27b"
CANDIDATES = 20
TOP_N = 5

STOP = set(
    "a an the of and or to in on for with is are was were be been what who how why when which do does did "
    "can could should would may might i you he she it we they my your his her its our their this that these "
    "those about as at by from into than then so such not no if any some there have has had".split()
)

def stem(t):
    for suf in ("ments", "ment", "ations", "ation", "ing", "ies", "ied", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return t

def tokenize(text):
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOP and (len(t) > 1 or t.isdigit())]

def format_docs(docs):
    return "\n\n".join(
        f"[{n}] ({d.metadata['source']} | {d.metadata['question']})\n{d.metadata['raw']}"
        for n, d in enumerate(docs, 1)
    )

# --- Load Models & Indexes (Cached so it runs only once) ---
@st.cache_resource(show_spinner="Loading medical knowledge base & AI models...")
def load_rag_pipeline():
    # 1. Load Embeddings & FAISS (Force CPU)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": CANDIDATES})

    # 2. Load Docs & BM25
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(docs, k=CANDIDATES, preprocess_func=tokenize)

    # 3. Hybrid + Cross-Encoder Reranker
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6])
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL, model_kwargs={"device": "cpu"})
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=TOP_N)
    rerank_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid_retriever)

    return rerank_retriever

retriever = load_rag_pipeline()

# --- Initialize LLM & Chains ---
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
if not groq_api_key:
    st.error("Missing `GROQ_API_KEY`. Please add it in Streamlit App Settings -> Secrets.")
    st.stop()

llm = ChatGroq(model=LLM_MODEL, api_key=groq_api_key, temperature=0)

SYSTEM_PROMPT = (
    "You are a medical information assistant. Answer the user's question using ONLY the numbered "
    "context passages (from NIH sources).\n"
    "- If the passages do not contain the answer, say you could not find it in the knowledge base. Do not guess.\n"
    "- Cite passages inline like [1], [2].\n"
    "- Be clear and concise, in plain language.\n"
    "- Never diagnose or prescribe. For urgent or personal medical concerns, tell the user to see a doctor.\n"
    "- If the user describes an emergency (chest pain, stroke signs, overdose, suicidal thoughts), "
    "tell them to contact local emergency services first."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

condense_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the follow-up as a standalone medical question using the chat history. Return ONLY the question."),
    ("human", "History:\n{history}\n\nFollow-up: {question}"),
])

rag_chain = (
    RunnableParallel(docs=retriever, question=RunnablePassthrough())
    | RunnablePassthrough.assign(
        answer=(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
        | prompt | llm | StrOutputParser()
    )
)
condense_chain = condense_prompt | llm | StrOutputParser()

# --- Chat UI & Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.header("Chat Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.info("**Disclaimer:** This is general information from NIH sources, not medical advice. Please consult a qualified healthcare professional.")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("standalone"):
            st.caption(f"🔍 *Searched for: {msg['standalone']}*")
        if msg.get("sources"):
            with st.expander("View NIH Sources"):
                for n, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{n}] {src['source']}** — *{src['question']}*\n\n> {src['raw']}")

# Handle new user input
if user_input := st.chat_input("Ask a medical question (e.g., What are the symptoms of glaucoma?)"):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching NIH knowledge base..."):
            standalone = user_input
            # Build history from last 3 turns
            past_turns = [
                (st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"])
                for i in range(0, len(st.session_state.messages) - 1, 2)
            ]
            if past_turns:
                hist_str = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in past_turns[-3:])
                standalone = condense_chain.invoke({"history": hist_str, "question": user_input}).strip()

            result = rag_chain.invoke(standalone)
            answer_text = result["answer"]
            retrieved_docs = result["docs"]

        st.markdown(answer_text)
        if standalone != user_input:
            st.caption(f"🔍 *Searched for: {standalone}*")

        sources_data = [
            {"source": d.metadata["source"], "question": d.metadata["question"], "raw": d.metadata["raw"]}
            for d in retrieved_docs
        ]
        with st.expander("View NIH Sources"):
            for n, src in enumerate(sources_data, 1):
                st.markdown(f"**[{n}] {src['source']}** — *{src['question']}*\n\n> {src['raw']}")

    # Save assistant turn
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "standalone": standalone if standalone != user_input else None,
        "sources": sources_data
    })