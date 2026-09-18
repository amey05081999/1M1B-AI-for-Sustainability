# 🌿 Eco Lifestyle Agent

An AI-powered assistant that helps users adopt a greener lifestyle through **Retrieval-Augmented Generation (RAG)**. It retrieves practical sustainability guidance from a curated knowledge base and generates actionable, personalized advice using an LLM.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 **RAG Pipeline** | FAISS vector store + sentence-transformers embeddings (local, no API key needed) |
| 🤖 **LLM Support** | OpenAI GPT (3.5/4o) **or** local Ollama models (llama3, mistral, gemma) |
| 💬 **Multi-turn Chat** | Maintains conversation history for follow-up questions |
| 🌱 **Rich Knowledge Base** | 10 curated topic documents covering all major eco-lifestyle areas |
| 🏛️ **Government Schemes** | UK, US, EU grants and incentive programmes included |
| 🖥️ **Web UI** | Clean green-themed chat interface |
| ⚡ **FastAPI Backend** | REST API with automatic Swagger docs at `/docs` |
| 🔌 **Zero-key mode** | Works without any API key using context-only fallback |

---

## 📚 Knowledge Base Topics

1. **Reducing Plastic Use** — single-use plastics, alternatives, UK/US plastic regulations
2. **Home Energy Saving & Renewables** — insulation, solar PV, heat pumps, government grants
3. **Sustainable Travel & Transport** — carbon by mode, EVs, trains, eco holidays
4. **Recycling, Waste & Composting** — what to recycle, composting, food waste reduction
5. **Sustainable Diet & Food** — carbon footprint of foods, plant-based eating, seasonal buying
6. **Eco-Friendly Products & Shopping** — certifications, eco-brands, sustainable fashion
7. **Water Conservation** — indoor/outdoor tips, virtual water, government water schemes
8. **Government Schemes & Grants** — UK, US, EU incentives and environmental programmes
9. **Biodiversity & Wildlife Gardening** — wildlife-friendly gardens, citizen science
10. **Personal Carbon Footprint** — what it is, how to measure, high-impact actions

---

## 🚀 Quick Start

### 1. Clone and set up the environment

```bash
git clone <repo-url>
cd Eco-Lifestyleagent

# Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key (optional — works without it)
```

### 3. Run the server

```bash
python main.py
```

Open your browser at **http://localhost:8000** to use the chat interface.

API documentation: **http://localhost:8000/docs**

---

## 🛠️ Configuration

All settings are in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI API key (optional) |
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_CHAT_MODEL` | `gpt-3.5-turbo` | Any OpenAI chat model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `TOP_K_RETRIEVAL` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `60` | Overlapping words between chunks |
| `VECTOR_STORE_PATH` | `data/vectorstore` | Where the index is saved |

---

## 🦙 Using Local Ollama (no API key required)

1. Install [Ollama](https://ollama.com) and pull a model:
   ```bash
   ollama pull llama3
   ```
2. Set in `.env`:
   ```
   LLM_PROVIDER=ollama
   OLLAMA_MODEL=llama3
   ```
3. Start the server: `python main.py`

---

## 🔌 Zero-key / Offline Mode

If neither OpenAI key nor Ollama is configured, the agent runs in **context-only mode** — it retrieves and displays the most relevant knowledge base passages directly, without LLM generation. Useful for development and testing.

---

## 🧪 CLI Test

Test the agent from the command line without a browser:

```bash
# Run all built-in test questions
python test_agent.py

# Ask your own question
python test_agent.py "How can I make my commute more eco-friendly?"
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Chat UI (browser)                      │
│                 frontend/index.html + app.js                │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP (REST JSON)
┌───────────────────────▼─────────────────────────────────────┐
│               FastAPI Backend  (main.py)                    │
│  POST /api/chat   POST /api/reset   GET /api/health         │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│               EcoAgent  (agent.py)                          │
│  - RAG retrieval → context building → prompt construction   │
│  - Conversation history management                          │
└──────────┬────────────────────────────┬──────────────────────┘
           │                            │
┌──────────▼──────────┐      ┌──────────▼────────────────────┐
│   RAG Pipeline      │      │      LLM Backend              │
│   (rag_engine.py)   │      │  OpenAI / Ollama / Fallback   │
│                     │      └───────────────────────────────┘
│  ┌───────────────┐  │
│  │  FAISS /      │  │
│  │  NumPy Index  │  │
│  └───────┬───────┘  │
│          │           │
│  ┌───────▼───────┐  │
│  │  Sentence     │  │
│  │  Transformers │  │
│  │  Embeddings   │  │
│  └───────────────┘  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Knowledge Base     │
│  data/kb/*.txt      │
│  (10 topic files)   │
└─────────────────────┘
```

---

## 📁 Project Structure

```
Eco-Lifestyleagent/
├── main.py               # FastAPI server entry point
├── agent.py              # EcoAgent — LLM + RAG orchestration
├── rag_engine.py         # RAG pipeline (load, embed, index, retrieve)
├── config.py             # Central configuration
├── test_agent.py         # CLI test script
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── data/
│   ├── kb/               # Knowledge base documents (10 .txt files)
│   └── vectorstore/      # Auto-generated FAISS index (created on first run)
└── frontend/
    ├── index.html        # Chat web interface
    └── static/
        ├── style.css     # Green-themed styling
        └── app.js        # Chat UI logic
```

---

## 🌍 Example Questions

- *"How can I reduce plastic use at home?"*
- *"What are the most eco-friendly ways to travel between cities?"*
- *"What UK government grants are available for heat pumps?"*
- *"How do I start composting and what can I put in it?"*
- *"What is a carbon footprint and what are the highest-impact actions?"*
- *"How can I make my garden more wildlife-friendly?"*
- *"Which foods have the highest carbon footprint?"*
- *"How much water do I waste and how can I reduce it?"*
- *"What eco certifications should I look for when shopping?"*

---

## 📄 License

MIT License — free to use, modify, and distribute.
