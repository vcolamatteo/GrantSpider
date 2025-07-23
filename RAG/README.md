<div align="center">
  <img src="images/logo2.png" alt="GrantSpider Logo" width="400"/>
  
  # 🕷️ GrantSpider
  ### *Conversational Multi-Agent RAG System*
  
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![LangGraph](https://img.shields.io/badge/LangGraph-enabled-purple?style=for-the-badge&logo=langchain&logoColor=white)](https://python.langchain.com/docs/langgraph/)
  [![Gradio](https://img.shields.io/badge/Gradio-powered-orange?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
  [![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
  [![Groq](https://img.shields.io/badge/Groq-API-red?style=for-the-badge)](https://groq.com/)
  
</div>

---

### ✨ Key Features

🤖 **Multi-Agent Architecture** - Designed with LangGraphs

🔍 **Dense+Sparse Embeddings** - HF local models for semantic and textual understanding

🌐 **Dual Interface** - Choose between a Gradio web UI or the command-line interface

⚡ **Free to use** - Powered by Groq API for a free testing

🎨 **Web UI** - Gradio interface

🔧 **Easy Setup** - UV package


---

## 🚀 Quick Start

### 🛠️ Installation

```bash
# Clone the repository
git clone --recurse-submodules git@github.com:vcolamatteo/GrantSpider.git
cd GrantSpider/RAG

# Install UV if you haven't already
pip install uv
```

### 🔧 Environment Setup

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate 

# Install dependencies
uv pip install -r pyproject.toml
```

### 🔑 API Configuration

Create a `.env` file in the root directory and add your API keys:

```env
# Required API Keys
PINECONE_API_KEY=your_pinecone_api_key_here 
GROQ_API_KEY=your_groq_api_key_here
HF_API_KEY=your_hf_token_here

# Optional Configuration
LANGCHAIN_API_KEY=your_langchain_api_key_here 
```

---

## 🎮 Usage

The system offers two interfaces:

### 🖥️ Command Line Interface

```bash
python rag_cmd.py
```

### 🌐 Web Interface

```bash
python rag_gradio.py
```

## ⚙️ First Time Setup

When running GrantSpyder for the first time, enable the initial setup to configure your system:

### 📝 Command Line Setup
In `rag_cmd.py`, set the setup flag:

```python
# Set this to True for first run
run_initial_setup = True
```

![Screenshot from 2025-06-22 13-27-40](https://github.com/user-attachments/assets/d1092164-697b-4674-9e3b-ed4c9c1734c2)

### 🖱️ GUI Setup
Use the convenient setup toggle in the web interface:


https://github.com/user-attachments/assets/c4d8be7a-b14f-442c-91f6-0fd0abd51cfc


---

## 🏗️ RAG System Architecture

The full graph of the RAG

![image](https://github.com/user-attachments/assets/62e049bd-9d77-4682-a46c-564abf6fd22c)


---


## 🔧 Configuration

GrantSpyder uses a comprehensive configuration system that controls all aspects of the RAG pipeline. All parameters are organized in clear categories:

### 📁 Data Sources & Storage

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `source_path` | `"../preprocessing/cropsAll/"` | Folder path containing preprocessing output (pargraphs text) |
| `chunk_dst` | `"chunks_grants/multi_doc/"` | Folder path containing embeddings (.pkl). Used when `run_initial_setup` is True |

### 🗄️ Index Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `dense_index_name` | `'grant-dense-multi-doc'` | Name for the dense vector search index |
| `sparse_index_name` | `'grant-sparse-multi-doc'` | Name for the sparse keyword search index |

### ⚡ Processing Performance

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `processing_batch_size` | `1` | Batch size for document processing |
| `upsert_batch_size` | `10` | Batch size for vector uploads on the DB |

### 🔍 Search & Retrieval

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `sparse_top_k` | `1000` | Sparse dimensionality |
| `dense_search_top_k` | `5` | Dense vector search results (semantic similarity) |
| `sparse_search_top_k` | `5` | Sparse keyword search results (SPLADE matches) |
| `rerank_top_k` | `3` | Final reranked results presented to user |
| `use_reranker` | `False` | Enable/disable the reranking. NOt active by default |

### 🔀 Fusion & Ranking

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `fusion_method` | `'rrf'` | Fusion method: `'rrf'` (Reciprocal Rank) or `'score-based'` |
| `alpha` | `0.7` | Weight for dense vs sparse (0.0 = pure sparse, 1.0 = pure dense) in `'score-based'` |
| `rrf_k` | `60` | RRF constant for rank fusion algorithm |

### 🏗️ Setup & Initialization

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `create_new_embeddings` | `False` | Set to `True` to force recreation of all embeddings |
| `run_initial_setup` | `False` | Set to `True` for first-time setup or index rebuild. It creates the DBs on Pinecone |

### 💬 Chat Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `max_chat_history` | `20` | Maximum conversation turns stored in memory |
| `max_iterations` | `15` | Maximum node to be explorable per query |
| `max_time_seconds` | `60.0` | Timeout for query processing (prevents hanging) |

---
### 🎯 Configuration Tips

**For First-Time Setup:**
- Set `create_new_embeddings: True` if you want to regenerate embeddings, not mandatory
- Set `run_initial_setup: True` to load embeddings from files and creating the Pinecone db

---
