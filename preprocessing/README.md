<div align="center">
  <img src="../RAG/images/logo2.png" alt="GrantSpider Logo" width="400"/>
  
  # 🕷️ GrantSpider
  ### *Document Layout Analysis for RAG*
  
  
  [![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
  [![Groq](https://img.shields.io/badge/Groq-API-red?style=for-the-badge)](https://groq.com/)
  
</div>

---

### ✨ Key Features

🤖 **Object detection** - tile, text, table, abandon, image

🔍 **Paragraph and sub-paragraph extraction** - In case of need, retrieved chunks are replaced by the entire relative paragraphs

🌐 **OCR** - Text is extracted trough OCR

⚡ **Free to use** - Table parsing for a correct text formatting is powered by a Groq API multimodal model 

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git
- UV package manager (recommended)

### 🛠️ Installation

```bash
# Clone the repository
git clone --recurse-submodules git@github.com:vcolamatteo/GrantSpider.git
cd GrantSpider/preprocessing

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
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🖥️ Usage

```bash
python run_doc_layout_analysis.py
```
---
