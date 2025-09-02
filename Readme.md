<h1 align="center">🚀 notebooklm-plus</h1>

<p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://github.com/f-gg/notebooklm-plus#readme" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/docs-available-brightgreen.svg" />
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank">
    <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" />
  </a>
</p>

> **notebooklm-plus** is an advanced **multi-source Retrieval-Augmented Generation (RAG) system** built with [LangChain](https://www.langchain.com/), [Google Gemini](https://ai.google/), and [Chroma DB](https://www.trychroma.com/).  
It supports ingesting and querying data from **GitHub repositories, PDFs, YouTube videos (yt-dlp + Whisper), websites, DOCX, CSV, Markdown, and TXT files**.  

The system includes **token-aware chunking**, **vector-based retrieval**, and an **interactive CLI** for Q&A across multiple knowledge sources.

---

## 📦 Features

- ✅ Multi-source ingestion: GitHub, PDF, YouTube, Website, DOCX, CSV, Markdown, TXT  
- ✅ Smart chunking with overlap & token estimation  
- ✅ Vector search powered by **Chroma DB**  
- ✅ **Google Gemini 1.5 Flash** + `text-embedding-004` embeddings  
- ✅ Fallback to Whisper transcription for YouTube  
- ✅ Interactive CLI with chat history and confidence scoring  
- ✅ Apache 2.0 licensed, easy to extend  

---

## 🏠 Homepage

[https://github.com/f-gg/notebooklm-plus](https://github.com/f-gg/notebooklm-plus)

---

## 🚀 Clone & Install

```bash
# Clone the repository
git clone https://github.com/f-gg/notebooklm-plus.git
cd notebooklm-plus

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
