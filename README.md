
## Demo

https://github.com/user-attachments/assets/6249c5a0-5024-4ac7-bdb8-ef515cee1f1f


## Screenshots

![Image](https://github.com/user-attachments/assets/ee94afc0-0def-41a8-a501-602ec3a901ad)
# Smolagents-systems
🔍 Multi-Agent Documentation Scraper with Semantic Search 

🚀 Automated documentation scraping using collaborative AI agents (web search + extraction)

📦 Powered by ChromaDB, DuckDuckGo, and Gradio 🤖 Features vector embeddings, tool-calling agents, and semantic search



# Multi-Agent Documentation Scraper

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered system for automated documentation scraping and semantic search using collaborative agents.

![Demo Screenshot](https://via.placeholder.com/800x400.png?text=Gradio+Interface+Demo)

## Features
- 🤖 Two specialized agents: Web Researcher + Documentation Extractor
- 🔍 Semantic search using ChromaDB vector database
- 🌐 DuckDuckGo integration for web searches
- 🎮 Gradio web interface for easy interaction

## Quick Start

### Google Colab
1. Open [this Colab notebook](https://colab.research.google.com/)
2. Run these commands:
```bash
!pip install chromadb gradio duckduckgo-search transformers torch
!git clone https://github.com/vaibhavgitt/Smolagents-systems
%cd Smolagents-systems
```

bash
```
python -m venv venv
source venv/bin/activate  # Linux/MacOS
.\venv\Scripts\activate   # Windows
Install dependencies:
```
bash
```
pip install -r requirements.txt
Run the application:
```
bash
```
python main.py
Requirements
Python 3.8+

requirements.txt:

chromadb>=0.4.0
gradio>=3.50.0
duckduckgo-search>=3.8.6
transformers>=4.30.0
torch>=2.0.0
Usage Example
Start the Gradio interface
```
In "Documentation Scraper" tab: Example

Library Name: PyTorch

Search Query: tensor operations

Click "Scrape Documentation"

Switch to "Documentation Q&A" tab to search stored docs

Example output:

```
### Scraped Documentation
PyTorch Tensor Operations Guide...
**Sources**: [pytorch.org/docs/stable/tensors.html]
How It Works
mermaid

graph TD
    A[User Query] --> B(Web Search Agent)
    B --> C{Found Resources?}
    C -->|Yes| D(Doc Extraction Agent)
    D --> E[ChromaDB Storage]
    E --> F[Semantic Search]
    C -->|No| G[Error Handling]
Configuration
Add to .env file:

```
```
HF_API_TOKEN=your_huggingface_token  # Optional
CHROMA_DB_PATH=./chroma_db
Troubleshooting
Port conflicts: Try python main.py --port 7861
```
```
Missing dependencies: Run pip install -r requirements.txt --force-reinstall

Chromadb errors: Delete chroma_db folder and restart

```
