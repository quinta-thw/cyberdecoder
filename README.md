# CyberDecoder

A local cybersecurity chatbot that explains jargon, CVEs, and security terms in plain English. Built with Gradio, LangChain, ChromaDB, and a local Ollama model, so it runs free with no API key.

#codingwithqueen

## Setup

1. Install [Ollama](https://ollama.com) and pull the model: `ollama pull llama3.2:3b`
2. Create and activate a virtual environment, then install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Build the knowledge database from `data/knowledge.txt`: `python ingest.py`
4. Start the chatbot: `python app.py` and open http://localhost:7860

## Adding knowledge

Add `Q:` / `A:` pairs to `data/knowledge.txt`, delete the `chroma_db` folder, and run `python ingest.py` again.
