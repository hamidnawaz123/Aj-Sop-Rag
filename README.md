# Store SOP Assistant (RAG)

Markdown SOP (text + tables) -> structure-aware chunks -> BGE-M3 -> ChromaDB -> Streamlit
(chatbot with citations + SOP viewer popup).

## Run
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows  (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
copy secrets.toml.example .streamlit\secrets.toml   # add your GROQ_API_KEY
streamlit run app.py
```
The first run builds the index from `data/sop.md` (BGE-M3 downloads once, ~2 GB).
Rebuild manually: `python ingest.py --input data/sop.md`, or use **Knowledge base -> Re-index** in the sidebar.

## Deploy on Streamlit Community Cloud
1. Push this project to a GitHub repository. Keep `.env`, `.streamlit/secrets.toml`, and `storage/` out of Git.
2. In Streamlit Community Cloud, choose **Create app**, select the repository and branch, and set the main file to `app.py`.
3. Open the app's **Settings -> Secrets** and add:
	```toml
	GROQ_API_KEY = "gsk_your_new_key_here"
	```
4. Deploy. The first startup downloads the embedding model and builds `storage/` from `data/sop.md`; allow several minutes for this initial build.

The deployed filesystem is temporary. The bundled SOP is re-indexed after a container restart, and files uploaded through **Re-index** are not permanent. For persistent user-managed documents, store the source files and vector database in an external service.

Before pushing, revoke any API key that has been placed in `.env` or `.streamlit/secrets.toml` and create a replacement in Groq.

## Files
- `chunker.py` structure-aware chunking (tune sizes in `config.py`)
- `rag.py` BGE-M3, ChromaDB, retrieval + threshold, Groq generation
- `ingest.py` builds the index and `storage/sops.json` (feeds the popup)
- `app.py` Streamlit UI
- `extract_pdf.py` optional PDF -> markdown with Docling
