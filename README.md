#  RAG Profile Search — Orants.AI Internship (Task 12)

##  Project Overview
This project demonstrates a **Retrieval-Augmented Generation (RAG)** based semantic search system for retrieving candidate profiles intelligently.  
Developed as part of the **Orants.AI Internship (Task 12)**, it showcases how AI-powered profile matching can be implemented using embeddings and similarity search.

The app enables users to search for candidates (e.g., “AI Engineer”, “Python Developer”) and fetch the most relevant profiles based on meaning — not just keywords.

---

##  Core Features
- **Semantic Search using Sentence Transformers**  
  Generates embeddings for each profile using the `all-MiniLM-L6-v2` model.  

- **Vector Similarity Search with FAISS**  
  Uses FAISS (Facebook AI Similarity Search) for efficient nearest-neighbor lookups.  

- **Streamlit Web Interface**  
  Simple, interactive UI for real-time profile search.  

- **Manual Implementation**  
  The project was completed manually (without Supabase or external integrations).

---

##  Project Files
| File | Description |
|------|--------------|
| `app.py` | Streamlit application implementing embedding, FAISS indexing, and UI. |
| `profiles.json` | Sample dataset of profiles for search queries. |
| `requirements.txt` | List of Python dependencies required to run the app. |
| `README.md` | Documentation of the project. |

---

## Technology Stack
- **Python 3.10+**
- **Streamlit** – for building the user interface  
- **Sentence Transformers** – to generate embeddings  
- **FAISS** – for vector similarity search  
- **NumPy** – for array and vector operations  
- **JSON** – for structured data storage  

---

##  Notes
- The application runs using a **local JSON dataset** of profiles.  
- It can be extended to connect with databases like **Supabase** or **Odoo Applicants**.  
- Run through : streamlit run app.py

---

##  Outcome
 Successfully implemented a working **RAG-based profile search app**.  
 Gained hands-on experience with **semantic retrieval, embeddings, and FAISS**.  
 Enhanced understanding of **AI-driven data retrieval** using modern NLP techniques.  

---

**Developed by:** Hithesh G — *AI Engineer Intern, Orants.AI*  
