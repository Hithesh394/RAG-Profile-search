import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# ------------------------------
# 🎨 PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="RAG Profile Search", layout="centered", page_icon="🔍")

# ------------------------------
# 🏠 HEADER SECTION
# ------------------------------
st.markdown("""
<h1 style='text-align: center; color: #2b4162;'>🔍 RAG Profile Search</h1>
<p style='text-align: center; font-size: 18px; color: #4b4b4b;'>
Find the most relevant tech professionals using <b>AI-powered semantic search</b>.<br>
Powered by <b>Sentence Transformers</b> + <b>FAISS</b>.
</p>
<hr style="border: 1px solid #ddd;">
""", unsafe_allow_html=True)

# ------------------------------
# 📂 LOAD DATASET
# ------------------------------
try:
    with open("profiles.json", "r", encoding="utf-8") as f:
        profiles = json.load(f)
except FileNotFoundError:
    st.error("❌ profiles.json file not found. Please make sure it's in the same folder as app.py.")
    st.stop()

# ------------------------------
# 🤖 LOAD EMBEDDING MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------------------
# 🧮 CREATE VECTOR INDEX
# ------------------------------
profile_texts = [p["bio"] for p in profiles]
embeddings = model.encode(profile_texts, show_progress_bar=False)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# ------------------------------
# 🔍 SEARCH SECTION
# ------------------------------


query = st.text_input("🔎 Enter your search query here:")

if query:
    query_vector = model.encode([query]).astype("float32")
    top_k = 3
    distances, indices = index.search(query_vector, top_k)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Top Matching Profiles")

    for i, idx in enumerate(indices[0]):
        profile = profiles[idx]
        st.markdown(f"""
        <div style='background-color:#f8f9fa; padding:15px 20px; border-radius:12px; margin-bottom:15px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);'>
            <h4 style='margin-bottom:5px; color:#2b4162;'>{i+1}. {profile['name']}</h4>
            <p style='margin:0; color:#0073e6; font-weight:500;'>{profile['role']}</p>
            <p style='margin:5px 0; color:#333;'>{profile['bio']}</p>
            <p style='font-size:14px; color:#777;'><b>Skills:</b> {", ".join(profile['skills'])}</p>
            <p style='font-size:14px; color:#777;'><b>Location:</b> {profile['location']} 
            <p style='font-size:14px; color:#777;'><b>Experience:</b> {profile['experience_years']} years</p>
        </div>
        """, unsafe_allow_html=True)

    st.success("✅ Results generated using semantic search (RAG).")

else:
    st.info("💡 Type a keyword above to search profiles intelligently...")

# ------------------------------
# 🧾 FOOTER SECTION
# ------------------------------
st.markdown("""
<hr style="border: 1px solid #eee;">
<p style='text-align:center; color:#999; font-size:14px;'>
Built with ❤️ using <b>Streamlit</b> • <b>Sentence Transformers</b> • <b>FAISS</b><br>
© 2025 RAG Search Demo
</p>
""", unsafe_allow_html=True)
