
# Import Statements
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import FastAPI,Body
from pydantic import BaseModel
import uvicorn

# SBERT Sentence Encoder model initialization
model_ST = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# FastAPI Setup
app = FastAPI()

# Pydantic Model: 
class Texts(BaseModel):
    text1: str
    text2: str

# Text Embedding Function: 
'''The embedded_vectors_text function takes a text input,
encodes it into a numerical vector using the Sentence Transformers model, and normalizes the vector. 
The normalization is done to ensure that the vectors have a unit length, which is required for cosine similarity calculation.'''
def embedded_vectors_text(text):
    original_vec = model_ST.encode([text])[0]
    original_vec_norm = original_vec/np.linalg.norm(original_vec)
    return original_vec_norm

# Similarity Score Calculation: 
'''The get_similarity_score function calculates the cosine similarity between two text inputs 
by first obtaining the embedded vectors using the embedded_vectors_text function and then applying the dot product.'''
def get_similarity_score(text1,text2):
    vector1 = embedded_vectors_text(text1)
    vector2 = embedded_vectors_text(text2)
    cosine_similarity = np.dot(vector1, vector2)
    if cosine_similarity < 0:
        cosine_similarity = 0
    return cosine_similarity
    
# Home Endpoint
@app.get('/')
def home():
    return {"APP IS RUNNING!!!"}

# Compare Texts Endpoint
@app.post("/compare",
    summary = 'Compares text1 and text2 sementically and generates a similarity score.')
async def compare_texts(texts: Texts = Body(...)):
    text1=texts.text1
    text2=texts.text2
    similarity_score = str(get_similarity_score(text1,text2))
    return {"similarity_score": similarity_score}

# Running the Application
if __name__ == '__main__':
    uvicorn.run("main:app",reload=True)
