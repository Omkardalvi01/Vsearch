import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
import pickle
from fastapi import FastAPI, UploadFile, File, Query
import uvicorn
import tempfile

app = FastAPI()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize the embeddings
embedding_model = HuggingFaceEmbeddings(model_name=model_name)

def similarity_cosine(a, b):
    return np.dot(a , b) / np.linalg.norm(a) * np.linalg.norm(b)

@app.post('/upload')
def upload_file(file : UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 200, chunk_overlap = 50)
    splits = splitter.split_documents(docs) 
    texts = [split.page_content for split in splits]
    embed_docs = embedding_model.embed_documents(texts)
    data = [{'text' : text , 'embedding' : embeds} for text, embeds in zip(texts,embed_docs)]
    print(len(data[0]['embedding']))
    with open("save.pkl", 'rb') as f:
        saved_data = pickle.load(f)
    for dic in data:
        saved_data.append(dic)
    with open("save.pkl","wb") as f:
        pickle.dump(saved_data, f)
    return {"result"  : "success"}

@app.get('/retrieve')
def retrieve(query : str = Query(..., description="search query")):
    vector = embedding_model.embed_query(query)
    results = []
    with open("save.pkl", 'rb') as f:
        data = pickle.load(f)
    for entry in data:
        results.append(similarity_cosine(entry['embedding'], vector))
    results = np.array(results)
    ind = np.argpartition(results, -5)[:-5]
    result = []
    for i in ind:
        result.append(data[i]['text'])
    return {"result" : result}

if __name__ == "__main__":
    uvicorn.run(app, port=5000)