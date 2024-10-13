import fitz
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from concurrent.futures import ThreadPoolExecutor

pdf_path = "dogovor.pdf"


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text("text")
    return text


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def vectorize_chunks(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name).to("cuda")
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings.cpu().detach().to(torch.float32).numpy()


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_index(query, index, embedding_model, chunks, top_k=30):
    query_vector = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vector, top_k)
    return [(chunks[i], distances[0][idx]) for idx, i in enumerate(indices[0])]


def generate_answer_with_vikhr(retrieved_chunks, query):
    model = AutoModelForCausalLM.from_pretrained(
        "Vikhrmodels/Vikhr-7B-instruct_0.4",
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("Vikhrmodels/Vikhr-7B-instruct_0.4")

    combined_chunks = " ".join(
        [chunk[0] for chunk in retrieved_chunks[:3]])  # Используем только 3 наиболее релевантных чанка

    prompt = f"Context: {combined_chunks}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

    output = model.generate(
        **inputs,
        max_length=4064,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.3,
    )

    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_answer

def initialize_llm(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=500)
    embeddings = vectorize_chunks(chunks)
    index = build_faiss_index(embeddings)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model, index, chunks

def get_answer_from_llm(query):
    # Инициализируем модель и индекс на основе PDF
    model, index, chunks = initialize_llm(pdf_path)  # Чтение PDF внутри LLM
    retrieved_chunks = search_index(query, index, model, chunks)
    answer = generate_answer_with_vikhr(retrieved_chunks, query)
    return answer