import os
from flask import Blueprint, jsonify, request
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import torch
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from openai import OpenAI
from ..utils.rag.graphrag import (
    build_retrieval_pipeline,
    build_vector_store,
    process_uploaded_files,
    retrieve_from_graph,
    split_documents,
)
from ..utils.rag.load_and_search_from_datasets import load_dataset_embedding, search_from_index

device = "cuda" if torch.cuda.is_available() else "cpu"
CROSS_ENCODER_MODEL = "../models/ms-marco-MiniLM-L6-v2"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
qianfan_api_key = os.getenv("QIANFAN_API_KEY")
QIANFAN_MODEL_MAPPING = {
    "ernie-x1":"ernie-x1-32k-preview",
    "ernie-4.5":"ernie-4.5-8k-preview",
    "deepseek-r1":"deepseek-r1",
    "deepseek-v3": "deepseek-v3-241226"
}
rag_bp = Blueprint('api', __name__)
    
INDEX_FILE = "../data/faiss_index"

retrieval_pipeline = None
_INDEX_CACHE = None
_LINES_CACHE = None


@rag_bp.route("/upload_eval", methods=["POST"])
def upload_files():
    global retrieval_pipeline
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    try:
        documents = process_uploaded_files(files)
        texts = split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vector_store = build_vector_store(texts, embeddings)
        retrieval_pipeline = build_retrieval_pipeline(texts, vector_store)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    kg = retrieval_pipeline["knowledge_graph"]
    return jsonify(
        {
            "message": "Files processed and embeddings generated successfully!",
            "total_nodes": len(kg.nodes),
            "total_edges": len(kg.edges),
        }
    )


@rag_bp.route("/chat", methods=["POST"])
def chat():
    global retrieval_pipeline
    global _INDEX_CACHE, _LINES_CACHE
    data = request.get_json()
    prompt = data.get("prompt", "")
    chat_history = data.get("chat_history", "")
    mode = data.get("mode", "custom")
    graphrag = data.get("graphrag", False)
    selected_api = data.get("selected_api", "openai")
    selected_model = data.get("selected_model", "gpt-3.5-turbo")

    if mode == "custom":

        if retrieval_pipeline is not None:
            # BM25检索
            ensemble = retrieval_pipeline["ensemble"]
            retrieved_docs = ensemble.invoke(prompt)
            if graphrag:
                # 图检索
                graph_results = retrieve_from_graph(
                    prompt, retrieval_pipeline["knowledge_graph"]
                )
                graph_docs = (
                    [Document(page_content=node) for node in graph_results]
                    if graph_results
                    else []
                )
                docs = graph_docs + retrieved_docs if graph_docs else retrieved_docs

                retrieved_context = "\n".join([doc.page_content for doc in docs])
                context = context + "\n" + retrieved_context
            else:
                context = "\n".join(
                    f"[Source {i+1}]: {doc.page_content}"
                    for i, doc in enumerate(retrieved_docs)
                )
        else:
            context = " "
    else:
        docs = search_from_index(prompt, k=5)
        context = "\n".join(
            f"[Source {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)
        )
    print(context)
    system_prompt = f"""Use the chat history to maintain context:
Chat History:
{chat_history}

Analyze the question and context through these steps:
1. If it's a casual conversation, skip the following analysis steps and engage in a regular Q&A style.
2. Identify key entities and relationships
3. Check for contradictions between sources
4. Synthesize information from multiple contexts
5. Formulate a structured response

Note: Do not include your analysis process in the response.

Context:
{context}

Question: {prompt}
Answer:"""

    def generate_openai():
        try:
            chat = ChatOpenAI(
                model_name=selected_model, openai_api_key=openai_api_key, streaming=True
            )
            response_generator = chat.invoke([HumanMessage(content=system_prompt)])

            return response_generator.content
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_qianfan():
        try:
            client = OpenAI(
                api_key=qianfan_api_key,
                base_url='https://qianfan.baidubce.com/v2',
            )
            model = QIANFAN_MODEL_MAPPING.get(selected_model, selected_model)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个智能助手"},
                    {"role": "user", "content": system_prompt},
                ],
                stream=False,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"\nError: {str(e)}"

    if selected_api.lower() == "openai":
        generator = generate_openai()
    elif selected_api.lower() in ["百度千帆", "qianfan"]:
        generator = generate_qianfan()
    else:

        def error_gen():
            return "不支持的 API 运营商。"

        generator = error_gen()

    return jsonify({"content": generator})


@rag_bp.route("/load_dataset_embedding", methods=["POST"])
def load_dataset_embedding_route():
    global _INDEX_CACHE, _LINES_CACHE
    # expert_type = request.args.get("expert_type")
    try:
        index_file = "../data/expert_index.index"
        text_file = "../data/expert_texts.txt"
        _INDEX_CACHE, _LINES_CACHE = load_dataset_embedding(
            index_file=index_file, text_file=text_file
        )
        return jsonify(
            {
                "message": "Dataset embedding loaded successfully!",
                "index_cache_length": len(_INDEX_CACHE) if _INDEX_CACHE else 0,
                "lines_cache_length": len(_LINES_CACHE) if _LINES_CACHE else 0,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
