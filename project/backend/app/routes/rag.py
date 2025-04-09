import os
import time
from flask import Blueprint, jsonify, request
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import openai
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
from ..utils.rag.load_and_search_from_datasets import (
    load_dataset_embedding,
    search_from_index,
)
from ..utils.rag.load_data import load_skeleton_keypoint

device = "cuda" if torch.cuda.is_available() else "cpu"
CROSS_ENCODER_MODEL = "../models/ms-marco-MiniLM-L6-v2"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
qianfan_api_key = os.getenv("QIANFAN_API_KEY")
QIANFAN_MODEL_MAPPING = {
    "ernie-x1": "ernie-x1-32k-preview",
    "ernie-4.5": "ernie-4.5-8k-preview",
    "deepseek-r1": "deepseek-r1",
    "deepseek-v3": "deepseek-v3-241226",
}
rag_bp = Blueprint("api", __name__)

INDEX_FILE = "../data/faiss_index"

retrieval_pipeline = None
_INDEX_CACHE = None
_LINES_CACHE = None


@rag_bp.route("/upload_eval", methods=["POST"])
def upload_files():
    global retrieval_pipeline
    if "files" not in request.files:
        return (
            jsonify({"status": "error", "error": "No files provided"}),
            400,
        )  # 添加 status 字段

    files = request.files.getlist("files")
    try:
        documents = process_uploaded_files(files)
        texts = split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vector_store = build_vector_store(texts, embeddings)
        retrieval_pipeline = build_retrieval_pipeline(texts, vector_store)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

    kg = retrieval_pipeline["knowledge_graph"]
    return jsonify(
        {
            "status": "success",
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
    report = data.get("report", "")
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

User's Analysis Report:
{report}

Analyze the following user input based on the context of table tennis and provide detailed feedback:
1. **Technical Analysis**: Analyze the user's body movement data (skeleton points) and ball trajectory. Identify strengths and weaknesses in their technique.
2. **Posture & Movement**: Offer suggestions to improve posture, grip, and swing mechanics based on the provided data.
3. **Injury Prevention**: Provide medical insights on avoiding injuries (e.g., wrist, elbow, shoulder) based on user movement patterns.
4. **Improvement Suggestions**: Recommend drills and exercises to enhance the player's performance based on movement analysis.

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
                base_url="https://qianfan.baidubce.com/v2",
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
    try:
        index_file = "../data/expert_index.index"
        text_file = "../data/expert_texts.txt"
        _INDEX_CACHE, _LINES_CACHE = load_dataset_embedding(
            index_file=index_file, text_file=text_file
        )
        return jsonify(
            {
                "status": "success",
                "message": "Dataset embedding loaded successfully!",
                "index_cache_length": len(_INDEX_CACHE) if _INDEX_CACHE else 0,
                "lines_cache_length": len(_LINES_CACHE) if _LINES_CACHE else 0,
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@rag_bp.route("/generate_report", methods=["POST"])
def generate_report():
    skeleton_instances, meta_info = load_skeleton_keypoint()
    client = OpenAI(
        api_key=qianfan_api_key,
        base_url="https://qianfan.baidubce.com/v2",
    )
    per_analysis = []

    batch_size = 5
    # for i in range(0, len(skeleton_instances), batch_size):
    for i in range(0, 20, batch_size):
        batch_instances = skeleton_instances[i : i + batch_size]

        # Build system prompt for the current batch of frames
        system_prompt = f"""
你是一个专业的乒乓球运动动作分析专家，擅长通过人体骨骼关键点坐标进行动作识别和姿态评估。请根据以下每帧图像中进行乒乓球对打时的人体骨骼点信息，结合字段说明，对每个人的动作进行详细分析。分析内容应包括但不限于：

1. 动作是否规范（如站立、跑步、跳跃等是否符合常规标准）； 
2. 姿态是否稳定，是否存在不协调或危险的动作；
3. 动作幅度是否合理；
4. 如存在问题，请指出具体关键点及改进建议。

为每个人进行分析，并且考虑到当前图像帧的编号，便于最后根据整体的运动轨迹进行分析。
输出标签使用：人体1、人体2、人体3
以下是骨骼点字段说明（metadata）： 
{meta_info}

以下是第 {i+1} 到 {min(i + batch_size, len(skeleton_instances))} 帧图像中的每个人体的骨骼点数据：
"""
        for j, skeleton_instance in enumerate(batch_instances):
            system_prompt += f"""
第 {i + j + 1} 帧图像中的每个人体的骨骼点数据： 
{skeleton_instance}
"""
        # Send the batch of frames to the OpenAI API for processing
        try:
            completion = client.chat.completions.create(
                model="deepseek-v3-241226",
                messages=[
                    {"role": "system", "content": "你是乒乓球运动动作分析专家"},
                    {"role": "user", "content": system_prompt},
                ],
                stream=False,
            )
            per_analysis.append(completion.choices[0].message.content)
            print(completion.choices[0].message.content)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting for 10 seconds...")
            time.sleep(10)
            completion = client.chat.completions.create(
                model="ernie-x1-32k-preview",
                messages=[
                    {"role": "system", "content": "你是乒乓球运动动作分析专家"},
                    {"role": "user", "content": system_prompt},
                ],
                stream=False,
            )
            per_analysis.append(completion.choices[0].message.content)
            print(completion.choices[0].message.content)

    # After processing all batches, generate the final report
    final_prompt = f"""
你是一个专业的乒乓球运动动作分析专家，请根据以下每一帧的分析结果，对每个人的乒乓球动作进行独立分析并撰写报告。报告应包括但不限于以下内容：

1. **动作规范性评价**：该人的动作是否符合标准乒乓球动作要求。包括站位、击球动作、挥拍角度等是否标准。
2. **危险/异常动作**：是否存在不协调或危险的动作（例如：过度伸展、非正常的身体姿势等）。
3. **动作连贯性和节奏**：该人动作是否流畅，是否有停顿或节奏问题，动作的起始和结束是否平衡。
4. **改进建议**：如果动作中存在问题，请根据每一帧中的骨骼点分析给出改进的具体建议。
5. **综合评价**：结合整体运动表现，给出该人运动表现的总体评价。

### 以下是每一帧的分析内容，基于骨骼点数据和动作分析：
{per_analysis}

请为每个的运动员生成详细的报告，每个人的分析按以下格式输出：
- **运动员1**:
    - 动作规范性评价：
    - 危险/异常动作：
    - 动作连贯性和节奏：
    - 改进建议：
    - 综合评价：
- **运动员2**:
    - 动作规范性评价：
    - 危险/异常动作：
    - 动作连贯性和节奏：
    - 改进建议：
    - 综合评价：

请确保每个运动员的动作分析清晰且详细，最终输出每个运动员的综合表现和改进建议。
"""

    response = client.chat.completions.create(
        model="ernie-x1-32k-preview",
        messages=[
            {"role": "system", "content": "你是乒乓球运动动作分析专家"},
            {"role": "user", "content": final_prompt},
        ],
        stream=False,
    )
    return jsonify({"report": response.choices[0].message.content})
