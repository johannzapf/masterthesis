import os.path
import time
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

from app.controllers.admin_controller import AdminController
from app.controllers.chat_controller import ChatController
from app.hyperparameters import hyperparameters
from app.model.pydantic_model import PNode
from app.settings import module_registry
from app.utils.db import db_session
from app.utils.helpers import update_env_vars, update_key_value

USER_ID = 1

HP_TOK_K = [2, 4, 6, 8, 10, 12, 14]
HP_CHUNK_SIZE = [64, 128, 256, 512]
HP_EMBEDDING_MODEL = ["BAAI/bge-m3", "intfloat/multilingual-e5-large"]
HP_LLM = [
    "gemini-1.5-pro",
    "gpt-4o",
    "meta--llama3.1-70b-instruct",
    "mistralai--mistral-large-instruct",
]
HP_RERANK_MODEL = [
    None,
    "hooman650/bge-reranker-v2-m3-onnx-o4",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "mixedbread-ai/mxbai-rerank-xsmall-v1",
]


def predict(
    prompt: str, context_id: int, retrieval_only: bool
) -> Tuple[bool, str, float, float, float, float, int, int, List[PNode]]:
    answer = ChatController.send_single_prompt(
        context_id=context_id,
        prompt=prompt,
        retrieval_only=retrieval_only,
        user_id=USER_ID,
        pass_key="",
        save_chat_history=False,
        convert_response_to_html=False,
    )
    if answer.is_error:
        print("WARNING: Prompt resulted in error.")
        time.sleep(5)  # Waiting 5 Seconds for GenAI Hub Usage to refresh
    return (
        answer.is_error,
        answer.response,
        answer.timings.total_runtime,
        answer.timings.retrieval_time,
        answer.timings.embedding_time,
        answer.timings.generation_time,
        answer.input_tokens,
        answer.completion_tokens,
        answer.nodes,
    )


def rerun_errors(file_path: str):
    data = pd.read_csv(file_path)
    data["rerank_model"] = data["rerank_model"].fillna("None")
    # error_rows = data[(data["answer"].isna()) | (data["is_error"] == True)].copy()
    error_rows = data[(data["is_error"] == True)].copy()

    if error_rows.empty:
        print("No errors found in the file.")
        data.to_csv(file_path.replace(".csv", "-fixed.csv"), index=False)
        return

    print(f"Retrying {len(error_rows)} failed predictions...")

    new_data = data.copy()
    for index, row in tqdm(error_rows.iterrows(), total=len(error_rows)):
        with update_env_vars(new_env_vars={"EMBED_ONNXEmbedding_onnx_model_name": row["embed_model"]}):
            AdminController.set_default_llm(row["llm"])
            hyperparameters.chunking_size = row["chunk_size"]

            if row["rerank_model"] == "None":
                hyperparameters.use_reranking = False
                hyperparameters.top_k = row["top_k"]
                hyperparameters.top_k_reranked = None
                hyperparameters.reranking_model = None
            else:
                hyperparameters.use_reranking = True
                hyperparameters.reranking_model = row["rerank_model"]
                hyperparameters.top_k = 30
                hyperparameters.top_k_reranked = row["top_k"]

            module_registry.clear()

            for i in range(1, 5):
                prediction = predict(
                    prompt=row["prompt"],
                    context_id=int(row["context_id"]),
                    retrieval_only=False,
                )
                if not prediction[0]:  # Check if is_error is False
                    print(f"Fixed error at index {index}.")
                    (
                        is_error,
                        answer,
                        runtime,
                        retrieval_time,
                        embedding_time,
                        generation_time,
                        input_tokens,
                        completion_tokens,
                        nodes,
                    ) = prediction

                    new_data.at[index, "is_error"] = is_error
                    new_data.at[index, "answer"] = answer
                    new_data.at[index, "runtime"] = runtime
                    new_data.at[index, "retrieval_time"] = retrieval_time
                    new_data.at[index, "embedding_time"] = embedding_time
                    new_data.at[index, "generation_time"] = generation_time
                    new_data.at[index, "input_tokens"] = input_tokens
                    new_data.at[index, "completion_tokens"] = completion_tokens
                    new_data.at[index, "nodes"] = nodes
                    break
                print(f"Retrying index {index} (Retry {i})...")
                time.sleep(5)

    new_data.to_csv(file_path.replace(".csv", "-fixed.csv"), index=False)
    print("Updated file with corrected predictions.")


def run_grid_search(
    file_name: str,
    retrieval_only=False,
    is_wikieval=False,
    stop_after: datetime | None = None,
    out_path=None,
):
    data = pd.read_csv(file_name)
    columns = [
        "embed_model",
        "llm",
        "chunk_size",
        "top_k",
        "rerank_model",
        "is_error",
        "answer",
        "runtime",
        "retrieval_time",
        "embedding_time",
        "generation_time",
        "input_tokens",
        "completion_tokens",
        "nodes",
        "prompt",
        "gold_answer",
        "relevant_text",
        "context_id",
    ]
    if is_wikieval:
        columns.extend(["poor_answer", "ungrounded_answer", "wiki_page"])
    else:
        columns.extend(["sources", "documents", "cluster"])

    if out_path:
        file_path = out_path
    else:
        pred_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H_%M_%S")
        file_path = f"{'eval_wikieval/' if is_wikieval else 'eval_sovanta/'}pred-{'retrieval' if retrieval_only else 'full'}-{pred_time}.csv"
    with tqdm(
        total=len(HP_EMBEDDING_MODEL)
        * len([None] if retrieval_only else HP_LLM)
        * len(HP_CHUNK_SIZE)
        * len(HP_TOK_K)
        * len(HP_RERANK_MODEL)
        * data.shape[0],
        smoothing=0,
    ) as pbar:
        for embed_model in HP_EMBEDDING_MODEL:
            with db_session() as sess:
                update_key_value(
                    sess,
                    "embed",
                    {
                        "EMBED_CLASS": "app.modules.embed.ONNXEmbedding",
                        "EMBED_ONNXEmbedding_onnx_model_name": embed_model,
                    },
                )
            for llm in [None] if retrieval_only else HP_LLM:
                if llm and "gemma" in llm:
                    new_env_vars = {
                        "LLM_CLASS": "app.modules.llm.LocalLLM",
                        "LLM_LocalLLM_genai_llm_name": "gemma3:27b",
                    }
                else:
                    new_env_vars = {
                        "LLM_CLASS": "app.modules.llm.GenAIHubLLMWithFallback",
                    }
                with update_env_vars(new_env_vars=new_env_vars):
                    if isinstance(llm, str):
                        AdminController.set_default_llm(llm)
                    for chunk_size in HP_CHUNK_SIZE:
                        for top_k in HP_TOK_K:
                            with open("augmentation-state.txt", "r") as f:
                                if f.readline() != "CONTINUE" or (stop_after and stop_after < datetime.now()):
                                    print(
                                        f"Stopping before iteration {embed_model} / {llm} on {top_k} chunks of {chunk_size} tokens"
                                    )
                                    return
                            for rerank_model in HP_RERANK_MODEL:
                                print(
                                    f"Running {embed_model} / {llm} on {top_k} chunks of {chunk_size} tokens with {rerank_model} rerank model"
                                )

                                hyperparameters.chunking_size = chunk_size
                                if rerank_model is None:
                                    hyperparameters.use_reranking = False
                                    hyperparameters.top_k = top_k
                                    hyperparameters.top_k_reranked = None
                                    hyperparameters.reranking_model = None
                                else:
                                    hyperparameters.use_reranking = True
                                    hyperparameters.reranking_model = rerank_model
                                    hyperparameters.top_k = 30
                                    hyperparameters.top_k_reranked = top_k

                                module_registry.clear()

                                ctx_column_name = f"eval_context_id_{chunk_size}_{embed_model}"

                                results = []

                                for _, row in data.iterrows():
                                    if retrieval_only and row["relevant_text"] == "[]":
                                        continue
                                    prediction = predict(
                                        prompt=row["prompt"],
                                        context_id=int(row[ctx_column_name]),
                                        retrieval_only=retrieval_only,
                                    )
                                    res = [
                                        embed_model,
                                        llm,
                                        chunk_size,
                                        top_k,
                                        rerank_model,
                                        *prediction,  # Unpack predict() output
                                        row["prompt"],
                                        row["gold_answer"],
                                        row["relevant_text"],
                                        row[ctx_column_name],
                                    ]
                                    if is_wikieval:
                                        res.extend(
                                            [
                                                row["poor_answer"],
                                                row["ungrounded_answer"],
                                                row["wiki_page"],
                                            ]
                                        )
                                    else:
                                        res.extend(
                                            [
                                                row["source"],
                                                row["documents"],
                                                row["cluster"],
                                            ]
                                        )
                                    results.append(res)
                                    pbar.update(1)

                                res = pd.DataFrame(results, columns=columns)
                                if os.path.exists(file_path):
                                    res.to_csv(file_path, index=False, mode="a", header=False)
                                else:
                                    res.to_csv(file_path, index=False)


if __name__ == "__main__":
    hyperparameters.is_eval = True
    run_grid_search("../dataset/wikieval-dataset.csv", retrieval_only=False, is_wikieval=True)
