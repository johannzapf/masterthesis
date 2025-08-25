import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama
import openai
import pandas as pd
from gen_ai_hub.orchestration.exceptions import OrchestrationError
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.proxy.native.openai import chat as genai_chat
from ollama import ChatResponse, chat
from retry import retry
from tqdm import tqdm

from app.modules.llm import GenAIHubLLM
from app.utils.logger import get_logger
from utils.genaihub_provider import get_orchestration_config
from utils.utils import get_node_recall

logger = get_logger(__name__)

ref_free_prompt = """
    Please act as an impartial judge and evaluate the quality of the response provided by an
    AI assistant to the user question displayed below. Your evaluation should consider factors
    such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
    the response. Begin your evaluation by providing a short explanation. Be as objective as
    possible. After providing your explanation, please rate the response on a scale of 1 to 5 by strictly following this format:
    "Rating: [[rating]]", for example: "Rating: [[3]]".
    """


@retry(
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        ollama.ResponseError,
        ollama.RequestError,
        OrchestrationError,
    ),
    tries=10,
    delay=10,
    backoff=2,
)
def _get_llm_judgement(row, model_name="meta--llama3.1-70b-instruct", is_ollama=False, system_prompt=None):
    # System Prompt https://mirascope.com/blog/llm-as-judge/
    orchestration_service = OrchestrationService(api_url=GenAIHubLLM.get_orchestration_deployment_url(), timeout=60)

    q, y, ref = row["prompt"], row["answer"], row["gold_answer"]
    # prompts from https://proceedings.neurips.cc/paper_files/paper/2023/file/91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
    # and https://arxiv.org/pdf/2502.09316v1
    if not system_prompt:
        system_prompt = """
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the helpfulness, relevance, accuracy, depth, and level of detail of the response. 
            You will be given a reference answer and the assistant’s answer. Be as objective as possible.
            You must rate the AI assistant's response on a scale of 1 to 5 by strictly following this format: 
            "Rating: [[rating]]", for example: "Rating: [[3]]".
            """

    if "reference" in system_prompt:
        msg = f"""
        [Question]
        '{q}'
                
        [The Start of Reference Answer]
        '{ref}'
        [The End of Reference Answer]
                
        [The Start of Assistant’s Answer]
        '{y}'
        [The End of Assistant’s Answer] 
        """
    else:
        msg = f"""
        [Question]
        '{q}'
                
        [The Start of Assistant’s Answer]
        '{y}'
        [The End of Assistant’s Answer] 
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]
    print(f"Calling {model_name} with {(' | '.join([m.get('content') for m in messages])).replace('\n', ' ')}")
    if is_ollama:
        response: ChatResponse = chat(model=model_name, messages=messages)
        message = response.message.content
    else:
        config = get_orchestration_config(model_engine=model_name, temperature=0, messages=messages)
        result = orchestration_service.run(config)
        message = result.orchestration_result.choices[0].message.content.replace("\n", " ")
        time.sleep(3)
    match = re.search(r"Rating: \[\[(\d+)]]", message.strip())
    if match:
        return (int(match.group(1)) - 1) / 4
    secondary_match = re.search(r"\[\[(\d+)]]", message.strip())
    if secondary_match:
        return (int(secondary_match.group(1)) - 1) / 4
    tertiary_match = re.search(r"Rating: (\d+)", message.strip())
    if tertiary_match:
        return (int(tertiary_match.group(1)) - 1) / 4
    print(f"Model {model_name} answer '{message}' does not contain a number or is too long")
    return -1


def get_llm_judgement(row, model_name="meta--llama3.1-70b-instruct", is_ollama=False, system_prompt=None):
    try:
        return _get_llm_judgement(row, model_name=model_name, is_ollama=is_ollama, system_prompt=system_prompt)
    except Exception as e:
        print(f"Model {model_name} threw exception {e}")
        return -1


def evaluate_recall(pred_path):
    eval_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H_%M_%S")
    tqdm.pandas()
    df = pd.read_csv(pred_path)
    df["nodes_recall"] = df.progress_apply(lambda row: get_node_recall(row), axis=1)
    df.to_csv(f"{pred_path.split('/')[0]}/eval-recall-{eval_time}.csv", index=False)


def evaluate_judgement(pred_path, model_name="meta--llama3.1-70b-instruct", system_prompt=None):
    eval_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_dir = "eval_wikieval" if "wikieval" in pred_path else "eval_sovanta"
    output_path = os.path.join(output_dir, f"eval-judgement-{eval_time}.csv")

    row_name = f"{model_name}_judgement{'_ref_free' if system_prompt == ref_free_prompt else ''}"

    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_ids = set(existing_df.index)
        print(f"Resuming from previous file — {len(processed_ids)} rows already processed.")
    else:
        processed_ids = set()
        print("No existing file found — starting fresh.")
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        # Write header once using empty DataFrame with correct columns
        df_preview = pd.read_csv(pred_path, nrows=1)
        df_preview[row_name] = ""
        df_preview.iloc[0:0].to_csv(output_path, index=False)

    df = pd.read_csv(pred_path)
    rows_to_process = [(i, row) for i, row in df.iterrows() if i not in processed_ids]

    def process_row(index_row):
        i, row = index_row
        row[row_name] = get_llm_judgement(row, model_name=model_name, is_ollama=False, system_prompt=system_prompt)
        return row

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_row, item): item[0] for item in rows_to_process}
        for future in tqdm(as_completed(futures), total=len(futures), smoothing=0):
            row = future.result()
            pd.DataFrame([row]).to_csv(output_path, mode="a", index=False, header=False)

    print(f"All judgements saved incrementally to {output_path}")


def fix_errors(path, model_name="meta--llama3.1-70b-instruct"):
    df = pd.read_csv(path)

    ref_free_judgement_col = f"{model_name}_judgement_ref_free"
    error_rows = df[df[ref_free_judgement_col] == -1]

    if error_rows.empty:
        print("No errors to fix — all judgements_ref_free are valid.")
        return
    print(f"Fixing {len(error_rows)} rows with judgement_ref_free== -1...")

    for idx in tqdm(error_rows.index):
        new_judgement = get_llm_judgement(
            df.loc[idx],
            model_name=model_name,
            is_ollama=False,
            system_prompt=ref_free_prompt,
        )
        if new_judgement == -1:
            print(f"({idx}) judgement_ref_free == -1...")
        df.at[idx, ref_free_judgement_col] = new_judgement

    df.to_csv(path.replace(".csv", "-fixed.csv"), index=False)


if __name__ == "__main__":
    evaluate_recall("../eval_sovanta/pred-retrieval-2025-05-25_22_56_07.csv")
    evaluate_judgement(
        "../eval_sovanta/pred-full-2025-05-27_14_48_21-fixed.csv",
        model_name="mistralai--mistral-large-instruct",
    )
    fix_errors(
        "../eval_sovanta/eval-full-judgement-ref-free-cr-ar-gr-cr.csv",
        model_name="mistralai--mistral-large-instruct",
    )
