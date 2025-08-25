import os
import re

import httpcore
import httpx
import openai
import pandas as pd
from gen_ai_hub.orchestration.exceptions import OrchestrationError
from gen_ai_hub.orchestration.service import OrchestrationService
from retry import retry
from tqdm import tqdm
from app.modules.llm import GenAIHubLLM
from utils.genaihub_provider import get_orchestration_config
from utils.utils import PNode

##### RETRIEVAL ERROR CLASSES
sys_prompt_ret = """
You are an expert language model evaluator tasked with analyzing retrieval quality in a retrieval-augmented generation (RAG) system. You will be given:
- A query: the user’s question.
- A relevant text: this contains the gold-standard information needed to correctly answer the query.
- A retrieved text: this is the passage retrieved by the system, intended to help answer the query.
Your task is to classify the retrieval error by comparing the retrieved text against the relevant text and the query. Choose exactly one of the following classes:

1: The retrieved text does contain the needed information, but the relevant text contains extra or broader content that is technically unnecessary to answer the query or describes the information differently. The essential answer is still present.
2: The retrieved text omits important details that are needed to answer the query properly or entirely lacks the necessary information to answer the question.
OTHER: Any other kind of deviation that does not fall into the categories above.

Output your result in the following format:
[[n]]
where [[n]] should be replaced by one of: [[1]], [[2]] or [[OTHER]].
"""
# Class 3: The retrieved text contains the relevant text and therefore has high recall, but Context Relevance fails to classify it as relevant.

##### GENERATION ERROR CLASSES
sys_prompt_legacy = """
You are an expert in evaluating natural language generation quality. Your task is to help analyze cases where automatic evaluation metrics (like BLEU and ROUGE) diverge significantly from LLM-based or human judgment of quality.
You will be given:
- A query: the original question asked by the user.
- A gold answer: the reference or expected answer.
- A model answer: the actual generated answer.
Your task is to classify the reason why BLEU/ROUGE may have failed, using one of the following error classes:

1 - Over-Elaboration: The model answer includes extensive elaboration, giving a lot more information than the gold answer, but with the information from the gold answer still present.
2 – Lexical Variation (Same Meaning, Different Words): The model answer expresses the same meaning as the gold answer using different words, phrasing, sentence structure or slightly more information. BLEU/ROUGE fail to capture the semantic equivalence.
OTHER: Any other kind of deviation that does not fall into the categories above.

Output your result in the following format:
[[n]]
where [[n]] should be replaced by one of: [[1]], [[2]] or [[OTHER]].
"""

sys_prompt_ref_free = """
You are an expert in evaluating natural language generation quality. Your task is to help analyze cases where reference-free LLM judgement diverges significantly from reference-based LLM judgement.
You will be given:
- A query: the original question asked by the user.
- A gold answer: the reference or expected answer.
- A model answer: the actual generated answer.
Your task is to classify the reason why the reference-free judgement may have failed, using one of the following error classes:

1 - Language Diversion: The model answer is semantically correct, but it uses a different language than the gold answer.
2 - Overly Short Answer: The model answer closely resembles the gold answer, but it is very short or minimal, leading the reference-free method to underestimate its correctness due to lack of elaboration or context.
3 - Missing Information: The answer partially answers the question, but certain information from the gold answer is missing.
4 - “I Don’t Know” is the Correct Answer: The gold answer indicates that the information is not present in the context. Reference-free scoring fails to recognize this as correct due to lack of supporting context.
OTHER: Any other kind of deviation that does not fall into the categories above.

Output your result in the following format:
[[n]]
where [[n]] should be replaced by one of: [[1]], [[2]], [[3]], [[4]] or [[OTHER]].
"""

sys_prompt_rag_triad = """
You are an expert in evaluating natural language generation quality. Your task is to help analyze cases where reference-free LLM judgement diverges significantly from reference-based LLM judgement.
You will be given:
- A query: the original question asked by the user.
- A gold answer: the reference or expected answer.
- A model answer: the actual generated answer.
Your task is to classify the reason why the reference-free judgement may have failed, using one of the following error classes:

1 - Partially Correct Answer: The answer contain parts of the information from the gold answer, but certain information from the gold answer is missing.
2 - Related Answer: The answer is topically related to the query and may seem plausible, but does not contain any of of the information from the gold answer.
3 - “I Don’t Know” is the Correct Answer or Question Unclear: The gold answer indicates that the information is not present in the context or asks a follow-up question for clarification. Reference-free scoring fails to recognize this as correct due to lack of supporting context.
OTHER: Any other kind of deviation that does not fall into the categories above.

Output your result in the following format:
[[n]]
where [[n]] should be replaced by one of: [[1]], [[2]], [[3]] or [[OTHER]].
"""


def human_label_errors(df, pred, y, is_retrieval, output_path, diff_thresh=0.5):
    if not is_retrieval:
        df["legacy_metrics"] = df[["bleu", "rouge1", "bertscore_recall"]].mean(
            axis=1
        )  # the legacy metrics that work best
    df["abs_diff"] = (df[y] - df[pred]).abs()
    highlighted_df = df[df["abs_diff"] > diff_thresh]
    print(
        f"Found {len(highlighted_df)} rows with absolute differences above {diff_thresh}. Sampling 30 for human labeling"
    )
    sample = highlighted_df.sample(n=30, random_state=42)
    sample["retrieved_text"] = sample.apply(
        lambda row: " ".join([n.text for n in pd.eval(row["nodes"], local_dict={"PNode": PNode})]),
        axis=1,
    )

    if os.path.exists(output_path):
        labeled_df = pd.read_csv(output_path)
        start_index = len(labeled_df)
        print(f"Resuming from row {start_index}")
    else:
        labeled_df = pd.DataFrame(columns=list(sample.columns) + ["human_error_class"])
        start_index = 0
        labeled_df.to_csv(output_path, index=False)

    for i in range(start_index, len(sample)):
        row = sample.iloc[i]
        print(f"\n--- Row {i + 1} ---")
        print(f"Prompt:      {row['prompt']}")
        print(f"{pred}:      {row[pred]}")
        print(f"{y}:      {row[y]}")
        if is_retrieval:
            print("-----------------------------\nRelevant Text:\n", row["relevant_text"])
            print(
                "-----------------------------\nRetrieved Text:\n",
                row["retrieved_text"],
            )
        else:
            print(f"-----------------------------\nGold Answer:\n{row['gold_answer']}")
            print(f"-----------------------------\nAnswer:\n{row['answer']}")

        score = input("Enter error class: ")

        row_with_score = row.to_dict()
        row_with_score["human_error_class"] = score

        pd.DataFrame([row_with_score]).to_csv(output_path, mode="a", header=False, index=False)

        print(f"Saved row {i + 1}/{len(sample)} to {output_path}")

    print("\nAll rows have been labeled.")


@retry(
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        OrchestrationError,
        httpx.HTTPError,
        httpcore.ReadTimeout,
    ),
    tries=10,
    delay=10,
    backoff=2,
)
def llm_label_row(row, llm, system_prompt, is_retrieval):
    orchestration_service = OrchestrationService(api_url=GenAIHubLLM.get_orchestration_deployment_url(), timeout=60)
    if is_retrieval:
        if row["nodes_recall_0.5"] > 0.6:
            print("Row has high Recall, automatically assigning Class 3")
            return 3
        msg = f"""
        Prompt: {row["prompt"]}
        -----------------------------\nRelevant Text:\n{row["relevant_text"]}
        -----------------------------\nRetrieved Text:\n{row["retrieved_text"]}
        """
    else:
        msg = f"""
        Prompt: {row["prompt"]}
        Gold Answer: {row["gold_answer"]}
        Actual Answer: {row["answer"]}
        """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": msg},
    ]
    config = get_orchestration_config(model_engine=llm, temperature=0, messages=messages)
    # print(f"Calling {llm} with {(" | ".join([m.get("content") for m in messages])).replace("\n", " ")}")
    try:
        result = orchestration_service.run(config)
    except OrchestrationError as e:
        if "content management policy" in e.message:
            print("Content filter triggered, skipping")
            return -1
        else:
            raise e

    message = result.orchestration_result.choices[0].message.content.replace("\n", " ")
    match = re.search(r"\[\[(1|2|3|4|OTHER)]]", message.strip(), re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        print("Model did not output a valid class")
        return -1


def llm_label_errors(df_path, llm, system_prompt):
    df = pd.read_csv(df_path)
    tqdm.pandas(desc=f"Labeling {df_path} with {llm}")
    df[f"{llm}_error_class"] = df.progress_apply(
        lambda row: llm_label_row(row, llm, system_prompt, "ret" in df_path), axis=1
    )
    df.to_csv(df_path, index=False)


def llm_label_all_errors(df, llm, y, pred, system_prompt, is_retrieval, output_path, diff_thresh=0.5):
    if is_retrieval:
        print("Dataframe is Retrieval Eval, creating retrieved_text column")
        df["retrieved_text"] = df.apply(
            lambda row: " ".join([n.text for n in pd.eval(row["nodes"], local_dict={"PNode": PNode})]),
            axis=1,
        )
    else:
        df["legacy_metrics"] = df[["bleu", "rouge1", "bertscore_recall"]].mean(
            axis=1
        )  # the legacy metrics that work best
    df["abs_diff"] = (df[y] - df[pred]).abs()
    highlighted_df = df[df["abs_diff"] > diff_thresh]
    print(f"Found {len(highlighted_df)} rows with absolute differences above {diff_thresh}.")

    tqdm.pandas(desc=f"Labeling on {pred} with {llm}")
    highlighted_df[f"{llm}_error_class"] = highlighted_df.progress_apply(
        lambda row: llm_label_row(row, llm, system_prompt, is_retrieval), axis=1
    )
    highlighted_df.to_csv(output_path, index=False)


def fix_errors():
    df = pd.read_csv("wikieval-ref-free.csv")
    mask = df["gpt-4o_error_class"] == "-1"
    print(len(df[mask]))
    tqdm.pandas(desc=f"Fixing errors", total=len(df[mask]))
    df.loc[mask, "gpt-4o_error_class"] = df.loc[mask].progress_apply(
        lambda row: llm_label_row(row, "gpt-4o", system_prompt=sys_prompt_ref_free, is_retrieval=False),
        axis=1,
    )
    df.to_csv("wikieval-ref-free-fixed.csv", index=False)


def main():
    df_sov_ret = pd.read_csv(
        "../eval_sovanta/eval-retrieval-meta--llama3.1-70b-instruct_context_relevance_with_cot.csv"
    )
    df_wik_ret = pd.read_csv(
        "../eval_wikieval/eval-retrieval-gpt-4o_context_relevance_with_cot-2025-06-09_18_52_33.csv"
    )
    df_sov = pd.read_csv("../eval_sovanta/eval-full-judgement-ref-free-cr-ar-gr.csv")
    df_sov["rag_triad"] = (
        df_sov["mistralai--mistral-large-instruct_answer_relevance"] * 0.5
        + df_sov["meta--llama3.1-70b-instruct_groundedness"] * 0.25
        + df_sov["meta--llama3.1-70b-instruct_context_relevance_with_cot"] * 0.25
    )
    df_wik = pd.read_csv("../eval_wikieval/eval-full-judgement-ref-free-cr-ar-gr.csv")
    df_wik["rag_triad"] = (
        df_wik["anthropic--claude-3.7-sonnet_answer_relevance_with_cot"] * 0.25
        + df_wik["anthropic--claude-3.7-sonnet_groundedness_filter_trivial"] * 0.5
        + df_wik["gpt-4o_context_relevance_with_cot"] * 0.25
    )

    # Retrieval
    llm_label_all_errors(
        df_sov_ret,
        llm="mistralai--mistral-large-instruct",
        y="nodes_recall_0.5",
        pred="meta--llama3.1-70b-instruct_context_relevance_with_cot",
        system_prompt=sys_prompt_ret,
        is_retrieval=True,
        output_path="sovanta-ret.csv",
    )
    llm_label_all_errors(
        df_wik_ret,
        llm="anthropic--claude-3.7-sonnet",
        y="nodes_recall_0.5",
        pred="gpt-4o_context_relevance_with_cot",
        system_prompt=sys_prompt_ret,
        is_retrieval=True,
        output_path="wikieval-ret.csv",
    )

    # Legacy Metrics
    llm_label_all_errors(
        df_wik,
        llm="gpt-4o",
        y="mistralai--mistral-large-instruct_judgement",
        pred="legacy_metrics",
        system_prompt=sys_prompt_legacy,
        is_retrieval=False,
        output_path="wikieval-legacy.csv",
    )
    llm_label_all_errors(
        df_sov,
        llm="mistralai--mistral-large-instruct",
        y="meta--llama3.1-70b-instruct_judgement",
        pred="legacy_metrics",
        system_prompt=sys_prompt_legacy,
        is_retrieval=False,
        output_path="sovanta-legacy.csv",
    )

    # Reference-free
    llm_label_all_errors(
        df_wik,
        llm="gpt-4o",
        y="mistralai--mistral-large-instruct_judgement",
        pred="mistralai--mistral-large-instruct_judgement_ref_free",
        system_prompt=sys_prompt_ref_free,
        is_retrieval=False,
        output_path="wikieval-ref-free.csv",
        diff_thresh=0.3,
    )
    llm_label_all_errors(
        df_sov,
        llm="mistralai--mistral-large-instruct",
        y="meta--llama3.1-70b-instruct_judgement",
        pred="meta--llama3.1-70b-instruct_judgement_ref_free",
        system_prompt=sys_prompt_ref_free,
        is_retrieval=False,
        output_path="sovanta-ref-free.csv",
    )

    # RAG Triad
    llm_label_all_errors(
        df_wik,
        llm="anthropic--claude-3.7-sonnet",
        y="mistralai--mistral-large-instruct_judgement",
        pred="rag_triad",
        system_prompt=sys_prompt_rag_triad,
        is_retrieval=False,
        output_path="wikieval-rag-triad.csv",
    )
    llm_label_all_errors(
        df_sov,
        llm="mistralai--mistral-large-instruct",
        y="meta--llama3.1-70b-instruct_judgement",
        pred="rag_triad",
        system_prompt=sys_prompt_rag_triad,
        is_retrieval=False,
        output_path="sovanta-rag-triad.csv",
    )


if __name__ == "__main__":
    main()
