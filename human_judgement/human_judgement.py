import os

import pandas as pd
from tqdm import tqdm

from pipeline.evaluation_ground_truth import get_llm_judgement
from metric_analysis.evaluation_judges import get_judges

system_prompts = [
    (
        """
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the helpfulness, relevance, accuracy, depth, and level of detail of the response. 
            You will be given a reference answer and the assistant’s answer. Be as objective as possible.
            You must rate the AI assistant's response on a scale of 1 to 5 by strictly following this format: 
            "Rating: [[rating]]", for example: "Rating: [[3]]".
        """,
        "no_cot",
    ),
    (
        """
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the helpfulness, relevance, accuracy, depth, and level of detail of the response. 
            You will be given a reference answer and the assistant’s answer. 
            Begin your evaluation by comparing the assistant’s answer with the reference answer. 
            Identify and correct any mistakes. Be as objective as possible.
            After providing your explanation, you must rate the AI assistant's response on a scale of 1 to 5 
            by strictly following this format: "Rating: [[rating]]", for example: "Rating: [[3]]".
        """,
        "cot",
    ),
    (
        """
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the helpfulness, relevance, accuracy, depth, and level of detail of the response. 
            You will be given a reference answer and the assistant’s answer. 
            Begin your evaluation by comparing the assistant’s answer with the reference answer. 
            Identify and correct any mistakes. Be as objective as possible.
            After providing your explanation, you must rate the AI assistant's response on the following scale of 1 to 5:
            - 5	Perfect: Output matches the gold answer exactly or improves upon it with no errors.
            - 4	Good: Minor deviations from gold answer; still valid, useful, and accurate.
            - 3	Acceptable: Reasonable but some flaws; not as complete or precise.
            - 2	Poor: Significant issues; only partially correct or unclear.
            - 1	Bad: Completely wrong or misleading.
            Output the score strictly following this format: "Rating: [[rating]]", for example: "Rating: [[3]]".
        """,
        "cot_likert",
    ),
    (
        """
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
            the response. Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, please rate the response on a scale of 1 to 5 by strictly following this format:
            "Rating: [[rating]]", for example: "Rating: [[3]]".
        """,
        "no_reference",
    ),
]


# Likert Scale:
# 5	Perfect: Output matches the gold answer exactly or improves upon it with no errors.
# 4	Good: Minor deviations from gold answer; still valid, useful, and accurate.
# 3	Acceptable: Reasonable but some flaws; not as complete or precise.
# 2	Poor: Significant issues; only partially correct or unclear.
# 1	Bad: Completely wrong or misleading.


def create_sample_sovanta():
    df = pd.read_csv("../eval_sovanta/pred-full-2025-05-27_14_48_21-fixed.csv")
    sampled_prompts = df.groupby("cluster").sample(n=10, random_state=42)
    filtered_df = df[df["prompt"].isin(sampled_prompts["prompt"])]
    grp = filtered_df.groupby("prompt")
    final_sample = grp.sample(n=2, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(final_sample)} sovanta samples")
    return final_sample


def create_sample_wikieval():
    df = pd.read_csv("../eval_wikieval/pred-wikieval-full-2025-05-12_20_33_35.csv")
    sampled_prompts = df.groupby("prompt").sample(n=2, random_state=42)
    print(f"Loaded {len(sampled_prompts)} wikieval samples")
    return sampled_prompts


def label_data_interactively(sample_df, output_csv):
    if os.path.exists(output_csv):
        labeled_df = pd.read_csv(output_csv)
        start_index = len(labeled_df)
        print(f"Resuming from row {start_index}")
    else:
        labeled_df = pd.DataFrame(columns=list(sample_df.columns) + ["human_judgement"])
        start_index = 0
        labeled_df.to_csv(output_csv, index=False)

    for i in range(start_index, len(sample_df)):
        row = sample_df.iloc[i]
        print(f"\n--- Row {i + 1} ---")
        print(f"Prompt:      {row['prompt']}")
        print(f"Gold Answer: {row['gold_answer']}")
        print(f"Answer:      {row['answer']}")

        while True:
            try:
                score = (int(input("Enter score (1-5): ")) - 1) / 4
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

        row_with_score = row.to_dict()
        row_with_score["human_judgement"] = score

        pd.DataFrame([row_with_score]).to_csv(output_csv, mode="a", header=False, index=False)

        print(f"Saved row {i + 1}/{len(sample_df)} to {output_csv}")

    print("\nAll rows have been labeled.")


def evaluate_judges(df_labeled_path: str):
    judges = get_judges(df_labeled_path)
    df = pd.read_csv(df_labeled_path)
    tqdm.pandas()
    for judge in judges:
        for prompt, prompt_name in system_prompts:
            print(f"Running {judge} on {df_labeled_path} with prompt {prompt_name}")
            df[f"{judge}_judgement_{prompt_name}"] = df.progress_apply(
                lambda row: get_llm_judgement(row, model_name=judge, system_prompt=prompt),
                axis=1,
            )
            print(
                f"{judge} / {prompt_name} correlation with human judgement: {df[f'{judge}_judgement_{prompt_name}'].corr(df['human_judgement'])}"
            )
            df.to_csv(f"eval-llm-judge-{'wikieval' if 'wikieval' in df_labeled_path else 'sovanta'}.csv")


if __name__ == "__main__":
    df_sov = create_sample_sovanta()
    label_data_interactively(df_sov, output_csv="human-label-sovanta.csv")

    df_wik = create_sample_wikieval()
    label_data_interactively(df_wik, output_csv="human-label-wikieval.csv")

    evaluate_judges("human-label-wikieval.csv")
    evaluate_judges("human-label-sovanta.csv")
