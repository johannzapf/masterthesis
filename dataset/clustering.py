from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gen_ai_hub.proxy.native.openai import chat as genai_chat
from ollama import ChatResponse, chat
from tqdm import tqdm

pd.set_option("display.width", 1000)

clusters = [
    "'HR': Everything related to human resources and company policies",
    "'TECH': Everything related to information technology and data science",
    "'SUMMARY': When the user asks for a summary, a creative writing or a long text, or when the Answer is a long text",
    "'CONTRACTS': Questions regarding contracts and projects",
    "'OTHER': Everything that you can not clearly attribute to one of the other categories",
]

cluster_df = pd.read_csv("sovanta-dataset.csv")


def predict_category(row, model_name="meta--llama3.1-70b-instruct", is_ollama=False):
    if row["relevant_text"] == "[]":
        return "UNKNOWN"
    base_model = genai_chat.completions.create

    system_prompt = (
        "You are an AI classifier. Your task is to assign a category label to a pair of Q&A entries "
        "based on the following categories:\n\n"
        + "\n".join(clusters)
        + "\n\nReturn only the category name in uppercase, such as HR, TECH, SUMMARY, CONTRACTS, or OTHER.\n"
        "Do not explain your choice."
    )
    user_input = f"""Prompt: {row["prompt"]}\nAnswer: {row["gold_answer"]}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    if is_ollama:
        response: ChatResponse = chat(model=model_name, messages=messages)
        category = response.message.content
    else:
        response = base_model(model=model_name, messages=messages)
        category = response.choices[0].message.content.replace("\n", " ")
    print(f"{model_name} predicted category {category} for prompt {row['prompt']}".replace("\n", ""))
    return category


def cluster():
    df = pd.read_csv("legacy/prompts-target-sov.csv")
    tqdm.pandas()
    df["cluster"] = df.progress_apply(
        lambda row: predict_category(row, model_name="gemma3:27b", is_ollama=True),
        axis=1,
    )
    df.to_csv("prompts-target-sov-clustered.csv", index=False)


def cluster_distribution():
    print("Cluster Distribution:\n")
    print(cluster_df["cluster"].value_counts())


def get_cluster(prompt):
    matches = cluster_df[cluster_df["prompt"] == prompt]
    if matches.empty:
        raise ValueError("No cluster found for prompt '{}'".format(prompt))
    if matches.shape[0] != 1:
        raise ValueError("Dataset not unique WRT prompt '{}'".format(prompt))
    return matches.iloc[0]["cluster"]


def add_clusters_to_df(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["cluster"] = df.apply(lambda row: get_cluster(row["prompt"]), axis=1)
    return df


if __name__ == "__main__":
    cluster_distribution()
