import ast
import re
from functools import lru_cache
from typing import Optional, Any

import pandas as pd
import pylcs
import seaborn as sns
import matplotlib.pyplot as plt
from nltk import sent_tokenize
from lingua import Language, LanguageDetectorBuilder
from pydantic import BaseModel

languages = [Language.ENGLISH, Language.GERMAN]
language_detector = LanguageDetectorBuilder.from_languages(*languages).with_low_accuracy_mode().build()


class PNode(BaseModel):
    """
    Defines a retrieved node of the RAG pipeline
    Taken from Document Chat
    """

    nodeID: str
    documentID: int
    documentName: str
    text: str
    score: float
    source: str
    origin: Optional[str]
    type: Optional[str]
    documentPage: Optional[int]
    ocr: bool = False


class PToolResponse(BaseModel):
    """
    Defines a tool response
    Taken from Document Chat
    """

    tool_name: str
    tool_inputs: dict[str, Any]
    tool_response: str


class PChatResponse(BaseModel):
    """
    Defines a response of the RAG pipeline
    Taken from Document Chat
    """

    response: str
    response_markdown: str
    nodes: list[PNode]
    duration_ms: Optional[float] = None
    input_tokens: int
    completion_tokens: int
    span_tree: str
    is_error: bool
    hp_candidate: Optional[dict] = None
    choice_id: Optional[int] = None
    used_llm: Optional[str] = None
    used_tools: Optional[list[PToolResponse]] = None


def compute_german_confidence(text: str) -> float:
    """
    Returns the confidence that given text is German
    :param text:
    """
    confidence_values = language_detector.compute_language_confidence_values(text)
    for c in confidence_values:
        if c.language == Language.GERMAN:
            return round(c.value, 2)
    return 0.0


def cleanup_string(s):
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower()


@lru_cache(maxsize=100000)
def get_node_score(y: str, ground_truth: str, lcs_thresh=0.3):
    if not ground_truth or not isinstance(ground_truth, str) or not y:
        return 0
    if ground_truth in y:
        return 1
    ns = pylcs.lcs_string_length(y, ground_truth) / min(len(y), len(ground_truth))
    if ns > lcs_thresh:
        return ns
    return 0


def get_node_recall(row, lcs_thresh=0.3, rel_text_column="relevant_text"):
    if isinstance(row[rel_text_column], list):
        target = [cleanup_string(s) for s in row[rel_text_column]]
    else:
        target = [cleanup_string(s) for s in ast.literal_eval(row[rel_text_column].replace("\n", ""))]
    if len(target) == 0:
        return 1
    nodes = pd.eval(row["nodes"], local_dict={"PNode": PNode})
    y = cleanup_string("".join([n.text for n in nodes]))
    nodes_to_find = len(target)
    nodes_found = 0
    for node in target:
        nodes_found += get_node_score(y, node, lcs_thresh=lcs_thresh)
    return nodes_found / nodes_to_find


def sentence_tokenize_text(text: str):
    r = ast.literal_eval(text.replace("\n", ""))
    new_chunks = []
    for r in r:
        new_chunks.extend(sent_tokenize(r))
    return new_chunks


def print_correlation_table(df, judges, group_by, context_col_template, out=True):
    records = []
    df["rerank_model"] = df["rerank_model"].fillna("None")

    grouped = df.groupby(group_by)

    for group_vals, group_df in grouped:
        group_dict = dict(zip(group_by, group_vals if isinstance(group_vals, tuple) else [group_vals]))
        row = group_dict.copy()
        for judge in judges:
            col_name = context_col_template.format(judge=judge)
            if col_name in group_df.columns:
                corr = group_df["nodes_recall_0.5"].corr(group_df[col_name])
                row[judge] = corr
        records.append(row)

    result_df = pd.DataFrame(records)
    if out:
        print(result_df.set_index(group_by))
    return result_df


def plot_recall_correlations(
    df,
    judges,
    group_by,
    context_col_template,
    xlabel="Chunk Size / Top K",
    ylabel="Correlation with Nodes Recall",
    title="Correlations between Judge Assessments and Nodes Recall",
):
    result_df = print_correlation_table(df, judges, group_by, context_col_template, out=False)
    plt.figure(figsize=(16, 6))
    # If a specific group_by is provided
    if isinstance(group_by, str):
        group_by = [group_by]
    x_col = group_by[0] if len(group_by) == 1 else "Group"
    # If there are multiple group_by columns, combine them
    if len(group_by) > 1:
        plot_df = result_df.copy()
        plot_df["Group"] = plot_df[group_by].apply(lambda row: "/".join(str(val) for val in row), axis=1)
    else:
        plot_df = result_df.copy()

    # Determine which judge columns to plot
    judge_cols = [col for col in judges if col in result_df.columns]

    # Create color map for judges
    color_map = {judge: color for judge, color in zip(judge_cols, sns.color_palette("hsv", len(judge_cols)))}

    # Plot each judge's correlation
    for judge in judge_cols:
        if x_col in plot_df.columns:
            x_values = plot_df[x_col]
        else:
            x_values = plot_df.index

        plt.scatter(x_values, plot_df[judge], label=judge, color=color_map[judge], s=100)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(title="Judge")

    # Adjust x-tick labels if they're too crowded
    if len(plot_df) > 8:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def plot_metric(
    data,
    group_by: str,
    y: str,
    key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    exclude_small_rerank=False,
    exclude_rerank=False,
    metric="mean",
    exclude_n_worst=0,
):
    data = data[data[y] != -1]
    data["rerank_model"] = data["rerank_model"].fillna("None")
    data["llm"] = data["llm"].fillna("None")
    data["chunk_top_k"] = data["chunk_size"].astype(str) + "/" + data["top_k"].astype(str)
    data["chunk_top_k"] = pd.Categorical(
        data["chunk_top_k"],
        categories=sorted(data["chunk_top_k"].unique(), key=lambda x: tuple(map(int, x.split("/")))),
        ordered=True,
    )
    data["embed_rerank"] = data["embed_model"] + " + " + data["rerank_model"]
    data["llm_rerank"] = data["llm"] + " + " + data["rerank_model"]
    if exclude_small_rerank:
        df = data[~data["rerank_model"].str.contains("mixedbread", na=False)]
    elif exclude_rerank:
        df = data[~data["rerank_model"].str.contains("rerank", na=False)]
    else:
        df = data

    if exclude_n_worst > 0:
        df = (
            df.sort_values(by=y, ascending=True)
            .groupby([group_by, key], observed=False)
            .apply(lambda g: g.iloc[exclude_n_worst:] if len(g) > exclude_n_worst else g)
            .reset_index(drop=True)
        )

    if metric == "mean":
        grouped_data = df.groupby([group_by, key], as_index=False, observed=False)[y].mean()
    elif metric == "median":
        grouped_data = df.groupby([group_by, key], as_index=False, observed=False)[y].median()
    elif metric == "min":
        grouped_data = df.groupby([group_by, key], as_index=False, observed=False)[y].min()
    elif metric == "max":
        grouped_data = df.groupby([group_by, key], as_index=False, observed=False)[y].max()
    elif metric == "std":
        grouped_data = df.groupby([group_by, key], as_index=False, observed=False)[y].std()

    unique_models = grouped_data[key].unique()
    color_map = {model: color for model, color in zip(unique_models, sns.color_palette("hsv", len(unique_models)))}
    colors = grouped_data[key].map(color_map)

    plt.figure(figsize=(16, 6))
    plt.scatter(grouped_data[group_by], grouped_data[y], c=colors, s=100)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_map[model],
            markersize=10,
            label=model,
        )
        for model in unique_models
    ]
    plt.legend(title=key, handles=handles)
    plt.title(title)
    plt.grid(True)

    plt.show()

    second_group = (
        grouped_data.groupby([key], as_index=False)[y]
        .mean()
        .sort_values(by=[y], ascending=False)
        .rename(columns={y: f"{metric}_{y}"})
    )
    print(second_group)


def plot_scores(
    df,
    group_by,
    columns_to_plot,
    xlabel="Group",
    ylabel="Mean Score",
    title="Mean Scores by Group",
):
    df["rerank_model"] = df["rerank_model"].fillna("None")
    df["embed_model"] = df["embed_model"].str.replace("BAAI/", "").str.replace("intfloat/", "")
    df["llm"] = df["llm"].fillna("None")
    df["chunk_top_k"] = df["chunk_size"].astype(str) + "/" + df["top_k"].astype(str)
    df["chunk_top_k"] = pd.Categorical(
        df["chunk_top_k"],
        categories=sorted(df["chunk_top_k"].unique(), key=lambda x: tuple(map(int, x.split("/")))),
        ordered=True,
    )
    df["embed_rerank"] = df["embed_model"] + " + " + df["rerank_model"]
    df["llm_rerank"] = df["llm"] + " + " + df["rerank_model"]
    plt.figure(figsize=(16, 6))

    plot_df = df.copy()
    x_col = group_by

    # Calculate means for each group and column
    grouped_means = {}
    for col in columns_to_plot:
        if plot_df[x_col].dtype == "object":
            plot_df[x_col] = plot_df[x_col].apply(lambda x: (str(x)[:25] + "...") if len(str(x)) > 25 else str(x))
        grouped_means[col] = plot_df.groupby(x_col, observed=False)[col].mean()

    # Create color map for different columns
    color_map = {col: color for col, color in zip(columns_to_plot, sns.color_palette("hsv", len(columns_to_plot)))}

    # Plot each column
    for col in columns_to_plot:
        plt.scatter(
            grouped_means[col].index,
            grouped_means[col].values,
            label=col,
            color=color_map[col],
            s=100,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(title="Metrics")

    # Adjust x-tick labels if they're too crowded
    if len(grouped_means[columns_to_plot[0]]) > 8:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

    # Display mean values for each column across all groups
    overall_means = df[columns_to_plot].mean().sort_values(ascending=False)
    print("Overall mean values:")
    print(overall_means)


def display_grouped_correlations(data, group_by: str, y1: str, y2: str):
    data = data[data[y1] != -1]
    data = data[data[y2] != -1]
    data["rerank_model"] = data["rerank_model"].fillna("None")
    data["llm"] = data["llm"].fillna("None")
    data["chunk_top_k"] = data["chunk_size"].astype(str) + "/" + data["top_k"].astype(str)
    data["chunk_top_k"] = pd.Categorical(
        data["chunk_top_k"],
        categories=sorted(data["chunk_top_k"].unique(), key=lambda x: tuple(map(int, x.split("/")))),
        ordered=True,
    )
    data["embed_rerank"] = data["embed_model"] + " + " + data["rerank_model"]
    data["llm_rerank"] = data["llm"] + " + " + data["rerank_model"]

    correlations = []
    for group, group_df in data.groupby(group_by, observed=False):
        if len(group_df) >= 2:
            corr = group_df[[y1, y2]].corr().iloc[0, 1]
        else:
            corr = float("nan")
        correlations.append({group_by: group, f"corr_{y1}_{y2}": corr})

    corr_df = pd.DataFrame(correlations).dropna()
    corr_df = corr_df.sort_values(by=f"corr_{y1}_{y2}", ascending=False)

    print(corr_df[[group_by, f"corr_{y1}_{y2}"]])


def add_recall_and_judgement_to_trulens_data(trulens_df, judgement_df):
    idx_columns = [
        "embed_model",
        "llm",
        "rerank_model",
        "top_k",
        "chunk_size",
        "prompt",
        "documents",
    ]
    is_not_unique = (
        trulens_df.duplicated(subset=idx_columns).any() and judgement_df.duplicated(subset=idx_columns).any()
    )
    if is_not_unique:
        raise ValueError("One of the two Dataframes is not unique")
    merged_df = trulens_df.merge(
        judgement_df[idx_columns + ["gemma3_judgement", "nodes_recall"]],
        on=idx_columns,
        how="inner",
    )
    print("Share before merge: ", trulens_df.shape, "Shape after merge: ", merged_df.shape)
    return merged_df


def fix_relevant_texts(file_path: str):
    df = pd.read_csv(file_path)
    # print(df["relevant_text"].value_counts())
    occ = df["relevant_text"]
    exp_str = None
    for i in occ:
        if "experience management, a broad discipline" in i:
            exp_str = i
    strs = [
        (
            '["enabling Erste Bank to make strategic, customer-focused decisions and continuously enhance the overall customer experience (CX)."]',
            "[]",
        ),
        (
            '["CUSTOMER RESPONSE LEVELS", "CAS Service Description Guide.\nService Description Documentation for SAP Cloud Application Services for BTP core operations", ""]',
            '["CUSTOMER RESPONSE LEVELS", "CAS Service Description Guide", "Service Description Documentation for SAP Cloud Application Services for BTP core operations"]',
        ),
        (
            exp_str,
            '["experience management**, a broad discipline which includes a client’s interactions with its clients (“CX”, or customer experience”), employees (“EX”, or employee experience”), and IT transformation (“ETX”, or employee technology experience)"]',
        ),
    ]
    for old_str, new_str in strs:
        count_replacements = df["relevant_text"].value_counts().get(old_str)
        df["relevant_text"] = df["relevant_text"].replace(old_str, new_str)
        print(f"Replacec {count_replacements} of {old_str}")
    df.to_csv(f"{file_path.replace('.csv', '')}-fixed-rt.csv", index=False)


def fix_add_prompts(pred_file: str):
    pred = pd.read_csv(pred_file)
    target = pd.read_csv("../dataset/prompts-target-sov.csv")
    if target.duplicated(subset=["relevant_text", "gold_answer", "sources", "documents"]).any():
        raise ValueError("Duplicate values found for 'relevant_text' and 'gold_answer' in pred")

    pred = pred.merge(
        target[["relevant_text", "gold_answer", "sources", "documents", "prompt"]],
        on=["relevant_text", "gold_answer", "sources", "documents"],
        how="left",
    )
    if pred["prompt"].isna().any():
        missing_rows = pred[pred["prompt"].isna()]
        print(f"Found {len(missing_rows)} missing rows in {pred_file}")
        pred["prompt"] = pred["prompt"].fillna("Erstelle mir eine Zusammenafassung des Dokuments")

    if pred["prompt"].isna().any():
        missing_rows = pred[pred["prompt"].isna()]
        raise ValueError(f"Some rows in pred could not be matched with target:\n{missing_rows}")
    pred.to_csv(pred_file.replace(".csv", "-with-prompts.csv"), index=False)
