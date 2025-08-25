import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utils import compute_german_confidence, get_node_recall

# Load the metrics once
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")


def get_scores(answer, reference, prefix=""):
    if not answer or not reference or (isinstance(answer, float) and np.isnan(answer)):
        return {
            prefix + "bleu": None,
            prefix + "rouge1": None,
            prefix + "rouge2": None,
            prefix + "rougeL": None,
            prefix + "bertscore_precision": None,
            prefix + "bertscore_recall": None,
            prefix + "bertscore_f1": None,
        }
    # BLEU
    bleu_result = bleu_metric.compute(predictions=[answer], references=[[reference]])

    # ROUGE
    rouge_result = rouge_metric.compute(predictions=[answer], references=[reference])

    # BERTScore
    lang = "en" if compute_german_confidence(reference) < 0.5 else "de"
    bertscore_result = bertscore_metric.compute(predictions=[answer], references=[reference], lang=lang, device="mps")
    return {
        prefix + "bleu": bleu_result["bleu"],
        prefix + "rouge1": rouge_result["rouge1"],
        prefix + "rouge2": rouge_result["rouge2"],
        prefix + "rougeL": rouge_result["rougeL"],
        prefix + "bertscore_precision": bertscore_result["precision"][0],
        prefix + "bertscore_recall": bertscore_result["recall"][0],
        prefix + "bertscore_f1": bertscore_result["f1"][0],
    }


def compute_metrics(row, is_wikieval=False):
    if not is_wikieval:
        return pd.Series(get_scores(row["answer"], row["gold_answer"]))

    res = get_scores(row["answer"], row["gold_answer"])
    res.update(get_scores(row["poor_answer"], row["gold_answer"], "poor_"))
    res.update(get_scores(row["ungrounded_answer"], row["gold_answer"], "ungrounded_"))
    return pd.Series(res)


def run_eval(file_path: str):
    df = pd.read_csv(file_path)
    # df = df[df["answer"].isna()]

    tqdm.pandas(desc="Evaluating")
    metrics_df = df.progress_apply(lambda row: compute_metrics(row, is_wikieval="wikieval" in file_path), axis=1)

    result_df = pd.concat([df, metrics_df], axis=1)
    result_df["nodes_recall_0.5"] = result_df.progress_apply(lambda row: get_node_recall(row, lcs_thresh=0.5), axis=1)
    result_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    run_eval("../eval_sovanta/pred-full-2025-05-27_14_48_21-fixed.csv")
