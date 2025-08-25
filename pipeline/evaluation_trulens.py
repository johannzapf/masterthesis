import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from trulens.apps.virtual import TruVirtual, VirtualRecord
from trulens.core import Feedback, Select, TruSession
from trulens.core.feedback.feedback import GroundednessConfigs
from trulens.core.utils.threading import TP
from trulens.feedback.dummy.endpoint import DummyEndpoint
from trulens.providers.litellm import LiteLLM

from utils.genaihub_provider import GenAIHubProvider
from utils.utils import PNode

session = TruSession()
session.reset_database()
session.DEFERRED_NUM_RUNS = 1
TP.MAX_THREADS = 2

retriever_component = Select.RecordCalls.retriever
context_call = retriever_component.get_context


def create_records(df: pd.DataFrame, start=0, limit=None) -> list[VirtualRecord]:
    data = []
    num_skip = 0
    for i, record in enumerate(df.to_dict("records")):
        if limit and i >= limit + start:
            break
        # if record.get("relevant_text") == "[]":
        #    num_skip += 1
        #    continue

        if i < start + num_skip:
            print("Skipping record", record)
            continue

        nodes = pd.eval(record["nodes"], local_dict={"PNode": PNode})

        context = "\n".join([n.documentName + "\n" + n.text for n in nodes])
        rec = VirtualRecord(
            main_input=record["prompt"],
            main_output=record["answer"],
            calls={context_call: {"args": [record["prompt"]], "rets": [context]}},
            meta={"original_record": record},
        )
        data.append(rec)
    return data


def _eval(df, feedback_functions, save_dir, start=0, limit=None):
    eval_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H_%M_%S")
    feedback_list = "-".join([f.name for f in feedback_functions])
    save_path = f"{save_dir}eval-trulens-{feedback_list}-{eval_time}.csv"

    virtual_recorder = TruVirtual(
        app_id="Johann Masterarbeit",
        app={retriever_component: "This is the retriever"},
        feedbacks=feedback_functions,
    )
    records = create_records(df, start=start, limit=limit)
    for record in records:
        virtual_recorder.add_record(record)

    all_results = []
    for rec in tqdm(records, smoothing=0):
        feedbacks = {}
        for feedback, feedback_result in rec.wait_for_feedback_results().items():
            # print("\t", feedback.name, feedback_result.result)
            feedbacks.update({feedback.name: feedback_result})
        try:
            results = (
                list(rec.meta["original_record"].values())
                + [feedbacks[f.name].result for f in feedback_functions]
                + [feedbacks[f.name] for f in feedback_functions]
            )
        except KeyError:
            print("WARNING: KeyError while extracting feedback results, skipping")
            results = list(rec.meta["original_record"].values()) + [-1] + ["KeyError while extracting feedback"]

        all_results.append(results)
        res = pd.DataFrame(
            [results],
            columns=df.columns.tolist()
            + [f.name for f in feedback_functions]
            + [f"{f.name}_meta" for f in feedback_functions],
        )
        if os.path.exists(save_path):
            res.to_csv(save_path, index=False, mode="a", header=False)
        else:
            res.to_csv(save_path, index=False)
    return pd.DataFrame(
        all_results,
        columns=df.columns.tolist()
        + [f.name for f in feedback_functions]
        + [f"{f.name}_meta" for f in feedback_functions],
    )


def get_provider_from_model(model_name: str):
    if "gemma" in model_name or ("llama" in model_name and "meta" not in model_name):
        return LiteLLM(model_engine=f"ollama/{model_name}", api_base="http://localhost:11434")
    else:
        return GenAIHubProvider(model_engine=model_name, endpoint=DummyEndpoint())


def eval_context_relevance(df, save_dir, eval_model="gemma3:27b", use_cot=True, max_score_val=3, start=0):
    if use_cot:
        assert max_score_val == 3
    provider = get_provider_from_model(eval_model)
    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons if use_cot else provider.context_relevance,
            name=f"{eval_model}_context_relevance{'_with_cot' if use_cot else ''}{f'_max_{max_score_val}' if max_score_val != 3 else ''}",
            max_score_val=max_score_val,
        )
        .on_input()
        .on(context_call.rets[:])
        .aggregate(np.mean)
    )
    return _eval(df, [f_context_relevance], save_dir, start=start)


def eval_groundedness_answerability(df, save_dir, eval_model="gemma3:27b", filter_trivial=True):
    provider = get_provider_from_model(eval_model)
    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons_consider_answerability,
            name=f"{eval_model}_groundedness_answerability{'_filter_trivial' if filter_trivial else ''}",
            groundedness_configs=GroundednessConfigs(use_sent_tokenize=True, filter_trivial_statements=filter_trivial),
        )
        .on(context_call.rets.collect())
        .on_output()
        .on_input()
    )
    return _eval(df, [f_groundedness], save_dir)


def eval_groundedness(df, save_dir, eval_model="gemma3:27b", filter_trivial=True, start=0):
    provider = get_provider_from_model(eval_model)
    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons,
            name=f"{eval_model}_groundedness{'_filter_trivial' if filter_trivial else ''}",
            groundedness_configs=GroundednessConfigs(use_sent_tokenize=True, filter_trivial_statements=filter_trivial),
        )
        .on(context_call.rets.collect())
        .on_output()
    )
    return _eval(df, [f_groundedness], save_dir, start=start)


def eval_answer_relevance(df, save_dir, eval_model="gemma3:27b", use_cot=True):
    provider = get_provider_from_model(eval_model)
    f_answer_relevance = (
        Feedback(
            provider.relevance_with_cot_reasons if use_cot else provider.relevance,
            name=f"{eval_model}_answer_relevance{'_with_cot' if use_cot else ''}",
        )
        .on_input()
        .on_output()
    )
    return _eval(df, [f_answer_relevance], save_dir)


if __name__ == "__main__":
    df_wikieval = pd.read_csv("../eval_wikieval/pred-wikieval-full-2025-05-12_20_33_35.csv")
    eval_groundedness(
        df_wikieval,
        "../eval_wikieval/",
        eval_model="anthropic--claude-3.7-sonnet",
        filter_trivial=True,
    )
