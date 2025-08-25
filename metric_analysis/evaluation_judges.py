import pandas as pd
from tqdm import tqdm

from pipeline.evaluation_ground_truth import get_llm_judgement, ref_free_prompt
from pipeline.evaluation_trulens import (
    eval_answer_relevance,
    eval_context_relevance,
    eval_groundedness,
    eval_groundedness_answerability,
)


def create_sample(file_path, group_by, unknown_only=False):
    df = pd.read_csv(file_path)
    if "rerank_model" in group_by:
        df = df[~df["rerank_model"].str.contains("mixedbread", na=False)]
    else:
        df = df[df["rerank_model"].isna()]  # Ignore rerank
    df = df[df["top_k"].isin([2 if "retrieval" in file_path else 6, 12])]  # Only take 2 top_k values
    df = (
        df[
            df["gold_answer"].str.contains("ontext", case=False, na=False)
            & df["gold_answer"].str.contains("information", case=False, na=False)
            & (df["relevant_text"] == "[]")
        ]
        if unknown_only
        else df[df["relevant_text"] != "[]"]
    )
    grouped = df.groupby(group_by)

    all_prompts = set.intersection(*[set(group["prompt"].unique()) for _, group in grouped])

    all_prompts = set(pd.Series(list(all_prompts)).sample(n=10, random_state=42))
    df_filtered = df[df["prompt"].isin(all_prompts)]
    print(f"Created {len(df_filtered)} samples")

    return df_filtered


def create_sample_clustered(file_path, group_by):
    df = pd.read_csv(file_path)

    # Filter rerank_model
    if "rerank_model" in group_by:
        df = df[~df["rerank_model"].str.contains("mixedbread", na=False)]
    else:
        df = df[df["rerank_model"].isna()]  # Ignore rerank

    # Filter top_k and non-empty relevant_text
    df = df[df["top_k"].isin([2 if "retrieval" in file_path else 6, 12])]

    grouped = df.groupby(group_by, dropna=False)
    result_frames = []

    for group_key, group_df in grouped:
        sampled_prompts = []

        for cluster_val in sorted(group_df["cluster"].unique()):
            cluster_prompts = group_df[group_df["cluster"] == cluster_val]["prompt"].unique()
            if len(cluster_prompts) >= 3:
                sampled = pd.Series(cluster_prompts).sample(n=3, random_state=42).tolist()
                sampled_prompts.extend(sampled)
            else:
                print(f"⚠️ Skipping cluster {cluster_val} in group {group_key} — not enough prompts.")

        if len(sampled_prompts) == 18:
            filtered_group = group_df[group_df["prompt"].isin(sampled_prompts)]
            result_frames.append(filtered_group)
        else:
            print(f"❌ Skipping group {group_key} — only got {len(sampled_prompts)} prompts.")

    final_df = pd.concat(result_frames, ignore_index=True)
    print(f"✅ Created {len(final_df)} samples from {len(result_frames)} groups.")

    return final_df


def fix_errors(file_path):
    judges = get_judges(file_path)

    if "ret" in file_path:
        eval_columns = ["context_relevance", "context_relevance_with_cot"]
    else:
        eval_columns = ["context_relevance_with_cot"]

    print(f"Reading file {file_path}")
    df = pd.read_csv(file_path)
    orig_len = len(df)

    for judge in judges:
        # Create a list of all evaluation columns for this judge
        judge_columns = []

        if "ret" in file_path:
            # For retrieval evaluations
            for col in eval_columns:
                judge_columns.append(f"{judge}_{col}")
        else:
            # For generation evaluations
            if f"{judge}_judgement" in df.columns:
                judge_columns.append(f"{judge}_judgement")
            elif f"{judge}_judgement_ref_free" in df.columns:
                judge_columns.append(f"{judge}_judgement_ref_free")

            for col in eval_columns:
                judge_columns.append(f"{judge}_{col}")

        # Check each column
        for col in judge_columns:
            if col not in df.columns:
                print(f"Column {col} not found in dataframe, skipping")
                continue

            nan_mask = df[col].isna() | (df[col] == -1)
            if nan_mask.sum() == 0:
                print(f"No NaN values found for {col}, skipping")
                continue

            print(f"Found {nan_mask.sum()} rows with NaN values for {col}")
            nan_rows = df[nan_mask].copy()

            nan_rows["__orig_index__"] = nan_rows.index

            if col.endswith("_judgement") or col.endswith("_judgement_ref_free") or f"{col}_meta" not in df.columns:
                nan_rows.drop(columns=[col], inplace=True)
            else:
                nan_rows.drop(columns=[col, f"{col}_meta"], inplace=True)

            # Evaluate
            result_df = None
            if col.endswith("_judgement"):
                tqdm.pandas(desc=f"Fixing Ground Truth Eval for {judge}")
                nan_rows[col] = nan_rows.progress_apply(
                    lambda row: get_llm_judgement(row, model_name=judge, is_ollama="gemma" in judge),
                    axis=1,
                )
            elif col.endswith("_judgement_ref_free"):
                tqdm.pandas(desc=f"Fixing Ground Truth Eval for {judge}")
                nan_rows[col] = nan_rows.progress_apply(
                    lambda row: get_llm_judgement(
                        row,
                        model_name=judge,
                        is_ollama="gemma" in judge,
                        system_prompt=ref_free_prompt,
                    ),
                    axis=1,
                )
            elif "context_relevance_with_cot" in col:
                result_df = eval_context_relevance(nan_rows, save_dir="tmp/", eval_model=judge, use_cot=True)
            elif "context_relevance" in col and "with_cot" not in col:
                result_df = eval_context_relevance(nan_rows, save_dir="tmp/", eval_model=judge, use_cot=False)
            elif "answer_relevance_with_cot" in col:
                result_df = eval_answer_relevance(nan_rows, save_dir="tmp/", eval_model=judge, use_cot=True)
            elif "answer_relevance" in col and "with_cot" not in col:
                result_df = eval_answer_relevance(nan_rows, save_dir="tmp/", eval_model=judge, use_cot=False)
            elif "groundedness_filter_trivial" in col:
                result_df = eval_groundedness(nan_rows, save_dir="tmp/", eval_model=judge, filter_trivial=True)
            elif "groundedness" in col and "filter_trivial" not in col:
                result_df = eval_groundedness(nan_rows, save_dir="tmp/", eval_model=judge, filter_trivial=False)

            # Assign the evaluated values back using the saved index
            if result_df is not None:
                # Make sure result_df[col] exists and matches length
                result_df["__orig_index__"] = nan_rows["__orig_index__"].values
                for idx, val in zip(result_df["__orig_index__"], result_df[col]):
                    df.at[idx, col] = val
            else:
                # If already evaluated in-place (e.g., with apply), just assign back
                for idx, val in zip(nan_rows["__orig_index__"], nan_rows[col]):
                    df.at[idx, col] = val

            output_path = file_path.replace(".csv", f"-fixed.csv")
            df.to_csv(output_path, index=False)
            print(f"Fixed dataframe saved to {output_path}")

    new_len = len(df)
    if new_len != orig_len:
        print(f"WARNING: Dataframe length changed from {orig_len} to {new_len}")


def get_judges(path):
    if "wikieval" in path:
        return [
            "gpt-4o-mini",
            "gpt-4o",
            "mistralai--mistral-large-instruct",
            "anthropic--claude-3.7-sonnet",
            "meta--llama3.1-70b-instruct",
        ]
    else:
        return ["mistralai--mistral-large-instruct", "meta--llama3.1-70b-instruct"]


def evaluate_judges_gen(generation_path):
    judges = get_judges(generation_path)

    sample_func = create_sample if "wikieval" in generation_path else create_sample_clustered
    df_gen = sample_func(generation_path, ["llm", "chunk_size", "top_k"])
    print(f"Loaded {len(df_gen)} samples")
    eval_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_path = f"eval-judges-gen-{'wikieval' if 'wikieval' in generation_path else 'sov'}-{eval_time}.csv"

    for judge in judges:
        tqdm.pandas(desc=f"Reference-free Eval for {judge}")
        df_gen[f"{judge}_judgement_ref_free"] = df_gen.progress_apply(
            lambda row: get_llm_judgement(
                row,
                model_name=judge,
                is_ollama="gemma" in judge,
                system_prompt=ref_free_prompt,
            ),
            axis=1,
        )
        df_gen.to_csv(output_path, index=False)

        for b in [True, False]:
            print(f"Running groundedness_answerability for {judge} with filter_trivial={b}")
            df_gen = eval_groundedness_answerability(df_gen, save_dir="tmp/", eval_model=judge, filter_trivial=b)
            df_gen.to_csv(output_path, index=False)
            print(f"Running groundedness for {judge} with filter_trivial={b}")
            df_gen = eval_groundedness(df_gen, save_dir="tmp/", eval_model=judge, filter_trivial=b)
            df_gen.to_csv(output_path, index=False)
            print(f"Running answer relevance for {judge} with use_cot={b}")
            df_gen = eval_answer_relevance(df_gen, save_dir="tmp/", eval_model=judge, use_cot=b)
            df_gen.to_csv(output_path, index=False)


def evaluate_judges_ret(retrieval_path):
    judges = get_judges(retrieval_path)

    sample_func = create_sample if "wikieval" in retrieval_path else create_sample_clustered
    df_ret = sample_func(retrieval_path, ["embed_model", "chunk_size", "top_k", "rerank_model"])
    print(f"Loaded {len(df_ret)} samples")
    eval_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_path = f"eval-judges-ret-{'wikieval' if 'wikieval' in retrieval_path else 'sov'}-{eval_time}.csv"

    for judge in judges:
        for b in [(False, 3), (False, 10), (True, 3)]:
            print(f"Running context relevance for {judge} with use_cot={b}")
            use_cot, max_score_val = b
            df_ret = eval_context_relevance(
                df_ret,
                save_dir="tmp/",
                eval_model=judge,
                use_cot=use_cot,
                max_score_val=max_score_val,
            )
            df_ret.to_csv(output_path, index=False)


def add_ref_free_eval(generation_path):
    judges = get_judges(generation_path)
    df = pd.read_csv(generation_path)
    print(f"Loaded {len(df)} samples")
    for judge in judges:
        tqdm.pandas(desc=f"Reference-free eval for {judge}")
        df[f"{judge}_judgement_ref_free"] = df.progress_apply(
            lambda row: get_llm_judgement(row, model_name=judge, system_prompt=ref_free_prompt),
            axis=1,
        )
        df.to_csv(generation_path, index=False)


if __name__ == "__main__":
    evaluate_judges_ret("../eval_wikieval/pred-wikieval-retrieval-2025-04-27_04_11_05.csv")
    evaluate_judges_gen("../eval_wikieval/pred-wikieval-retrieval-2025-04-27_04_11_05.csv")
