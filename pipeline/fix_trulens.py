import pandas as pd

from evaluation_trulens import (
    eval_context_relevance,
    eval_answer_relevance,
    eval_groundedness,
)


def fix_errors(file_path, eval_column: str, eval_model: str):
    print(f"Reading file {file_path}")
    df = pd.read_csv(file_path)
    orig_len = len(df)

    if eval_column not in df.columns:
        print(f"Column {eval_column} not found in dataframe")
        return

    nan_mask = df[eval_column].isna() | (df[eval_column] == -1)
    if nan_mask.sum() == 0:
        print(f"No NaN values found for {eval_column}")
        return

    print(f"Found {nan_mask.sum()} rows with invalid values for {eval_column}")
    nan_rows = df[nan_mask].copy()

    nan_rows["__orig_index__"] = nan_rows.index
    nan_rows.drop(columns=[eval_column, f"{eval_column}_meta"], inplace=True)

    # Evaluate
    if "context_relevance_with_cot" in eval_column:
        result_df = eval_context_relevance(nan_rows, save_dir="tmp/", eval_model=eval_model, use_cot=True)
    elif "context_relevance" in eval_column and "with_cot" not in eval_column:
        result_df = eval_context_relevance(nan_rows, save_dir="tmp/", eval_model=eval_model, use_cot=False)
    elif "answer_relevance_with_cot" in eval_column:
        result_df = eval_answer_relevance(nan_rows, save_dir="tmp/", eval_model=eval_model, use_cot=True)
    elif "answer_relevance" in eval_column and "with_cot" not in eval_column:
        result_df = eval_answer_relevance(nan_rows, save_dir="tmp/", eval_model=eval_model, use_cot=False)
    elif "groundedness_filter_trivial" in eval_column:
        result_df = eval_groundedness(nan_rows, save_dir="tmp/", eval_model=eval_model, filter_trivial=True)
    elif "groundedness" in eval_column and "filter_trivial" not in eval_column:
        result_df = eval_groundedness(nan_rows, save_dir="tmp/", eval_model=eval_model, filter_trivial=False)
    else:
        print("No column found that is implemented for fixing errors")
        return

    result_df["__orig_index__"] = nan_rows["__orig_index__"].values
    for idx, val in zip(result_df["__orig_index__"], result_df[eval_column]):
        df.at[idx, eval_column] = val

    nan_mask = df[eval_column].isna() | (df[eval_column] == -1)
    print(f"Fixed dataframe has {nan_mask.sum()} rows with invalid values")

    output_path = file_path.replace(".csv", f"-fixed.csv")
    df.to_csv(output_path, index=False)
    print(f"Fixed dataframe saved to {output_path}")

    new_len = len(df)
    if new_len != orig_len:
        print(f"WARNING: Dataframe length changed from {orig_len} to {new_len}")


if __name__ == "__main__":
    fix_errors(
        "../eval_sovanta/eval-full-judgement-ref-free-cr-ar-gr-cr.csv",
        eval_column="meta--llama3.1-70b-instruct_groundedness",
        eval_model="meta--llama3.1-70b-instruct",
    )
