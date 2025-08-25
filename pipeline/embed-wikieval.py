import pandas as pd
from nltk import sent_tokenize
from tqdm import tqdm

from app.controllers.context_controller import ContextController
from app.controllers.document_controller import DocumentController
from app.controllers.user_controller import UserController
from app.hyperparameters import hyperparameters
from app.model.pydantic_model import DocumentSource
from app.model.response_model import PCreateContext
from app.settings import module_registry
from app.utils.db import apply_alembic_migrations
from app.utils.helpers import update_env_vars

from pipeline.prediction import USER_ID

CHUNK_SIZES = [64, 128, 256, 512]
EMBEDDING_MODELS = ["BAAI/bge-m3", "intfloat/multilingual-e5-large"]


def create_user():
    UserController.get_or_create_user_by_name("TEST", "TEST", "TEST", True)


def generate_dataset():
    prompts = pd.read_parquet(
        "hf://datasets/explodinggradients/WikiEval/data/train-00000-of-00001-385c01e94624e9b7.parquet"
    )
    results = []

    for idx, prompt in tqdm(prompts.iterrows(), total=len(prompts)):
        ctx_small = prompt["context_v1"][0]
        ctx_big = prompt["context_v2"][0]

        chunks = sent_tokenize(ctx_small)
        results.append(
            [
                prompt["question"].replace("Question: ", ""),
                prompt["answer"].replace("Answer: ", ""),
                prompt["poor_answer"],
                prompt["ungrounded_answer"],
                chunks,
                ctx_big,
                prompt["source"],
            ]
        )

    res = pd.DataFrame(
        results,
        columns=[
            "prompt",
            "gold_answer",
            "poor_answer",
            "ungrounded_answer",
            "relevant_text",
            "context",
            "wiki_page",
        ],
    )
    res.to_csv("data/wikieval-dataset.csv", index=False)


def embed_documents(prompts: pd.DataFrame):
    embed_model = module_registry.embedding.onnx_model_name
    ctx_column_name = f"eval_context_id_{hyperparameters.chunking_size}_{embed_model}"
    print(f"ctx_column_name: {ctx_column_name}")

    step_name = f"{hyperparameters.chunking_size}_{embed_model}".replace("/", "-")

    context_id = ContextController.create_context(PCreateContext(contextName=step_name), user_id=USER_ID).contextID

    context = ""

    for i, prompt in prompts.iterrows():
        prompts.loc[i, ctx_column_name] = context_id
        context += prompt["context"] + "\n\n"

    DocumentController.upload_document(
        context_id=context_id,
        file_body=context.encode("utf-8"),
        file_name=f"{step_name}.txt",
        source=DocumentSource.TEXT,
        user_id=USER_ID,
    )


def embed_all():
    prompts = pd.read_csv("../dataset/wikieval-dataset.csv")
    with tqdm(total=len(EMBEDDING_MODELS) * len(CHUNK_SIZES)) as pbar:
        for embed_model in EMBEDDING_MODELS:
            with update_env_vars(new_env_vars={"EMBED_ONNXEmbedding_onnx_model_name": embed_model}):
                for chunk_size in CHUNK_SIZES:
                    with open("../augmentation-state.txt", "r") as f:
                        if f.readline() != "CONTINUE":
                            print(f"Stopping before iteration {embed_model} / {chunk_size}")
                            return
                    hyperparameters.chunking_size = chunk_size
                    module_registry.clear()
                    embed_documents(prompts)
                    prompts.to_csv("data/prompts-target-wikieval.csv", index=False)
                    pbar.update(1)


if __name__ == "__main__":
    hyperparameters.is_eval = True
    apply_alembic_migrations()
    create_user()
    embed_all()
