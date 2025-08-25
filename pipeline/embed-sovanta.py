import json
import os
import pickle

import pandas as pd
from pandas import DataFrame

from app.controllers.context_controller import ContextController
from app.controllers.document_controller import DocumentController
from app.controllers.user_controller import UserController
from app.hyperparameters import hyperparameters
from app.model.pydantic_model import DocumentSource
from app.model.response_model import PCreateContext
from app.settings import module_registry
from app.utils.db import apply_alembic_migrations, db_session
from app.utils.helpers import update_key_value
from masterarbeit.utils.eval_plugin import EvalPlugin

CHUNK_SIZES = [64, 128, 256, 512]
EMBEDDING_MODELS = ["intfloat/multilingual-e5-large", "BAAI/bge-m3"]
USER_ID = 1

confluence_sites = {
    "CONFLUENCE-SDS": [21562387],
    "CONFLUENCE-ITPUB": [67961112],
    "CONFLUENCE-XS-QUAL": [2264334366, 30572957],
    "CONFLUENCE-SOSAP": [2644869171],
    "CONFLUENCE-SD": [21540440],
    "CONFLUENCE-SIFP": [2631074191],
}


def create_user():
    UserController.get_or_create_user_by_name("TEST", "TEST", "TEST", True)


def embed_plugin(name: str, source: DocumentSource, context_name: str, user_id: int) -> int:
    print(f"Creating {name} plugin")
    context_id = ContextController.create_context(PCreateContext(contextName=context_name), user_id=user_id).contextID
    assert context_id is not None
    plg = EvalPlugin(plugin_id=0, context_id=context_id, metadata={}, source=source)
    with open(f"../documents/{name}.pkl", "rb") as handle:  # This references the files contained in the sovanta-dataset-documents.zip file
        objects = pickle.load(handle)
    plg.embed_objects(objects, ignore_plugin_id=True)
    return context_id


def embed_documents(prompts: DataFrame):
    if_plugin = 0
    sp_plugin = 0
    cf_plugin = {
        "CONFLUENCE-SDS": 0,
        "CONFLUENCE-ITPUB": 0,
        "CONFLUENCE-XS-QUAL": 0,
        "CONFLUENCE-SOSAP": 0,
        "CONFLUENCE-SD": 0,
        "CONFLUENCE-SIFP": 0,
    }

    # Check presence of all documents before starting embedding
    for i, prompt in prompts.iterrows():
        source = prompt["source"]
        if source in ("SHAREPOINT", "INNOVATION_FACTORY"):
            continue
        if confluence_sites.get(source) is not None:
            continue
        for document in json.loads(prompt["documents"]):
            if not os.path.exists(f"../documents/file/{document}"):
                raise ValueError(f"documents/file/{document} does not exist")

    embed_model = module_registry.embedding.onnx_model_name
    ctx_column_name = f"eval_context_id_{hyperparameters.chunking_size}_{embed_model}"
    print(f"ctx_column_name: {ctx_column_name}")

    for i, prompt in prompts.iterrows():
        source = prompt["source"]
        if source == "INNOVATION_FACTORY":
            if if_plugin == 0:
                if_plugin = embed_plugin(
                    name=source,
                    source=DocumentSource.INNOVATION_FACTORY,
                    context_name=source,
                    user_id=USER_ID,
                )
            prompts.loc[i, ctx_column_name] = if_plugin
            continue

        if source == "SHAREPOINT":
            if sp_plugin == 0:
                sp_plugin = embed_plugin(
                    name=source,
                    source=DocumentSource.SHAREPOINT,
                    context_name=source,
                    user_id=USER_ID,
                )
            prompts.loc[i, ctx_column_name] = sp_plugin
            continue

        if "CONFLUENCE" in source:
            if cf_plugin[source] == 0:
                cf_plugin[source] = embed_plugin(
                    name=source,
                    source=DocumentSource.CONFLUENCE,
                    context_name=source,
                    user_id=USER_ID,
                )
            prompts.loc[i, ctx_column_name] = cf_plugin[source]
            continue

        context_id = ContextController.create_context(PCreateContext(contextName=str(i)), user_id=USER_ID).contextID
        prompts.loc[i, ctx_column_name] = context_id

        for document in json.loads(prompt["documents"]):
            with open(f"../documents/file/{document}", "rb") as f:
                DocumentController.upload_document(
                    context_id=context_id,
                    file_body=f.read(),
                    file_name=document,
                    source=DocumentSource.from_file_extension(f".{source.lower()}"),
                    user_id=USER_ID,
                )


def embed_all():
    prompts = pd.read_csv("../dataset/prompts-target-sov-v5-pre.csv")
    for embed_model in EMBEDDING_MODELS:
        with db_session() as sess:
            update_key_value(
                sess,
                "embed",
                {
                    "EMBED_CLASS": "app.modules.embed.ONNXEmbedding",
                    "EMBED_ONNXEmbedding_onnx_model_name": embed_model,
                },
            )
        for chunk_size in CHUNK_SIZES:
            with open("../augmentation-state.txt", "r") as f:
                if f.readline() != "CONTINUE":
                    print(f"Stopping before iteration {embed_model} / {chunk_size}")
                    return
            hyperparameters.chunking_size = chunk_size
            module_registry.clear()
            embed_documents(prompts)
            prompts.to_csv("../dataset/prompts-target-sov-v5.csv", index=False)


if __name__ == "__main__":
    hyperparameters.is_eval = True
    apply_alembic_migrations()
    create_user()
    embed_all()
