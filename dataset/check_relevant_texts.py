import ast
import json
import pickle

import pandas as pd

from app.controllers.search_controller import SearchController
from utils.utils import cleanup_string, get_node_score


def check():
    sources = {
        "INNOVATION_FACTORY": "",
        "SHAREPOINT": "",
        "CONFLUENCE-ITPUB": "",
        "CONFLUENCE-SDS": "",
        "CONFLUENCE-XS-QUAL": "",
        "CONFLUENCE-SOSAP": "",
        "CONFLUENCE-SD": "",
        "CONFLUENCE-SIFP": "",
    }
    for source in sources.keys():
        with open(f"../documents/{source}.pkl", "rb") as handle:
            objects = pickle.load(handle)
            sources[source] = cleanup_string("\n".join([n.text for o in objects for n in o.nodes]))

    dataset = pd.read_csv("sovanta-dataset.csv")
    found = 0
    not_found = 0
    for _, row in dataset.iterrows():
        source = row["source"]
        rel_text = ast.literal_eval(row["relevant_text"].replace("\n", ""))
        if source in sources:
            y = sources[source]
        else:
            y = ""
            for doc in json.loads(row["documents"]):
                with open(f"../documents/file/{doc}", "rb") as f:
                    y += SearchController.extract_text(f.read(), doc).text
        for t in rel_text:
            score = get_node_score(y=cleanup_string(y), ground_truth=cleanup_string(t))
            if score != 1:
                print(
                    f"Prompt: {row['prompt']}\nRelevant Text: {t}\nSource: {source}\nDocuments: {row['documents']}\nScore: {score}\n"
                )
                not_found += 1
            else:
                found += 1

    print("Found: ", found)
    print("Not Found: ", not_found)


if __name__ == "__main__":
    check()
