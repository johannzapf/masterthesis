# Datasets

This directory contains the following files:
- [preprocessing.ipynb](preprocessing.ipynb) contains the code to extract the sovanta dataset from Document Chat's database and to bring it into the correct form
- [sovanta-dataset.csv](sovanta-dataset.csv) and [wikieval-dataset.csv](wikieval-dataset.csv) are the two datasets
- [sovanta-dataset-documents.zip](sovanta-dataset-documents.zip) contains the documents (files and saved external system states) that are the basis for the sovanta dataset and that are used by [../pipeline/embed-sovanta.py](../pipeline/embed-sovanta.py) to create the vector index
- [dataset/sovanta-dataset-open-examples.csv](dataset/sovanta-dataset-open-examples.csv) is the unencrypted example dataset that does not contain sensntive information.
- [clustering.py](clustering.py) contains the code to cluster the queries in the sovanta dataset using LLMs
- [check_relevant_texts.py](check_relevant_texts.py) checks for all relevant text values in the sovanta dataset if the given content is actually part of the underlying context to prevent mismatches
- [validate_datasets.ipynb](validate_datasets.ipynb) contains code to validate the labels for both datasets. It uses the four *val-[...].py* datasets. These datasets represent a single pipeline run with a large context size