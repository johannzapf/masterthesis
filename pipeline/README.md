# Pipeline

This directory contains the following files:
- [embed-sovanta.py](embed-sovanta.py) contains the code to create the vector index for the sovanta dataset. Running this code requires access to the /documents directory that is stored as the encrypted [../dataset/sovanta-dataset-documents.zip](../dataset/sovanta-dataset-documents.zip) in this repository.
- [embed-wikieval.py](embed-wikieval.py) contains the code to create the vector index for the WikiEval dataset
- [prediction.py](prediction.py) contains the code to run the RAG pipeline on the datasets
- [evaluation_ground_truth.py](evaluation_ground_truth.py) contains the code to compute recall and LLM judgement on the prediction datasets
- [evaluation_legacy.py](evaluation_legacy.py) contains the code to compute BLEU, ROUGE and BERTScore on the prediction datasets
- [evaluation_trulens.py](evaluation_trulens.py) contains the code to compute the RAG Triad metrics on the prediction datasets
- [fix_trulens.py](fix_trulens.py) contains the code to rerun any failed RAG Triad metrics
- [timing_analysis.ipynb](timing_analysis.ipynb) contains the analysis of timings and tokens presented in sections 4.1.5 and 4.2.8 of the thesis