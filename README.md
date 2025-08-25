# Master's Thesis - Evaluating RAG Pipelines with Real-World User Data

The repository is organized in the following way:
- [dataset/](dataset) contains the two datasets (WikiEval & sovanta), and the code for data extraction, validation and preprocessing
- [pipeline/](pipeline) contains the code to embed the documents from the datasets to create the vector index, the code to run predictions based on this index and the code to compute the various metrics
- [human_judgement/](human_judgement) contains the code to label the datasets, the labeled datasets and the correlation analysis for the metrics
- [metric_analysis/](metric_analysis) contains the code to run the sample-based grid search for the various metrics and the respective results
- [eval_sovanta/](eval_sovanta) and [eval_wikieval/](eval_wikieval) contain the respective prediction datasets for generation and retrieval as well as the final datasets with the computed metrics
- [error_analysis/](error_analysis) contains the code for human and LLM-based error labeling, the resulting data and its visualization
- [user_feedback/](user_feedback) contains the data from the user preference evaluation and the resulting model leaderboard

Note: All Code that requires imports from module *app* is not executable in this repository, as this code is part of the Document Chat productive application.

## Encryption
Since the sovanta dataset contains partially sensitive information from a productive database, all files that contain its data are encrypted in this repository.

The required encryption key to decrypt the files can be provided on request. [utils/encryption.py](utils/encryption.py) contains the code to decrypt the files.

To get a sense of the dataset without access to the encryption key, [dataset/sovanta-dataset-open-examples.csv](dataset/sovanta-dataset-open-examples.csv) contains a reduced, unencrypted version of the dataset with example prompts that do not contain sensitive information.
