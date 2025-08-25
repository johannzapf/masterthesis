# Human Judgement

This directory contains the following files:
- [human_judgement.py](human_judgement.py) contains the code to create the sample datasets and label them with human and LLM judgement
- [human-label-sovanta.csv](human-label-sovanta.csv) and [human-label-wikieval.csv](human-label-wikieval.csv) are the respective human-labeled dataset samples
- [eval-llm-judge-sovanta.csv](eval-llm-judge-sovanta.csv) and [eval-llm-judge-wikieval.csv](eval-llm-judge-wikieval.csv) are the same samples with all variants of the LLM-based metrics computed
- [compare_llm_judge_alignment.ipynb](compare_llm_judge_alignment.ipynb) presents the correlation results fo the various metrics with human judgement