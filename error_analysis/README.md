# Error Analysis

This directory contains the following files:
- [analysis_ret.ipynb](analysis_ret.ipynb) and [analysis_gen.ipynb](analysis_gen.ipynb) investigate the error classes of the different datasets and metrics
- [label_errors.py](label_errors.py) contains the code to label the errors using human judgement and with the LLM-based heuristic
- The eight CSV files in the [samples/](samples) subdirectory contain the LLM- and human-labeled samples of the error cases
- The eight CSV files in this directory contain the LLM-labeled error cases for each of the two datasets and four metric types
- [error-distribution.ipynb](error-distribution.ipynb) presents the results in terms of best-performing LLMs and the resulting estimated error class distribution