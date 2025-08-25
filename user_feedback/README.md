# User Feedback

This directory contains the following files:
- [llm_feedback_2025-08-25.csv](llm_feedback_2025-08-25.csv) is an export from the feedback table in Document Chat that contains the user votes
- [feedback.ipynb](feedback.ipynb) contains the code to sample the models for side-by-side-feedback as well as the code that calculates Elo and win rate for the models. The code is taken from the code base of Document Chat and is adapted to run with the CSV file instead of against the database.