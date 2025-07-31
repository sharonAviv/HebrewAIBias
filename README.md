# 1_data

## Dependencies

- pandas
- pyreadstat
- deep-translator
- sentence-transformers
- torch
- numpy
- scipy

## Usage

- **converted_questions.json**: Master JSON with all questions, answers, and distributions.
- **fix_json_format.py**: Converts raw survey JSONs to the standardized question format.
- **group_questions.py**: Groups questions by semantic similarity using embeddings.
- **populate_distributions.py**: Fills missing "distribution" fields in the master JSON using .sav survey files.
- **question_groups.json**: Output JSON listing groups of similar questions by their IDs.
- **topic_to_question.json**: Maps topics to question IDs.
- **translate_questions.py**: Translates questions and answers in a JSON file to Hebrew.
- **translated_questions.json**: Output JSON with translated questions and answers.
- **wd_compare.py**: Computes Wasserstein distance between two question sets, including refusal diagnostics.
- **wd_per_questions_example.json**: Example output of Wasserstein distance calculations.
