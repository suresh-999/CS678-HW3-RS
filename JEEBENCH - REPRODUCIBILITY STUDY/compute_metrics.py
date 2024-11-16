# Import necessary libraries
import numpy as np  # For numerical operations
import json         # For handling JSON data
import pandas as pd # For data manipulation and creating tables

# Define constants for question types
QUES_TYPES = ['MCQ', 'MCQ(multiple)', 'Integer', 'Numeric']

# Define a list of models to evaluate
models = [
    "Random",                    # Random guessing baseline
    "GPT3_normal",               # GPT-3 in normal mode
    "GPT3.5_normal",             # GPT-3.5 in normal mode
    "GPT4_normal",               # GPT-4 in normal mode
    "GPT4_CoT",                  # GPT-4 with Chain of Thought (CoT)
    'GPT4_CoT_self_refine',      # GPT-4 with CoT and self-refinement
    "GPT4_CoT+OneShot",          # GPT-4 with one-shot CoT
    "GPT4_CoT+SC@8"              # GPT-4 with CoT and self-critique (SC) at 8 responses
]

# Function to aggregate multiple answers into a single response
def get_aggregate(answers, question_type, single_threshold=None, multiple_threshold=None):
    """
    Aggregates multiple model outputs into a final response for evaluation.
    
    Parameters:
        answers (list): List of answers from the model.
        question_type (str): Type of question (e.g., MCQ, Integer).
        single_threshold (float): Threshold for single-option MCQ aggregation.
        multiple_threshold (float): Threshold for multiple-option MCQ aggregation.
    """
    if question_type in ['MCQ', 'MCQ(multiple)']:
        letter_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'None': 4}
        idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'None'}
        abcd = [0, 0, 0, 0, 0]  # Initialize counts for each option (A-D, None)

        # Count the occurrences of each option in the answers
        for ans in answers:
            if ans == 'None':
                abcd[letter_to_idx[ans]] += 1
            else:
                for c in ans:
                    abcd[letter_to_idx[c]] += 1

        # Process single-choice MCQ
        if question_type == 'MCQ':
            abcd = abcd[:-1]  # Remove "None"
            answer = idx_to_letter[np.argmax(abcd)]
            if single_threshold is not None:
                answer = answer if abcd[np.argmax(abcd)] / len(answers) >= single_threshold else "None"
        else:
            # Process multiple-choice MCQ
            if multiple_threshold is not None:
                options_selected = [idx_to_letter[x] for x in range(len(abcd))
                                    if abcd[x] >= len(answers) * multiple_threshold and idx_to_letter[x] != 'None']
            else:
                options_selected = [idx_to_letter[x] for x in range(len(abcd))
                                    if abcd[x] >= len(answers) / 2 and idx_to_letter[x] != 'None']
            answer = ''.join(sorted(options_selected)) if options_selected else "None"
    else:
        # For Integer and Numeric questions, choose the most frequent non-"None" answer
        answers = [a for a in answers if a != "None"]
        if not answers:
            return "None"
        unique, counts = np.unique(answers, return_counts=True)
        answer = unique[np.argmax(counts)]
    return answer

# Function to compute the score of a model response
def compute_score(gold, resp, question_type, year):
    """
    Computes the score of a response compared to the gold standard.
    
    Parameters:
        gold (str): The gold standard answer.
        resp (str): The model's response.
        question_type (str): Type of question.
        year (int): Year of the question (for reference).
    """
    assert question_type in QUES_TYPES
    if question_type == 'MCQ(multiple)':
        gold = set(c for c in 'ABCD' if c in gold)
        resp = set(c for c in 'ABCD' if c in resp)
        if resp == gold:
            return 1.0
        elif len(resp - gold) == 0:
            return 0.25 * len(resp)
        return 0.0  # Penalize incorrect options
    elif question_type == 'MCQ':
        return int(set(gold) == set(resp))
    else:
        if resp == "None":
            return 0.0
        return int(abs(float(gold) - float(resp)) <= 0.01)

# Function to construct a table of responses for evaluation
def construct_responses_table():
    """
    Constructs a table summarizing model responses and their corresponding scores.
    """
    responses = {}
    for model in models:
        if "Random" != model and "SC@" not in model:
            responses[model] = json.load(open(f"data/responses/{model}_responses/responses.json"))

    dataset = json.load(open('data/dataset.json'))
    extracts = {key: [] for key in ["Type", "Index", "Description", "Subject", "Gold"] + models}

    # Populate extracts with dataset details and responses
    for i, q in enumerate(dataset):
        extracts['Type'].append(q['type'])
        extracts['Index'].append(q['index'])
        extracts['Description'].append(q['description'])
        extracts['Subject'].append(q['subject'])
        extracts['Gold'].append(q['gold'])

        for model in models:
            if "Random" != model and "SC@" not in model:
                extracts[f"{model}"].append(responses[model][i].get('extract', "None"))

    pd.DataFrame(extracts).to_csv('results/extracts.csv', index=False)
    return pd.read_csv('results/extracts.csv', dtype=str)

# Compute scores for all responses
responses = construct_responses_table()
output = []

for i, response in responses.iterrows():
    out = {key: response[key] for key in ["Type", "Index", "Description", "Subject", "Gold"]}
    out["Random"] = 0.25 if response["Type"] == "MCQ" else 0.0
    for model in models:
        if model != "Random":
            resp = response.get(model, "None")
            out[model] = compute_score(out["Gold"], resp, out["Type"], out["Description"])
    output.append(out)

# Save scores to a CSV file
df = pd.DataFrame(output)
df.to_csv("results/scores.csv", index=False)

# Aggregate scores by different modes
modes = ['overall', 'type_wise', 'subject_wise']
for mode in modes:
    if mode == 'overall':
        grouped = df.agg({model: 'mean' for model in models})
    elif mode == 'type_wise':
        grouped = df.groupby('Type').mean()
    elif mode == 'subject_wise':
        grouped = df.groupby('Subject').mean()
    grouped.to_csv(f"results/aggregated_scores_{mode}.csv")
print("Done!")
