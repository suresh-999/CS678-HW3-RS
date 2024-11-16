import json
import random

# Paths to input and output datasets
dataset_path = "data/dataset.json"  # Original dataset path
output_path = "data/dataset_with_errors.json"  # Path to save the perturbed dataset

def introduce_grammatical_mistakes(text):
    """
    Function to introduce grammatical mistakes into text.
    Adds noise like misspellings, extra spaces, or word order changes.
    """
    words = text.split()
    error_types = ["misspelling", "extra_spaces", "grammar_error"]

    for i in range(len(words)):
        error_type = random.choice(error_types)
        
        if error_type == "misspelling" and len(words[i]) > 3:
            # Randomly replace a character in the word
            char_idx = random.randint(0, len(words[i]) - 1)
            words[i] = words[i][:char_idx] + random.choice("abcdefghijklmnopqrstuvwxyz") + words[i][char_idx+1:]
        
        elif error_type == "extra_spaces":
            # Add extra spaces before the word
            words[i] = " " * random.randint(1, 3) + words[i]
        
        elif error_type == "grammar_error":
            # Randomly swap two adjacent words
            if i < len(words) - 1:
                words[i], words[i+1] = words[i+1], words[i]

    return " ".join(words)

def add_errors_to_dataset(input_path, output_path):
    """
    Adds grammatical errors to the dataset and saves the result.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    perturbed_data = []
    for item in data:
        # Apply grammatical mistakes to each question
        perturbed_question = introduce_grammatical_mistakes(item["question"])
        perturbed_item = item.copy()
        perturbed_item["question"] = perturbed_question
        perturbed_data.append(perturbed_item)
    
    # Save perturbed dataset
    with open(output_path, 'w') as f:
        json.dump(perturbed_data, f, indent=4)
    print(f"Dataset with grammatical mistakes saved to {output_path}")

if __name__ == "__main__":
    add_errors_to_dataset(dataset_path, output_path)
