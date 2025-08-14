import json
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from azure.ai.evaluation import evaluate
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from citation_evaluator import CitationEvaluator



def evaluate_alignment(alignment_data_path):
    """
    Evaluates the alignment of the custom evaluator
    Args:
        args (dict): A dictionary of arguments including input data path and other configurations.
    Returns:
        None
    """

    evaluate_results = evaluate(
        data=alignment_data_path,
        evaluators={
            "citation": CitationEvaluator(model_config_dict={
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
                "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "api_key": os.getenv("AZURE_OPENAI_API_KEY")
            })
        },
        evaluator_config={
            "citation": {
                "column_mapping": {
                    "question": "${data.question}",
                    "chatbot_answer": "${data.chatbot_answer}"
                }
            },
        },
    )

    eval_result = pd.DataFrame(evaluate_results["rows"])
    
    print(eval_result)

    evaluator_labels = normalize_labels(eval_result["outputs.citation.is_valid"])
    human_labels = normalize_labels(eval_result["inputs.human_label"])

    # Get misaligned rows
    misaligned_mask = [eval_label != human_label for eval_label, human_label in zip(evaluator_labels, human_labels)]
    misaligned_rows = eval_result[misaligned_mask]
    if len(misaligned_rows) > 0:
        for idx, reason in enumerate(misaligned_rows['outputs.citation.reason']):
            print(f"{idx+1}. {reason}")
    else:
        print("No misaligned rows")

    kappa = cohen_kappa_score(human_labels, evaluator_labels)

    print(f"**Cohen's Kappa: {kappa}**\n\n"
        "Interpreting Cohen’s Kappa:\n"
        "κ < 0.20: Poor agreement\n"
        "κ = 0.21−0.39: Fair agreement\n"
        "κ = 0.40−0.59: Moderate agreement\n"
        "κ = 0.60−0.79: Substantial agreement\n"
        "κ ≥ 0.80: Almost perfect agreement")


    # Calculate the confusion matrix
    cm = confusion_matrix(human_labels, evaluator_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.xlabel('Evaluator Labels')
    plt.ylabel('Human Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Should be moved to a utility module
def normalize_labels(labels):
    """
    Normalize labels to boolean values, handling various string representations
    """
    normalized = []
    for label in labels:
        if isinstance(label, str):
            # Convert to lowercase and strip whitespace
            label_str = label.lower().strip()
            # Handle various true representations
            if label_str in ['true', '1', 'yes', 'y']:
                normalized.append(True)
            # Handle various false representations
            elif label_str in ['false', '0', 'no', 'n']:
                normalized.append(False)
            else:
                # If it's an unexpected string, try to evaluate it as boolean
                try:
                    normalized.append(bool(eval(label_str.title())))
                except:
                    # Default to False if unable to parse
                    print(f"Warning: Unable to parse label '{label}', defaulting to False")
                    normalized.append(False)
        elif isinstance(label, bool):
            normalized.append(label)
        elif isinstance(label, (int, float)):
            normalized.append(bool(label))
        else:
            # Default to False for any other type
            print(f"Warning: Unexpected label type '{type(label)}' for value '{label}', defaulting to False")
            normalized.append(False)
    return normalized

if __name__ == "__main__":

    alignment_data_path = str(Path(__file__).parent / "alignment_data.jsonl")

    evaluate_alignment(alignment_data_path)