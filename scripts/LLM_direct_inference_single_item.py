"""This script performs direct inference with a language model (LLM), one item at a time."""
import argparse
import pandas as pd
from google import genai


def get_llm_response(prompt, model):
    response = ""
    full_output = ""
    if model == 'gemini':
        client = genai.Client(api_key=open(
            '../tokens/GOOGLE_API_KEY.txt', 'r').read())
        full_output = None
        while full_output == None:
            full_output = client.models.generate_content(
                model="gemini-2.0-flash-001", contents=prompt).text
            # full_output = client.models.generate_content(model="gemini-2.5-pro-preview-03-25", contents=prompt).text
            if full_output == None:
                print("Full output is None, retrying...")
        print("Full Response:", full_output)
        response = full_output[-3:]
        # if it's a percentage, we convert it to a float
        if response[-1] == "%":
            try:
                response = int(response[:-1]) / 100.0  # Convert to float 0-1
            except ValueError:
                print("Error converting percentage to float:", response)

    else:
        print("Model not recognized")
        return None

    return response, full_output


def format_prompt(question, correct_answer_key, dataset):
    example_question = ""

    # Retrieve example question
    if dataset == 'nn':
        example_question = """
        If \( X \) is a \( k \times l \) sized matrix, \( X' \) its transpose, then \( X X' X \) is an \( l \times l \) sized matrix.
        A: True 
        B: False (correct)
        """
    elif dataset == 'aml':
        example_question = """
        (elementary math) The inner product \( \mathbf{{x}}' \mathbf{{y}} \) of two vectors gives the product of the lengths of the two vectors.
        A: True 
        B: False (correct)
        """
    #  Mark correct answer in actual question
    if correct_answer_key == 'A':
        question = question + "A: True (correct)\n B: False"
    elif correct_answer_key == 'B':
        question = question + "A: True\n B: False (correct)"

    # Generate prompt
    if dataset == 'nn':
        prompt = f"""Below is a question from the Neural Networks Course, taught as an elective in the AI Bachelor's Programme in the University of Groningen. The correct answer is given. Estimate, from the examiner's perspective, what percentage of students will answer the question correctly. An example is presented below, where the percentage of students who selected the correct answer is provided. Think step by step, but always end your message with a percentage: The estimated proportion of students who will answer the question correctly. \n

        Question: {example_question}\n
        Selection Rate: 85%\n
        
        Below  is the actual question.\n
        
        Question: {question}\n
        """
    elif dataset == 'aml':
        prompt = f"""Below is a question from the Advaned Machine Learning Course, taught as an elective in the AI Bachelor's Programme in the University of Groningen. The correct answer is given. Estimate, from the examiner's perspective, what percentage of students will answer the question correctly. An example is presented below, where the percentage of students who selected the correct answer is provided. Think step by step, but always end your message with a percentage: The estimated proportion of students who will answer the question correctly. \n

        Question: {example_question}\n
        Selection Rate: 85%\n
        
        Below  is the actual question.\n
        
        Question: {question}\n
        """
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Direct inference/prompting with an LLM')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Dataset to use (folder name of the dataset, "aml" or "nn")', required=True)
    parser.add_argument('-m', '--model', type=str,
                        help='Model used for inference', required=True)
    parser.add_argument('-t', '--test_mode', action="store_true",
                        help='Test mode only uses 3 questions', default=False)

    return dict(vars(parser.parse_args()))


# Main function
if __name__ == "__main__":
    args = parse_args()

    dataset = args['dataset']
    model = args['model']

    question_set = pd.read_csv("../data/" + dataset + "/raw/all.csv")
    if args['test_mode']:
        question_set = question_set.head(3)

    estimates = []
    full_outputs = []
    for index, row in list(question_set.iterrows()):
        question = row['Question']  # Question, without options True/False
        correct_answer_key = str(row['Answer_Key'])  # A or B
        prompt = format_prompt(question, correct_answer_key, dataset)
        response, full_output = get_llm_response(prompt, model)
        estimates.append(response)
        full_outputs.append(full_output)
        print("--------Next question--------")

    # Add estimates to the question set
    question_set[model + '_direct_estimate'] = estimates
    question_set[model + '_full_output'] = full_outputs

    output_file_name = "results/" + dataset + "_" + \
        model + "_LLM_single-shot_2-0" + ".csv"
    question_set.to_csv(output_file_name, index=False)
    print("Results saved to:", output_file_name)
