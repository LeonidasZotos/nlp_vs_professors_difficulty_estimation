"""This script performs direct inference with a language model (LLM), all items at once"""
import argparse
import pandas as pd
from google import genai
import regex as re


def get_llm_response(prompt, model):
    full_output = ""
    if model == 'gemini':
        print("Generating Response with Gemini...")
        client = genai.Client(api_key=open(
            '../tokens/GOOGLE_API_KEY.txt', 'r').read())
        # full_output = client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt).text
        print("Requesting Gemini...")
        full_output = client.models.generate_content(
            model="gemini-2.5-pro-preview-03-25", contents=prompt).text
        print("Response:", full_output)

    return full_output


def format_prompt(questions, correct_answer_keys, dataset):
    def format_question_set(questions, correct_answer_keys):
        formatted_questions = ""
        for i, question in enumerate(questions):
            if correct_answer_keys[i] == 'A':
                formatted_questions += f"({i}) Question: {question} A: True (correct) B: False\n"
            elif correct_answer_keys[i] == 'B':
                formatted_questions += f"({i})Question: {question} A: True B: False (correct)\n"
        return formatted_questions

    example_question = ""
    # Retrieve example question
    if dataset == 'aml':
        example_question = """
        If \( X \) is a \( k \times l \) sized matrix, \( X' \) its transpose, then \( X X' X \) is an \( l \times l \) sized matrix.
        A: True 
        B: False (correct)
        """
    elif dataset == 'nn':
        example_question = """
        (elementary math) The inner product \( \mathbf{{x}}' \mathbf{{y}} \) of two vectors gives the product of the lengths of the two vectors.
        A: True 
        B: False (correct)
        """
    # Generate prompt
    prompt = ""

    if dataset == 'nn':
        prompt = f"""Below are 59 questions from the Neural Networks Course, taught as an elective in the AI Bachelor's Programme in the University of Groningen. The correct answer is given. Estimate, from the examiner's perspective, what percentage of students will answer each question correctly. An example is presented below, where the percentage of students who selected the correct answer is provided. Think step by step, but always end your message with a list of 60 (including the 1st example) numbers in an array format ("[, , ,]" ) : The estimated proportion of students who will answer each question correctly. \n

        Question: {example_question}\n
        Selection Rate: 85%\n
        
        Below is the actual question set.\n
        
        Questions: {format_question_set(questions, correct_answer_keys)}\n
        Response:
        """
    elif dataset == 'aml':
        prompt = f"""Below are 53 questions from the Advanced Machine Learning Course, taught as an elective in the AI Bachelor's Programme in the University of Groningen. The correct answer is given. Estimate, from the examiner's perspective, what percentage of students will answer each question correctly. An example is presented below, where the percentage of students who selected the correct answer is provided. Think step by step, but always end your message with a list of 53 (including the 1st example) numbers in an array format ("[, , ,]" ) : The estimated proportion of students who will answer each question correctly. \n

        Question: {example_question}\n
        Selection Rate: 85%\n
        
        Below is the actual question set.\n
        
        Questions: {format_question_set(questions, correct_answer_keys)}\n
        Response:
        """
    return prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Direct inference/prompting with an LLM')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Dataset to use (folder name of the dataset, "aml" or "nn")', required=True)
    parser.add_argument('-m', '--model', type=str,
                        help='Model used for inference.', required=True)
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

    # Question, without options True/False
    questions = question_set['Question'].tolist()
    correct_answer_keys = question_set['Answer_Key'].tolist()  # A or B
    prompt = format_prompt(questions, correct_answer_keys, dataset)
    full_output = get_llm_response(prompt, model)
    print("Full Output", full_output)
    with open("results/" + dataset + "_" + model + "_LLM_single-shot_all_questions.txt", "w") as f:
        f.write(full_output)

    pattern = r'\[(.*?)\]'
    matches = re.findall(pattern, full_output)

    estimates = []
    for match in matches:
        numbers = match.split(',')
        for number in numbers:
            try:
                estimates.append(float(number))
            except ValueError:
                print("ValueError:", number)
    # if the estimates are 1 short of the number of questions, add 0.0 to the start. That happens when the model doesn't include the example question in the output.
    if len(estimates) == len(questions) - 1:
        estimates.insert(0, 0.0)
    estimates = [x / 100 for x in estimates]
    # Add estimates to the question set
    question_set[model + '_direct_estimate'] = estimates

    output_file_name = "results/" + dataset + "_" + \
        model + "_LLM_single-shot_all_questions_2-5" + ".csv"
    question_set.to_csv(output_file_name, index=False)
    print("Results saved to:", output_file_name)
