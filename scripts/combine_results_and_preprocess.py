"""This script combines the uncertainties of different models into one dataframe."""
import pandas as pd
import argparse


MAIN_DF_NAME = "Qwen2_5-14b-chat"
MODEL_NAMES = ['phi3_5-chat', 'Llama3_2-3b-chat', 'Qwen2_5-3b-chat', 'Llama3_1-8b-chat',
               'Qwen2_5-14b-chat', 'Qwen2_5-32b-chat', 'Yi-34b-chat', 'Llama3_1-70b-chat', 'Qwen2_5-72b-chat']

HF_TOKEN = open('../tokens/HF_TOKEN.txt', 'r').read()


def combine_uncertainties(df, dataset_name, split_name):
    """Combines the uncertainties of the different models into one dataframe"""
    for cols in ["first_token_probability", "order_probability", "first_token_probability_selected_choice", "order_probability_selected_choice", "full_answer", "model_is_correct"]:
        df = df.rename(columns={cols: cols + "_" + MAIN_DF_NAME})

    for extra_model_df_name in MODEL_NAMES:
        df_extra = pd.read_csv("../data/" + dataset_name + "/with_uncertainty/" +
                               extra_model_df_name + "_" + split_name + "_set.csv")
        df["first_token_probability" + "_" +
            extra_model_df_name] = df_extra["first_token_probability"]
        df["order_probability" + "_" +
            extra_model_df_name] = df_extra["order_probability"]
        df["first_token_probability_selected_choice" + "_" +
            extra_model_df_name] = df_extra["first_token_probability_selected_choice"]
        df["order_probability_selected_choice" + "_" +
            extra_model_df_name] = df_extra["order_probability_selected_choice"]
        df["model_is_correct" + "_" +
            extra_model_df_name] = df_extra["model_is_correct"]

    return df


def create_question_with_options_string(df):
    """Creates a new column with the question and options only"""
    choice_columns = ['Answer_' + chr(ord('A') + i)
                      for i in range(10)]
    choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for row in df.iterrows():
        choices = ""
        for i in range(10):
            if pd.notna(row[1][choice_columns[i]]):
                choice_text = row[1][choice_columns[i]]
                if choice_text:
                    choices += choice_letters[i] + \
                        ") " + str(choice_text) + "\n"
        df.at[row[0], 'question_with_options'] = row[1]['Question'] + '\n' + choices

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine the uncertainties of the different models into one dataframe.')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Dataset to use (folder name of the dataset)', required=True)
    parser.add_argument('-m', '--model_names', type=str, nargs='+',
                        help='List of model names to use (default is all)', default=MODEL_NAMES)

    return dict(vars(parser.parse_args()))


# Main function
if __name__ == "__main__":
    args = parse_args()

    globals()["MODEL_NAMES"] = args['model_names']
    globals()["MAIN_DF_NAME"] = MODEL_NAMES[0]

    dataset = args['dataset']

    for split in ['train', 'test']:
        print("Processing " + dataset + " " + split)
        input_file_name = "../data/" + dataset + "/with_uncertainty/" + \
            MAIN_DF_NAME + "_" + split + "_set.csv"
        output_file_name = "../data/" + dataset + \
            "/preprocessed/combined_results_" + split + "_set.csv"

        combined_df = combine_uncertainties(
            pd.read_csv(input_file_name), dataset, split)
        combined_df = create_question_with_options_string(combined_df)

        combined_df.to_csv(output_file_name, index=False)
