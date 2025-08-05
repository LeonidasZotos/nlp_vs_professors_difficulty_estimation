"""This script trains and evaluates various regression models to predict item difficulty. Based on: This is based on: https://github.com/LeonidasZotos/Are-You-Doubtful-Oh-It-Might-Be-Difficult-Then/"""
# General Packages
import pandas as pd
import numpy as np
import multiprocessing
import argparse
from scipy.stats import spearmanr

# Sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error


DATASET = ''
MODEL_NAMES = ''
HF_TOKEN = open('../tokens/HF_TOKEN.txt', 'r').read()
TEXT_COLUMNS_TO_EMBED = ['question_with_options']
TARGET_LABEL_COL_NAME = 'Correct_Answer_Rate'
TFIDF_THRESHOLD = 0.0007  # best found: 0.0007, higher->fewer features

# Number of cores to use for sklearn's n_jobs parameter, whenever possible
NUM_OF_CORES_TO_USE = multiprocessing.cpu_count()

GLOBAL_ALL_RESULTS = {}  # Store all results here


def retrieve_models_uncertainties_col_names(metric_names):
    uncertainty_feature_column_names = []
    for metric in metric_names:
        for model in MODEL_NAMES:
            uncertainty_feature_column_names.append(f'{metric}_{model}')
    return uncertainty_feature_column_names


def load_train_test():
    def convert_cols_to_str(df, cols_names):
        """Convert columns to string type."""
        for col in cols_names:
            df[col] = df[col].apply(str)
        return df

    path_prefix = "../data/" + DATASET + "/preprocessed/combined_results_"
    train = convert_cols_to_str(pd.read_csv(
        path_prefix + 'train_set.csv', index_col=0), TEXT_COLUMNS_TO_EMBED)
    test = convert_cols_to_str(pd.read_csv(
        path_prefix + 'test_set.csv', index_col=0), TEXT_COLUMNS_TO_EMBED)

    return train, test


def get_tf_idf_features(train, test):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Use 1-grams and 2-grams.
        # Ignore terms that appear in less than 0.1% of the documents.
        min_df=0.001,
        # Ignore terms that appear in more than 75% of documents.
        max_df=0.75,
        max_features=100,  # Use only the top 100 most frequent words.
        stop_words='english'
    )

    text_train = vectorizer.fit_transform(
        train['question_with_options']).toarray()
    text_test = vectorizer.transform(test['question_with_options']).toarray()

    text_train = pd.DataFrame(
        text_train,
        columns=['"' + w + '"' for w in vectorizer.get_feature_names_out()],
        index=train.index
    )

    text_test = pd.DataFrame(
        text_test,
        columns=['"' + w + '"' for w in vectorizer.get_feature_names_out()],
        index=test.index
    )

    # Adjust threshold as needed, the lower the threshold, the more features will be selected.
    selector = VarianceThreshold(threshold=TFIDF_THRESHOLD)
    text_train_selected = selector.fit_transform(text_train)
    text_train_selected = pd.DataFrame(text_train,
                                       columns=text_train.columns[selector.get_support(
                                       )],
                                       index=text_train.index)

    # Apply to test set
    text_test_selected = selector.transform(text_test)
    text_test_selected = pd.DataFrame(text_test_selected,
                                      columns=text_train.columns[selector.get_support(
                                      )],
                                      index=text_test.index)

    tf_idf_text_train = text_train_selected
    tf_idf_text_test = text_test_selected

    return tf_idf_text_train, tf_idf_text_test


def train_test_models_kfold(all_questions, n_splits=5, train_set_size=0.8):
    def get_features(train_set, test_set, use_tf_idf, use_uncertainties):
        uncertainty_feature_columns = retrieve_models_uncertainties_col_names(
            ['first_token_probability', 'order_probability'])
        train_set = train_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)
        tf_idf_text_train, tf_idf_text_test = None, None

        tf_idf_text_train, tf_idf_text_test = get_tf_idf_features(
            train_set, test_set)
        ensemble_cols_train = pd.concat(
            [train_set[col] for col in uncertainty_feature_columns], axis=1)
        ensemble_cols_test = pd.concat(
            [test_set[col] for col in uncertainty_feature_columns], axis=1)

        features_train, features_test = None, None
        target_train, target_test = None, None
        if use_tf_idf and use_uncertainties:
            features_train = pd.concat(
                [ensemble_cols_train, tf_idf_text_train], axis=1)
            features_test = pd.concat(
                [ensemble_cols_test, tf_idf_text_test], axis=1)
        elif use_tf_idf:
            features_train = tf_idf_text_train
            features_test = tf_idf_text_test
        elif use_uncertainties:
            features_train = ensemble_cols_train
            features_test = ensemble_cols_test

        target_train, target_test = train_set[TARGET_LABEL_COL_NAME], test_set[TARGET_LABEL_COL_NAME]
        return features_train, features_test, target_train, target_test

    def train_fit_and_get_rmse(features_train, features_test, target_train, target_test, regression_model):
        param_grid_ridge = {
            'alpha': [100.0],
            'fit_intercept': [True],
            'solver': ['sparse_cg']
        }
        param_grid_random_forest = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        param_grid_svr = {
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.1, 0.2, 0.5],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        # Define the model to test
        if regression_model == RandomForestRegressor:
            param_grid = param_grid_random_forest
            model_to_test = RandomForestRegressor()
        elif regression_model == SVR:
            param_grid = param_grid_svr
            model_to_test = SVR()
        elif regression_model == Ridge:
            param_grid = param_grid_ridge
            model_to_test = Ridge()
        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=model_to_test,
            param_grid=param_grid,
            cv=n_splits,
            scoring='neg_root_mean_squared_error',
            n_jobs=NUM_OF_CORES_TO_USE,
            refit=True,
            verbose=1
        )

        grid_search.fit(features_train, target_train)
        best_params_found = grid_search.best_params_
        print(f"Best parameters found via CV: {best_params_found}")

        # Final Model Evaluation on the Test set
        best_model = grid_search.best_estimator_

        final_predictions = best_model.predict(features_test)
        rmse = root_mean_squared_error(target_test, final_predictions)
        spearmans_rho = spearmanr(target_test, final_predictions).correlation
        mean_error = np.mean(final_predictions - target_test)
        ####################################################################
        # This produces the csv needed for the plot_estimates.py script.
        test_set_ids = test_set['id'].values
        predictions_df = pd.DataFrame({
            'id': test_set_ids,
            'estimate': final_predictions
        })
        predictions_df['golden_labels'] = target_test.values
        predictions_df.to_csv(f"results/{DATASET}_best_SVM.csv", index=False)
        ####################################################################

        return rmse, mean_error, spearmans_rho

    all_questions = all_questions.reset_index(drop=False)
    train_set, test_set = train_test_split(
        all_questions,
        train_size=train_set_size,
        test_size=1 - train_set_size,
        random_state=42,
        shuffle=True
    )
    test_ids = test_set['id'].values
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    rmse_per_setup = {}
    # Baseline
    # Features_train/features_test don't matter for the Dummy Regressor
    features_train, features_test = train_set.drop(
        columns=[TARGET_LABEL_COL_NAME]), test_set.drop(columns=[TARGET_LABEL_COL_NAME])
    target_train, target_test = train_set[TARGET_LABEL_COL_NAME], test_set[TARGET_LABEL_COL_NAME]
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(features_train, target_train)
    base_predictions = baseline.predict(features_test)
    base_rmse = root_mean_squared_error(target_test, base_predictions)
    spearmans_rho = spearmanr(target_test, base_predictions).correlation
    rmse_per_setup['baseline'] = base_rmse, 0, spearmans_rho

    for regression_model in [RandomForestRegressor, SVR, Ridge]:
        for use_tf_idf in [False, True]:
            for use_uncertainties in [False, True]:
                if not use_tf_idf and not use_uncertainties:
                    continue
                setup_name = f"{regression_model.__name__}_tf_idf_{use_tf_idf}_uncertainties_{use_uncertainties}"
                if use_tf_idf and use_uncertainties:
                    print("Using TF-IDF and uncertainties")
                    features_train, features_test, target_train, target_test = get_features(
                        train_set, test_set, use_tf_idf=True, use_uncertainties=True)
                elif use_tf_idf:
                    print("Using TF-IDF only")
                    features_train, features_test, target_train, target_test = get_features(
                        train_set, test_set, use_tf_idf=True, use_uncertainties=False)
                elif use_uncertainties:
                    print("Using uncertainties only")
                    features_train, features_test, target_train, target_test = get_features(
                        train_set, test_set, use_tf_idf=False, use_uncertainties=True)
                rmse_per_setup[setup_name] = train_fit_and_get_rmse(
                    features_train, features_test, target_train, target_test, regression_model)

    print("RMSE and spearman's rho per setup:")
    for key, value in rmse_per_setup.items():
        print(
            f"{key}: RMSE = {value[0]:.3f}, ME = {value[1]:.3f}, Spearman's rho = {value[2]:.3f}")

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate item difficulty using TF_IDF scores and other features')
    parser.add_argument('-d', '--dataset', type=str,
                        help='Dataset to use (folder name of the dataset)', required=True)
    parser.add_argument('-m', '--model_names', type=str, nargs='+',
                        help='List of model names to use for their uncertainties', required=True)
    parser.add_argument('-t', '--target_label', type=str,
                        help='Target label/column with difficulty measures', required=False, default='Correct_Answer_Rate')

    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    args = parse_args()

    globals()["DATASET"] = args['dataset']
    globals()["MODEL_NAMES"] = args['model_names']
    globals()["TARGET_LABEL_COL_NAME"] = args['target_label']

    train, test = load_train_test()

    # Merge train and test, but keep the id column
    all_questions = pd.concat([train, test], axis=0)
    train_test_models_kfold(all_questions, n_splits=5)
