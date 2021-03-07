import argparse
import os
import joblib

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


def model_fn(model_dir):
    """Deserialized and return fitted model
    """
    clf = joblib.load(os.path.join(model_dir, "rdf.joblib"))
    return clf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters for randomforest classifier
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--max-leaf-nodes', type=int, default= 30)
    parser.add_argument('--min-samples-split', type=int, default= 12)
    parser.add_argument('--min-samples-leaf', type=int, default= 12)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    train_data = pd.concat(raw_data)

    # labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_X = train_data.iloc[:, 1:]

    # tuning hyper parameter using randomsearch with specific random seed
    seed = args.seed
    max_depth = args.max_depth
    max_leaf_nodes = args.max_leaf_nodes
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf

    # specify parameters and distributions to sample from
    distributions = {"max_depth": [max_depth, None],
                  "max_leaf_nodes": randint(2, max_leaf_nodes),
                  "min_samples_split": randint(2, min_samples_split),
                  "min_samples_leaf": randint(2, min_samples_leaf),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    clf = RandomizedSearchCV(RandomForestClassifier(), distributions, random_state=seed)
    clf = clf.fit(train_X, train_y)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(clf.best_estimator_, os.path.join(args.model_dir, "rdf.joblib"))