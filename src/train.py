from typing import Dict
import pandas as pd
import pycrfsuite
from user_args import parse_train_arguments
from utils.feature_extraction import doc_to_features, doc_to_classes
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

args = parse_train_arguments()


# ====================
def select_x_y_features(data):
    print('Features selection')
    # Generate features/tags
    X_train = [doc_to_features(doc) for _, doc in data['all_cleaned'].iteritems()]
    y_train = [doc_to_classes(doc) for _, doc in data['all_cleaned'].iteritems()]
    return X_train, y_train


# ====================
def read_train_data(training_data_location):
    print('Reading training data')
    return pd.read_csv(training_data_location)


# ====================
def hyperparameter_selection(
        c1: float = 1.0,
        c2: float = 1e-3,
        max_iterations: int = 75,
        possible_transitions: bool = True) -> Dict:
    return {
        'c1': c1,  # coefficient for L1 penalty
        'c2': c2,  # coefficient for L2 penalty
        'max_iterations': max_iterations,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': possible_transitions
    }


# ====================
def train(training_data_location, trained_model_path):
    print('Beginning train')

    data = read_train_data(training_data_location)
    x, y = select_x_y_features(data)

    trainer = pycrfsuite.Trainer(verbose=False)

    for x_seq, y_seq in zip(x, y):
        trainer.append(x_seq, y_seq)

    trainer.set_params(
        hyperparameter_selection()
    )

    print(trainer.params())

    trainer.train(trained_model_path)
    print('Training complete')


if __name__ == '__main__':
    train(args.training_data_location, args.trained_model_path)
