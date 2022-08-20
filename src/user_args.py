import argparse


def parse_inference_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration for inference')
    parser.add_argument('--trained_model_path', default='./data/crf_restorer.pickle', type=str,)
    parser.add_argument('--input_file', default='./data/TED_TEST.csv', type=str,
                        help='list of documents to be restored')
    parser.add_argument('--input_file_target_column', default='col', type=str,
                        help='when supplying a CSV with a header, supply the target column to be read for inference')
    parser.add_argument('--output_file', default='./data/inference-results.csv', type=str,
                        help='list of document with restoration completed')
    args = parser.parse_args()
    return args


def parse_train_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration for training')
    parser.add_argument('--trained_model_path', default='./data/crf_restorer.pickle', type=str,
                        help='the final trained model is stored here')
    parser.add_argument('--training_data_location', default='./data/TED_TRAIN.csv', type=str,
                        help='list of documents used for training')
    args = parser.parse_args()
    return args
