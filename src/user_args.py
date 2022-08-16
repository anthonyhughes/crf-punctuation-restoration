import argparse


def parse_inference_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration for inference')
    parser.add_argument('--trained_model_path', default='./data/crf_restorer.pickle', type=str,)
    parser.add_argument('--input_file', default='./data/TED_TEST.csv', type=str,
                        help='list of documents to be restored')
    parser.add_argument('--output_file', default='./data/inference-results.csv', type=str,
                        help='list of document with restoration completed')
    args = parser.parse_args()
    return args


def parse_train_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration for training')
    return {}
