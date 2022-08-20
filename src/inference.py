import pycrfsuite
import pandas as pd
from user_args import parse_inference_arguments
from utils.feature_extraction import doc_to_features
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

args = parse_inference_arguments()


# ====================
def generate_predictions(doc: str, model: pycrfsuite.Tagger):
    """Generate the predictions for a given a document"""
    features = doc_to_features(doc)
    words = doc.split()
    return words, model.tag(features)


# ====================
def restore(doc: str, tagger: pycrfsuite.Tagger) -> str:
    """Restore capitalisation and punctuation to a document"""

    words, tags = generate_predictions(doc, tagger)

    output_words = []

    # Convert from words and tags back to continuous text
    for word, tag in list(zip(words, tags)):
        # Casing
        if 't' in tag:
            output_words.append(word.title())
        elif 'l' in tag:
            output_words.append(word)
        elif 'u' in tag:
            output_words.append(word.upper())

        # Punctuations
        if 'c' in tag:
            output_words.append(',')
        elif 'p' in tag:
            output_words.append('.')
        output_words.append(' ')

    return ''.join(output_words)


# ====================
def read_trained_model(trained_model_path: str) -> pycrfsuite.Tagger:
    """Restore a trained model"""
    print('Reading model')
    tagger = pycrfsuite.Tagger()
    tagger.open(trained_model_path)
    return tagger


# ====================
def infer(model: pycrfsuite.Tagger, input_file: str, output_file: str):
    print('Beginning Inference')
    input_df = pd.read_csv(input_file)
    output_df = pd.DataFrame(columns=['results'])
    for i, j in input_df.iterrows():
        to_be_inferred = j['lower']
        prediction = restore(to_be_inferred, model)
        output_df = pd.concat(
            [
                output_df,
                pd.DataFrame(({'results': prediction}), index=[f'{i}'])
            ],
            ignore_index=False)
    output_df.to_csv(output_file)
    print('Inference Completed')


if __name__ == '__main__':
    model = read_trained_model(args.trained_model_path)
    infer(model, args.input_file, args.output_file)
