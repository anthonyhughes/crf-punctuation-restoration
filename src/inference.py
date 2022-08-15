import pycrfsuite
from constants import TRAINED_MODEL_PATH

from utils.feature_extraction import doc_to_features


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
def read_trained_model():
    """Restore a trained model"""
    print('Reading model')
    tagger = pycrfsuite.Tagger()
    tagger.open(TRAINED_MODEL_PATH)
    return tagger


# ====================
def infer(model, input: str):
    print('Beginning Inference')
    prediction = restore(input, model)
    print(prediction)


if __name__ == '__main__':
    model = read_trained_model()
    infer(model,
          'the anger in me against corruption made me to make a big career change last year becoming a full time '
          'practicing lawyer my experiences over the last 18 months as a lawyer has seeded in me a new entrepreneurial '
          'idea which i believe is indeed worth spreading so i share it with all of you here today though the idea '
          'itself is getting crystallized and im still writing up the business plan of course it helps that fear '
          'of public failure diminishes'
          )
