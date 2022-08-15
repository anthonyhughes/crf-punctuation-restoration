import nltk
from constants import START_OF_DOC, END_OF_DOC


# ====================
def doc_to_classes(doc):
    return [get_word_classes(word) for word in doc.split()]


# ====================
def get_word_classes(word):
    # Upper, lower, or title
    if word.isupper():
        selected_classes = 'u'
    elif word.istitle():
        selected_classes = 't'
    else:
        selected_classes = 'l'

    # Comma, period, or nothing
    if word.endswith(','):
        selected_classes = selected_classes + 'c'
    elif word.endswith('.'):
        selected_classes = selected_classes + 'p'
    else:
        selected_classes = selected_classes + 'n'

    return selected_classes


# ====================
def lowercase_and_strip_punctuation(word: str):
    return ''.join([c.lower() for c in word if c.isalpha() or c.isnumeric()])


# ====================
def doc_to_features(doc: str) -> list:
    """Get features for a given document (e.g. a TED Talk transcript)"""

    all_features = []
    words_and_pos_tags = nltk.pos_tag(doc.split())
    words = [lowercase_and_strip_punctuation(word) for word, _ in words_and_pos_tags]
    pos_tags = [pos_tag for _, pos_tag in words_and_pos_tags]

    for i in range(len(words)):

        features = [
            'bias',
            'word=' + words[i],
            'pos_tag=' + pos_tags[i]
        ]

        if i > 0:
            features.extend([
                '-1:word=' + words[i-1],
                '-1:pos_tag=' + pos_tags[i-1]
            ])
        else:
            features.append(START_OF_DOC)
        if i > 1:
            features.extend([
                '-2:word=' + words[i-2],
                '-2:pos_tag=' + pos_tags[i-2]
            ])

        if i < len(words)-1:
            features.extend([
                '+1:word=' + words[i+1],
                '+1:pos_tag=' + pos_tags[i+1]
            ])
        else:
            features.append(END_OF_DOC)

        if i < len(words)-2:
            features.extend([
                '+2:word=' + words[i+2],
                '+2:pos_tag=' + pos_tags[i+2]
            ])

        all_features.append(features)

    return all_features
