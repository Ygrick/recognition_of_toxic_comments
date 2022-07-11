import re


def lemmatize(doc):
    patterns = "[A-Za-z0-9!#$%&'()*+,/:;<=>?@[\]^_`{|}~—\"\-]+"
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    doc = doc.lower()
    for token in doc.split():
        if token:
            tokens.append(token)
    return tokens