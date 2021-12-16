import re
import string


def remove_stop_words(text, stop_words):
    text_cleaned = list()
    stops = list()
    for word in text.lower().split():
        if word not in stop_words:
            text_cleaned.append(word)
        else:
            stops.append(word)
    return ' '.join(text_cleaned)


def text_process(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = text.replace('–', '').replace('--', '')
    text = ' '.join(i for i in text.split() if len(i) >= 3)
    return text


def clean_text(text):
    # punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    # single
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    # sev spaces
    text = re.sub(r'\s+', ' ', text)
    return text
