
# coding: utf-8

# In[35]:

from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from os import cpu_count
from pathlib import Path
import pickle
from pprint import pprint
import random

from tqdm import tqdm

import numpy as np

from nltk.corpus import framenet 

from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout

import tensorflow as tf
tf.python.control_flow_ops = tf # fix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

import spacy


print('loading spacy...')
nlp = spacy.load('en')

spacy_to_fn = {'ADJ': 'a',
               'ADV': 'adv',
               'CONJ': 'c',
               'DET': 'art',
               'INTJ': 'intj',
               'NOUN': 'n',
               'PROPN': 'n',
               'NUM': 'num',
               'VERB': 'v'}





def get_frames():
    lus = framenet.lus()
    print('num lus', len(lus))

    some_lu = random.choice(lus)
    print('Some LU:', some_lu.name, some_lu.POS, some_lu.frame.name)

    lus2frames = defaultdict(set)

    for lu in lus:
        lus2frames[lu.name].add(lu.frame.name)

    frames = chain.from_iterable(lus2frames.values())
    frames = sorted(set(frames))
    print('num frames', len(frames))


    # In[18]:

    mlb = MultiLabelBinarizer()
    mlb.fit(lus2frames.values())
    # lb = LabelBinarizer()
    # lb.fit(frames)

    return lus, lus2frames, frames, mlb


def get_lus2vecs(lus2frames):
    lus_lemmas = (lu.rsplit('.')[0] for lu in lus2frames)
    tokens = (nlp(lu_lemma)[0] for lu_lemma in lus_lemmas)

    tokens2vecs = {token.lower_: token.vector 
                   for token in tokens
                   if token.has_vector}
    return tokens2vecs



def get_corpora(fp):
    fp = Path(fp)
    for item in fp.iterdir():
        with item.open() as f:
            # return the lines
            yield from f


def prepare_data(corpora, lus2frames,
                 tokens2vecs=None,
                 new_token_threshold=5):

    if not tokens2vecs:
        tokens2vecs = {}
    tokens_with_frames = set()
    print('getting tokens from corpora')
    docs = nlp.pipe(corpora, batch_size=100,
                    n_threads=cpu_count())

    num_tokens = 0
    max_iter_no_update = 20000
    iter_no_update = 0

    for doc in tqdm(docs):
        these_tokens2vecs = {token.lower_: token.vector 
                            for token in doc if token.has_vector}

        tokens_pos_exist = (token for token in doc
                            if token.pos_ in spacy_to_fn
                            and token.has_vector)

        these_tokens_with_frames = {(token.lower_, token.lemma_, 
            spacy_to_fn[token.pos_]) 
            for token in tokens_pos_exist
            if '{}.{}'.format(token.lemma_, 
                              spacy_to_fn[token.pos_])
            in lus2frames}

        tokens2vecs.update(these_tokens2vecs)
        tokens_with_frames.update(these_tokens_with_frames)

        # break iteration when few new tokens are added
        num_new_tokens = len(tokens_with_frames) - num_tokens
        if num_new_tokens < new_token_threshold:
            if iter_no_update == max_iter_no_update:
                break
            else:
                iter_no_update += 1
        else:
            # reset the iteration update counter
            iter_no_update = 0

        num_tokens = len(tokens_with_frames)



    tokens_with_frames = sorted(tokens_with_frames)
    print('tokens with frames:', len(tokens_with_frames))
    assert tokens_with_frames
    return tokens_with_frames, tokens2vecs

# In[31]:

def make_training_data(tokens_with_frames, tokens2vecs, 
                       lus2frames, lb):

    # make matrices of vectors mapped to frames
    vector_size = 300

    features = []
    classes_ = []

    for token, token_lemma, fn_pos in tokens_with_frames:
        vec = tokens2vecs[token]

        token_frames = lus2frames['{}.{}'.format(token_lemma, fn_pos)]
        features.append(vec)
        classes_.append(token_frames)

    X = np.array(features)    
    y = lb.transform(classes_)


    print('X', X.shape)
    print('y', y.shape)

    assert X.shape[0] == y.shape[0]

    return X, y


def build_model(X, y, model_path='model.h5'):
    model = Sequential()
    
    l1 = Dense(768, input_dim=X.shape[1], activation='tanh')
    l2 = Dense(768, activation='tanh')
    l3 = Dense(y.shape[1], activation='sigmoid')
    
    model.add(l1)
    model.add(Dropout(0.2))
    model.add(l2)
    model.add(Dropout(0.2))
    model.add(l3)
    
    early_stopping = EarlyStopping(patience=5)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad', 
                  metrics=['categorical_accuracy'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    
    model.fit(X_train, y_train, batch_size=128, nb_epoch=1000,
              validation_data=(X_test, y_test),
              callbacks=[early_stopping])
    
    model.save(str(model_path))

    return model


def train_classifier(X, y):
    print('training the classifier')
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    classifier = RandomForestClassifier(n_estimators=30,
                                        verbose=2,
                                        n_jobs=-1)
    classifier.fit(X, y)
    train_score = classifier.score(X, y)
    test_score = classifier.score(X_test, y_test)

    print('train score: {:.2%}'.format(train_score))
    print('test score: {:.2%}'.format(test_score))

    return classifier


def get_frames_from_preds(preds, frames):
    for pred in preds:
        # max_prob = np.argmax(pred)
        pred_frames = set()
        for n, prob in enumerate(pred):
            if prob >= 0.1:
                pred_frames.add(frames[n])
        yield pred_frames

    
def get_frames_from_preds_rf(preds, frames):
    assert len(preds) == len(frames)
    for n, pred in enumerate(preds):
        pred = pred[0]
        if len(pred) == 2:
            if pred[1] > 0.1:
                yield frames[n]


def get_model(fp='model.h5'):
    lus, lus2frames, frames, lb = get_frames()

    if Path(fp).exists():
        model = load_model(str(fp))

    else:
        corpora = get_corpora('corpora')
        lus2vecs = get_lus2vecs(lus2frames)
        print('lus2vecs', len(lus2vecs))

        tokens_with_frames, tokens2vecs = prepare_data(corpora,
                                                       lus2frames,
                                                       lus2vecs)
        X, y = make_training_data(tokens_with_frames, tokens2vecs,
                                  lus2frames, lb)
        # vecs2frames = get_vecs2frames(lus2frames)
        # X, y = make_training_data(vecs2frames, lb)
        model = build_model(X, y)

    return model, frames


class AutoFrameNet:

    def __init__(self, model_path='model.h5'):
        self.model, self.frames = get_model(model_path)

    def get_tokens_frames(self, string):
        tokens = nlp(string)

        token_vecs = [token.vector for token in tokens
                      if token.pos_ in spacy_to_fn
                      and token.has_vector]

        tokens_frames = []

        if token_vecs:
            token_vecs = np.array(token_vecs)
            pred_probas = self.model.predict_proba(token_vecs)

            predicted_frames = get_frames_from_preds(pred_probas,
                                                     self.frames)
            predicted_frames = list(predicted_frames)

            for n, word_frames in enumerate(predicted_frames):
                if word_frames:
                    tokens_frames.append(word_frames)
                else:
                    token_vec = token_vecs[n]
                    token_vec = token_vec.reshape(1, -1)
                    pred_class = self.model.predict_classes(token_vec)[0]
                    token_frame = self.frames[pred_class]
                    predicted_frames.append({token_frame})

        return tokens_frames


def test_model(model_path):
    afn = AutoFrameNet(model_path)

    while True:
        string = input('> ')
        tokens_frames = afn.get_tokens_frames(string)
        if tokens_frames:
            print(tokens_frames)
        else:
            print('no suitable parts of speech or vectors')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-mp', '--model-path', type=Path,
                            default='model.h5', help='path to the model')
    args = arg_parser.parse_args()

    test_model(args.model_path)


# In[ ]:



