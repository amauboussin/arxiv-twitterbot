from collections import Counter
import json

from funcy import pluck
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.layers import (Activation, Conv1D, Dense, Dropout, Embedding,
                          Flatten, Input, MaxPooling1D, concatenate)
from keras.regularizers import L1L2
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.models import load_model
from sklearn.base import TransformerMixin
from sklearn.utils import class_weight
from spacy.tokens.doc import Doc


class ConvNet(object):
    """Convolution network for text classification
    Adapted from https://github.com/keon/keras-text-classification
    """

    def __init__(self, train, test, **model_options):
        """Create a conv net with keras
        Args:
            train: List of train examples
            test: List of test (validation) examples
        """
        embedding_size = model_options.get('embedding_size', 128)
        filter_sizes = model_options.get('filter_sizes', [2, 3, 4])
        n_filters = model_options.get('n_filters', 25)
        pool_size = model_options.get('pool_size', 4)
        hidden_dims = model_options.get('hidden_dims', 128)
        dropout_prob = model_options.get('dropout_prob', .5)
        conv_l2 = model_options.get('conv_l2', .05)
        fc_l2 = model_options.get('fc_l2', .05)
        balance_classes = model_options.get('balance_classes', False)

        self.train_labels = pluck('label', train)
        self.x_train, self.x_test = pluck('content', train), pluck('content', test)
        self.y_train, self.y_test = pluck('label', train), pluck('label', test)

        self.train_ids = pluck('id', train)
        self.test_ids = pluck('id', test)

        self.transform = DocToWordIndices().fit(self.x_train)
        self.x_train = self.transform.transform(self.x_train)
        self.x_test = self.transform.transform(self.x_test)

        self.vocab_size = np.max(self.x_train) + 1  # vocab and classes are 0 indexed
        self.n_labels = int(np.max(self.y_train)) + 1
        self.y_train, self.y_test = to_categorical(self.y_train), to_categorical(self.y_test)

        self.sequence_length = self.x_train.shape[1]
        self.n_labels = self.y_train.shape[1]
        self.balance_classes = balance_classes

        conv_input = Input(shape=(self.sequence_length, embedding_size))
        convs = []
        for filter_size in filter_sizes:
            conv = Conv1D(activation="relu", padding="valid",
                          strides=1, filters=n_filters, kernel_size=filter_size,
                          kernel_regularizer=L1L2(l1=0.0, l2=conv_l2))(conv_input)
            pool = MaxPooling1D(pool_size=pool_size)(conv)
            flatten = Flatten()(pool)
            convs.append(flatten)

        if len(filter_sizes) > 1:
            conv_output = concatenate(convs)
        else:
            conv_output = convs[0]

        conv_layer = Model(inputs=conv_input, outputs=conv_output)

        # main sequential model
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, embedding_size,
                                 input_length=self.sequence_length, weights=None))

        self.model.add(conv_layer)
        self.model.add(Dense(hidden_dims, kernel_regularizer=L1L2(l1=0.0, l2=fc_l2)))
        self.model.add(Dropout(dropout_prob))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.n_labels, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, batch_size, epochs, save_best_model_to_filepath=None):
        checkpoint = ModelCheckpoint(save_best_model_to_filepath,
                                     monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')

        weights = class_weight.compute_class_weight('balanced', np.unique(self.train_labels),
                                                    self.train_labels)
        weights[1] = weights[1] * 5
        # Fit the model
        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       class_weight=None if not self.balance_classes else weights,
                       callbacks=[checkpoint] if save_best_model_to_filepath is not None else [],
                       validation_data=[self.x_test, self.y_test])
        if save_best_model_to_filepath:
            self.model = load_model(save_best_model_to_filepath)
        return self.model

    def get_predict_proba(self):
        """Return a function that goes from docs to class probabilities"""
        def predict_proba(examples):
            x = self.transform.transform(examples)
            return self.model.predict(x)
        return predict_proba


def _get_tokens(doc_or_token_list, lowercase=True, remove_stopwords=False):
    """Return list of tokens from a spacy doc + associated metadata"""
    if type(doc_or_token_list) == Doc:
        return [t.text.lower() if lowercase else t.text
                for t in doc_or_token_list
                if not (remove_stopwords and t.is_stop)
                ]
    else:  # list of tokens
        return [token.lower() if lowercase else token
                for token in doc_or_token_list]


def load_doc_to_word_indices(filepath):
    data = json.load(open(filepath, 'r'))
    transform = DocToWordIndices(**data['params'])
    transform.seq_length = data['seq_length']
    transform.token_lookup = data['token_lookup']
    return transform


class DocToWordIndices(TransformerMixin):

    n_special_chars = 2
    padding_index = 0
    unk_index = 1

    def __init__(self, max_seq_length=None, vocab_size=None, case_sensitive=False,
                 pad_to_max_length=True, left_padding=0):
        """Create sklearn transformer that goes from spacy docs to a list of word indices
        Args:
            max_seq_length: If not None, truncate/pad sequences to make them all this length
            vocab_size: If not None, words outside the vocab_size most common words will be UNK
            case_sensitive: If True, different capitalization maps to different tokens
            pad_to_max_length: If True, zero pad sequences so they are all the same length

        """
        self.case_sensitive = case_sensitive
        self.vocab_size = vocab_size
        self.token_lookup = None
        self.pad_to_max_length = pad_to_max_length
        self.left_padding = left_padding

        self.max_seq_length = max_seq_length

    def fit(self, X, y=None):
        doc_tokens = [_get_tokens(doc) for doc in X]
        self.seq_length = self.max_seq_length or max(map(len, doc_tokens))
        token_counts = Counter([t for doc in doc_tokens for t in doc])
        n_tokens = self.vocab_size or sum(token_counts.itervalues())
        self.token_lookup = {token: i + self.n_special_chars
                             for i, (token, count) in enumerate(token_counts.most_common(n_tokens))}
        return self

    def _transform_doc(self, doc):
        """Transform a single spacy tokens into a list of indexed tokens"""
        doc_indices = [self.token_lookup.get(t, self.unk_index)
                       for t in _get_tokens(doc)]

        if self.pad_to_max_length and len(doc_indices) <= self.seq_length:
            right_padding = self.seq_length - len(doc_indices)
        else:
            right_padding = 0

        return np.pad(doc_indices[:self.seq_length],
                      (self.left_padding, right_padding),
                      mode='constant',
                      constant_values=self.padding_index).reshape(1, -1)

    def transform(self, X, y=None):
        return np.concatenate([self._transform_doc(doc) for doc in X], axis=0)

    def serialize(self, filepath):
        data_to_serialize = {
            'params': {
                'max_seq_length': self.max_seq_length,
                'vocab_size': self.vocab_size,
                'case_sensitive': self.case_sensitive,
                'pad_to_max_length': self.pad_to_max_length,
                'left_padding': self.left_padding
            },
            'seq_length': self.seq_length,
            'token_lookup': self.token_lookup
        }
        json.dump(data_to_serialize, open(filepath, 'w'))