import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from constants import DICT_TRANSFORM_PATH, MODEL_PATH
from preprocessing import get_tokenized_list_of_dicts
from conv_net import ConvNet


def train_conv_net(epochs=15, weights_file=MODEL_PATH,
                   dictionary_file=DICT_TRANSFORM_PATH):
    data = get_tokenized_list_of_dicts()
    train, test = train_test_split(data, test_size=0.1, random_state=0)
    cnet = ConvNet(train, test, filter_sizes=[1, 2, 3, 4], n_filters=10,
                   dropout_prob=.7, pool_size=10, hidden_dims=12, embedding_size=64,
                   fc_l2=.05, conv_l2=0.0, balance_classes=True)

    batch_size = 64
    cnet.fit(batch_size, epochs, weights_file)

    predict_proba = cnet.get_predict_proba()
    probs = predict_proba([r['content'] for r in test])
    preds = (probs[:,0] < .5).astype(int)
    y_test = [row['label'] for row in test]

    print 'Mean prediction: {}'.format(np.mean(preds))
    print classification_report(y_test, preds)
    print confusion_matrix(y_test, preds)

    cnet.transform.serialize(dictionary_file)
