from keras.models import load_model
import pandas as pd

from constants import DICT_TRANSFORM_PATH, MODEL_PATH
from conv_net import load_doc_to_word_indices
from preprocessing import arxiv_df_to_list_of_dicts, parse_content_serial


def add_predictions_to_date(paper_df, index):
    """Predict whether or not the papers published on a given day will be tweeted"""
    papers_to_predict = paper_df.loc[index]
    model = load_conv_net()

    data = arxiv_df_to_list_of_dicts(papers_to_predict)
    data = parse_content_serial(data)

    paper_df.loc[index, 'prediction'] = model([r['content'] for r in data])[:, 1]
    return paper_df


def load_conv_net():
    """Return a function from data frame row to probability miles would tweet it"""
    model = load_model(MODEL_PATH)
    dict_transform = load_doc_to_word_indices(DICT_TRANSFORM_PATH)

    def predict_proba(docs):
        x = dict_transform.transform(docs)
        return model.predict(x)
    return predict_proba
