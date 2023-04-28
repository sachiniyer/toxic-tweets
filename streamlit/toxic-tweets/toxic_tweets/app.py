import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

st.title('Sentiment Analysis with Streamlit')

speech = ""
with open("tweet.txt") as file:
    speech = "".join(line.rstrip() for line in file)

data = st.text_area(label="Text for Sentiment Analysis", value=speech)

models = ["sachiniyer/tweet_toxicity",
          "distilbert-base-uncased-finetuned-sst-2-english",
          "Ghost1/bert-base-uncased-finetuned_for_sentiment_analysis1-sst2",
          "Seethal/sentiment_analysis_generic_dataset",
          "sbcBI/sentiment_analysis_model",
          "juliensimon/reviews-sentiment-analysis"]

model_name = st.selectbox(
    'Which model do you want to use',
    models)


labels = ["toxic", "severe toxic", "obscene", "threat", "insult", "identity hate"]

def score(item):
    return item['score']

def get_tokens(data, model):
    tokenizer = AutoTokenizer.from_pretrained("sachiniyer/tweet_toxicity")
    tokens = tokenizer(data, return_tensors="pt")
    return tokens

def get_out(tokens, model):
    output = model(**tokens)
    return output

def get_perc(output):
    return torch.sigmoid(output.logits).detach().numpy()[0]

def get_dict(percs, data):
    sorted_indices = np.argsort(percs)[-2:]
    row = {"text": data,
           "label 1": labels[sorted_indices[1]],
           "perc 1": str(round(percs[sorted_indices[1]], 3)),
           "label 2": labels[sorted_indices[0]],
           "perc 2": str(round(percs[sorted_indices[0]], 3))}
    return row

def get(data, model):
    tokens = get_tokens(data, model)
    output = get_out(tokens, model)
    percs = get_perc(output)
    d = get_dict(percs, data)
    return pd.DataFrame([d])

if st.button('Run model'):
    if model_name == "sachiniyer/tweet_toxicity":
        model = AutoModelForSequenceClassification.from_pretrained("sachiniyer/tweet_toxicity")
        d = get(data, model)
        st.table(d)
    else:
        generator = pipeline(model=model_name)
        st.markdown(generator(model_name))
