import streamlit as st
import numpy as np
import pandas as pd
import joblib
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_vis = st.beta_container()

with header:
    st.title("Welcome to my capstone project!")
    st.text("Predicting success or failure of a Freedom of Information Act (FOIA) request.")

with dataset:
    st.header("MuckRock FOIA dataset")
    st.text("19002 FOIA requests with body, agency, and status label.")
    st.text("This dataset was collected using the MuckRock API.")

    st.image("images/tSNE/3D/tfidf_tsne_3d.png",
             caption='TF/IDF vectors of the dataset visualized with t-SNE')

with model_vis:
    sel_col, disp_col = st.beta_columns(2)
    agencies = {'FBI': 10,
                'CIA': 6,
                'NSA': 17,
                'DHS': 9,
                'FCC': 46,
                'ICE': 133,
                'FTC': 66,
                'DEA': 137,
                'State Dept': 14,
                'Other': 981}

    input_foia_body = sel_col.text_input('Enter your own FOIA request to get an estimated probability of success:',
                        '(Copy / paste your FOIA request here)')

    agency = sel_col.selectbox('Which agency are you filing with?',
                               options=agencies.keys())

    agency_as_int = agencies[agency]
                    
    lgb_model = joblib.load('../models/my_lgb_model.jl')

    X = np.array([[input_foia_body, agency_as_int]])
    preds = lgb_model.predict(X)

    st.write(preds)
    # sequence = tokenizer.texts_to_sequences(np.array([input_foia_body]))
    # word_index = tokenizer.word_index

    # # truncate or pad all the articles to the same length
    # sequence = [x[:250] for x in sequence]
    # input_data = pad_sequences(sequence, maxlen=250, padding='post', truncating='post')

    # raw_cnn_pred = cnn_model.predict(input_data)