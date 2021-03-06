import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from src.lgb_modeling import LGBM_NLP_Classifier

plt.style.use('ggplot')

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_vis = st.beta_container()

def make_tsne_graph(ax, model, preds):

    idxs = np.random.choice(range(len(model.preds)), size=1000, replace=False)

    full_preds = np.concatenate((model.preds[idxs], preds))
    full_colors = np.concatenate((model.colors[idxs], np.array(['b'])))

    scatter = model.tsne.fit_transform(full_preds)
    xs = scatter[:, 0]
    ys = scatter[:, 1]

    scatter = ax.scatter(xs[:-1], ys[:-1], c=full_colors[:-1], s=20, alpha=0.7)
    ax.scatter(xs[-1:], ys[-1:], c='yellow', s=150, marker='*', label='Your request')
    ax.set_title('t-SNE visualization of LightGBM model output')
    ax.set_xlabel('t-SNE feature 1')
    ax.set_ylabel('t-SNE feature 2')
    
    leg = ax.legend(*scatter.legend_elements(num=3))
    ax.add_artist(leg)

    return ax

with header:
    st.title("Predicting FOIA Request Success")
    st.text("Multiclass classification using deep learning models and NLP techniques")

with dataset:
    st.header("MuckRock FOIA dataset")
    st.text("Consists of 19002 FOIA requests with body, agency, and status label")
    st.text("This dataset was collected using the MuckRock API")

    st.image("images/tSNE/2D/tfidf_tsne_2d.png",
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

    # input_foia_body = sel_col.text_area('Enter your own FOIA request to get an estimated probability of success:',
    #                     '(Copy / paste your FOIA request here)')
    input_foia_body = st.text_area('Enter your own FOIA request to get an estimated probability of success:')

    agency = sel_col.selectbox('Which agency are you filing with?',
                               options=agencies.keys())

    agency_as_int = agencies[agency]
                    
    lgb_model = joblib.load('models/my_lgbm_model.jl')

    labels = ['Rejected', 'No Relevant Docs', 'Completed']

    X = np.array([[input_foia_body, agency_as_int]])
    preds = lgb_model.predict(X)
    preds_df = pd.DataFrame(data=preds,
                            index=['Your Request:' for i in range(len(preds))],
                            columns=labels)

    st.header("Your FOIA Request")
    pred_label = labels[preds.argmax()]

    if pred_label == 'No Relevant Docs':
        st.text("This model predicts your request will most likely return no relevant documents.")
    else:
        st.text(f"This model predicts your request will most likely be {labels[preds.argmax()].lower()}.")

    st.write(preds_df)

    st.header("Visualization of Model Predictions")

    fig, ax = plt.subplots()
    
    make_tsne_graph(ax, lgb_model, preds)
    
    st.write(fig)