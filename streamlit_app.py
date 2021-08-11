import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from src.lgb_modeling import LGBM_NLP_Classifier

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_vis = st.beta_container()

def make_tsne_graph(ax, model, preds):

    idxs = np.random.choice(range(len(preds)), size=1000, replace=False)

    full_preds = np.concatenate((model.preds[idxs], preds))
    full_colors = np.concatenate((model.colors[idxs], np.array(['y'])))

    scatter = model.tsne.fit_transform(full_preds)
    xs = scatter[:, 0]
    ys = scatter[:, 1]

    ax.scatter(xs[idxs], ys[idxs], c=full_colors[idxs])
    ax.scatter(xs[-1:], ys[-1:], c='y', s=100)

    return ax

with header:
    st.title("Predicting FOIA request success")
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
                    
    lgb_model = joblib.load('models/my_lgbm_model.jl')

    X = np.array([[input_foia_body, agency_as_int]])
    preds = lgb_model.predict(X)
    preds_df = pd.DataFrame(data=preds,
                            index=['Your Request:' for i in range(len(preds))],
                            columns=['Completed', 'Redacted', 'Rejected'])

    st.header("Probability of your FOIA Request")
    st.write(preds_df)

    st.header("Visualization of model's prediction")

    fig, ax = plt.subplots()
    
    make_tsne_graph(ax, lgb_model, preds)
    
    st.write(fig)