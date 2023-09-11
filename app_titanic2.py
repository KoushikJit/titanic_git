import streamlit as st
import pickle

import pandas as pd

st.write("""
    ## Hello!!!!
""")

pipe=pickle.load(open("titanic.pkl", 'rb'))

uploaded_file= st.sidebar.file_uploader(label="Upload File")
if uploaded_file is not None:
    X_New= pd.read_csv(uploaded_file)

preds=pd.DataFrame(pipe.predict(X_New), columns=["prediction"])
pred_proba=pd.DataFrame(pipe.predict_proba(X_New))
pred_proba= pred_proba.max(axis=1).rename("confidence")

df_output= pd.concat([X_New,pred_proba,preds], axis=1)
st.dataframe(df_output, hide_index=True)