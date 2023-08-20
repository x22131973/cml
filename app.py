import streamlit as st
import pandas as pd
from helper import *
df = pd.read_csv("default_sample.csv")
scaler = import_scaler()
encoder = import_encoder()
model = import_model()

st.title("House Price Prediction")

map_dict = {1: 'Fa', 2:'TA', 3:'Gd', 4:'Ex'}
def ordinal_mapper(value: int):
    if value == 1:
        return 'Fa'
    elif value == 2:
        return 'TA'
    elif value == 3:
        return 'Gd'
    else:
        return 'Ex'

def garage_attach(value: str):
    if value == 'Yes':
        return "Attchd"
    else:
        return "Detchd"

grlivarea = st.number_input("Enter the ground living area(in square feet):", step=1)

overall_quality = st.slider("Choose Overall Quality:", 1, 10)

exterior_quality = st.slider("Choose Exterior Material Quality:", 1, 4)
st.write(ordinal_mapper(exterior_quality))
KitchenQuality = st.slider("Choose Kitchen Quality:", 1, 4)
st.write(ordinal_mapper(KitchenQuality))
basement_quality = st.slider("Choose Basement Quality:", 1, 4)
st.write(ordinal_mapper(basement_quality))

garage_attached = st.radio("Do you want garage attached with house?", ("Yes", "No"))
st.write(garage_attach(garage_attached))

garage_cars = st.number_input("Enter size of garage(in car capacity):", step=1)

df_dict = {'OverallQual': overall_quality, 
            'ExterQual': ordinal_mapper(exterior_quality), 
            'GarageCars': garage_cars, 
            'BsmtQual': ordinal_mapper(basement_quality),
            'KitchenQual': ordinal_mapper(KitchenQuality),
            'GarageType': garage_attach(garage_attached), 
            'GrLivArea': grlivarea}
mod_df = pd.DataFrame(df_dict, index=[0])

def predict():
    df.loc[:, 'OverallQual'] = mod_df.loc[:, 'OverallQual']
    df.loc[:, 'ExterQual'] = mod_df.loc[:, 'ExterQual']
    df.loc[:, 'GarageCars'] = mod_df.loc[:, 'GarageCars']
    df.loc[:, 'BsmtQual'] = mod_df.loc[:, 'BsmtQual']
    df.loc[:, 'KitchenQual'] = mod_df.loc[:, 'KitchenQual']
    df.loc[:, 'GarageType'] = mod_df.loc[:, 'GarageType']
    df.loc[:, 'GrLivArea'] = mod_df.loc[:, 'GrLivArea']

    df_processed = input_preprocessor(df, scaler=scaler, encoder=encoder)
    prediction = model.predict(df_processed)
    return round(prediction[0])

if st.button("Submit"):
    amount = predict()
    st.markdown(f'<div style="font-size: 30px; color: black; background-color: white; padding: 10px; border: 1px solid black;">Predicted Amount:</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 30px; font-family: Arial, sans-serif; color: white; background-color: black; padding: 10px; border: 1px solid white;">${amount}</div>', unsafe_allow_html=True)