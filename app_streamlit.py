import streamlit as st 
import pandas as pd 
import pickle

df_wine = pd.read_csv("winequality-red.csv")

## Ouvrir le modèle avec pickle
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

st.title('Estime la note du vin')

st.write('Entrez les caractéristiques du vin pour obtenir une estimation de sa note de qualité.')

fixed_acidity = st.slider('Acidité fixe', min_value=3.0, max_value=15.0, value=8.0, step=0.1)
volatile_acidity = st.slider('Acidité volatile', min_value=0.1, max_value=2.0, value=0.5, step=0.1)
citric_acid = st.slider('Acide citrique', min_value=0.0, max_value=1.0, value=0.5, step=0.1)
residual_sugar = st.slider('Sucre résiduel', min_value=0.0, max_value=15.0, value=8.0, step=0.1)
chlorides = st.slider('Chlorures', min_value=0.0, max_value=0.5, value=0.1, step=0.01)
free_sulfur_dioxide = st.slider('Dioxyde de soufre libre', min_value=1, max_value=100, value=30, step=1)
total_sulfur_dioxide = st.slider('Dioxyde de soufre total', min_value=6, max_value=300, value=60, step=1)
density = st.slider('Densité', min_value=0.990, max_value=1.005, value=0.995, step=0.0001)
pH = st.slider('pH', min_value=2.0, max_value=5.0, value=3.0, step=0.1)
sulphates = st.slider('Sulfates', min_value=0.0, max_value=2.0, value=0.5, step=0.1)
alcohol = st.slider('Alcool', min_value=8.0, max_value=15.0, value=10.0, step=0.1)


feature_values = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                   free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
X = df_wine.drop('quality',axis=1)
feature_names = X.columns.tolist()
X_pred = pd.DataFrame(feature_values, columns=feature_names)

# Prédire la qualité du vin
prediction = model.predict(X_pred)


if st.button('Valider'):
    st.write(f'La note du vin est de {round(prediction[0])}')