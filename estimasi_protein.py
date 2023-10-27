import pickle 
import streamlit as st 

model = pickle.load(open('estimasi_protein.sav', 'rb'))

st.title('Estimasi Protein Dalam Menu McDonalds')

TotalFat = st.number_input('Input Total Lemak')
SaturatedFat = st.number_input('Input Jumlah Lemak Jenuh')
Sugars = st.number_input('Input Jumlah Gula')
Cholesterol = st.number_input('Input Jumlah Kolestrol')
Sodium = st.number_input('Input Jumlah Sodium')
Calories = st.number_input('Input Jumlah Kalori')
Carbohydrates = st.number_input('Input Jumlah Karbohidrat')

predict = ''

if st.button('Estimasi Protein '):
    predict = model.predict(
        [[TotalFat, SaturatedFat, Sugars, Cholesterol, Sodium, Calories, Carbohydrates]]
    )
    st.write('Estimasi Protein Menu McDonalds: ', predict)