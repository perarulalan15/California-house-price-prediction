import numpy as np
import pickle
import streamlit as st


load_model=pickle.load(open('F:/Coding/Project/House price predict/trained_model.sav','rb'))

def price_predict(input_data):
    id_asnumpy = np.asarray(input_data)
    input_reshaped = id_asnumpy.reshape(1,-1)
    predict = load_model.predict(input_reshaped)
    print(predict)
    if predict >0:
        for i in predict:
            return i

def main():
    st.title('House price prediction')
    MedInc = st.text_input('Median income')
    HouseAge = st.text_input('House Age')
    Ave_rooms = st.text_input('Average Rooms')
    AveBedrms = st.text_input('Average Bedrooms')
    Population = st.text_input('Population')
    AveOccup = st.text_input('Average number of household members')
    lat = st.text_input('Latitude')
    lang = st.text_input('Longitude')

    predicted = ''

    if st.button('Prediction Result'):
        predicted = price_predict([MedInc,HouseAge,Ave_rooms,AveBedrms,Population,AveOccup,lat,lang])

    st.success(predicted)


if __name__ == '__main__':
    main()