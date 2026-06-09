import pandas as pd
import numpy as np
import pickle
import streamlit as st
# LOADING THE SAVED MODEL

loaded_model=pickle.load(open('C:/Users/acer/OneDrive/Documents/final_trained_model.sav','rb'))

# LODING THE SAVED STANDARDSCALER

scaler = pickle.load(open('C:/Users/acer/OneDrive/Documents/scaler.sav', 'rb'))

# CFREATING A FUNCTION FOR PREDICTION

def customer_prediction(input_data):

    input_data_scaled = scaler.transform([input_data])

    result = loaded_model.predict(input_data_scaled)

    print(result)
    if(result[0]== 1):
        return "THE CUSTOMER WILL BUY THE CAR"
    else:
        return "THE CUSTOMER WILL NOT BUY THE CAR"
    
def main():
    
    # GIVING THE TITLE 
    st.title("🚗 Car Purchase Prediction System")
    st.write("Enter customer details to predict whether the customer is likely to purchase a car.")
    
    # GETTING THE INPUT DATA FROM THE USER
    age=st.text_input('ENTER YOUR AGE : ')
    salary=st.text_input('ENTER YOUR SALARY : ')
    
    # CODE FOR PREDICTION
    prediction=''
    
    # CREATING A BUTTON FOR PREDICTION
    if st.button("ANALYZE"):
        prediction=customer_prediction([age,salary])
    
    st.success(prediction)


if __name__ == '__main__':
    main()

