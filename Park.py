import streamlit as st
import pandas as pd
from pydantic import BaseModel
from typing import List
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pathlib import Path

st.image('logo.png')
st.title("Park")
st.subheader("Detecting And Tracking Parkinson's Disease With Mobility Metrics")
df = pd.read_csv("ParkData.csv")
st.subheader("94% Test Accuracy When Predicting Parkinson's With Mobility Metrics")

##hour_to_filter = st.slider('hour', 0, 23, 17)
##df['endDate'] = pd.to_datetime(df['endDate'], errors='coerce')
filtered_data = df[df["sourceName"] == 'Healthy']

st.header('Double Support Time')
st.write('Double support occurs when both feet are in contact with the ground simultaneously; double support time is the sum of the time elapsed during two periods of double support in the gait cycle')
st.subheader('Healthy')
mean = filtered_data['double'].mean()
st.write('Average: ' + str(mean))
st.line_chart(filtered_data["double"])


filtered_data2 = df[df["sourceName"] == 'Parkinsons']


st.subheader("Parkinson's")
mean = filtered_data2['double'].mean()
st.write('Average: ' + str(mean))
st.line_chart(filtered_data2["double"])


st.header('Step Length')
st.write('Step length is the distance covered when a person takes one step.')
st.subheader('Healthy')
mean = filtered_data['length'].mean()
median = filtered_data['length'].median()
st.write('Average: ' + str(mean))
st.line_chart(filtered_data["length"])


filtered_data2 = df[df["sourceName"] == 'Parkinsons']


st.subheader("Parkinson's")
mean = filtered_data2['length'].mean()
median = filtered_data2['length'].median()
st.write('Average: ' + str(mean))
st.line_chart(filtered_data2["length"])



st.header('Walking Speed')
st.write('Walking speed is the measurement of how fast a person walks.')
st.subheader('Healthy')
mean = filtered_data['length'].mean()
median = filtered_data['length'].median()
st.write('Average: ' + str(mean))
st.line_chart(filtered_data["length"])


filtered_data2 = df[df["sourceName"] == 'Parkinsons']


st.subheader("Parkinson's")
mean = filtered_data2['length'].mean()
median = filtered_data2['length'].median()
st.write('Average: ' + str(mean))
st.line_chart(filtered_data2["length"])



class MultipleInputs(BaseModel):
    double: List[float]
    speed: List[float]
    length: List[float]

def load_regression_model():
    import_dir = Path("/Users/andreas/Desktop/ParkML/reg_model.sav")
    model = pickle.load(open(import_dir, "rb"))
    return model

def load_classifier_model():
    import_dir = Path("/Users/andreas/Desktop/ParkML/class_model.sav")
    model = pickle.load(open(import_dir, "rb"))
    return model

def multi_pred(item: MultipleInputs):
    # reshape inputs
    model_input = []
    for d, s, l in zip(item.double, item.speed, item.length):
        model_input.append([d, s, l])
    reg_model = load_regression_model()
    class_model = load_classifier_model()

    reg_pred = reg_model.predict(model_input)
    class_pred = class_model.predict(model_input)
    return {
        "regression_predictions": [float(i) for i in list(reg_pred)],
        "classification_predictions": [int(i) for i in list(class_pred)],
    }

load_regression_model()

predictions = pd.DataFrame(multi_pred(filtered_data)['regression_predictions']) 


st.header('Predicted Score')
st.write('The predicted score is estimated using  double support time, step length, and walking speed.  A 0 indicates a healthier condition while a 1 indicates a poorer condition.')
st.subheader('Healthy')
mean = predictions.mean()
median = predictions.median()
strMedian = str(median).replace('dtype: float64', '')
st.write('Median: ' + strMedian)
st.line_chart(predictions)




predictions = pd.DataFrame(multi_pred(filtered_data2)['regression_predictions']) 

st.subheader("Parkinson's")
mean = predictions.mean()
median = predictions.median()
strMedian = str(median).replace('dtype: float64', '')
st.write('Median: ' + strMedian)
st.line_chart(predictions)


