import streamlit as st
import pandas as pd
from typing import List
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score
import numpy as np
st.image("logo.png")
st.title("Park")
st.subheader("Detecting And Tracking Parkinson's Disease With Mobility Metrics")
df = pd.read_csv("data/Parkinsons.csv")
feature_list = [i for i in list(df.columns) if i != "sourceName"]
parkinsons_df = pd.read_csv("data/ParkinsonsData.csv")
healthy_df = pd.read_csv("data/HealthyData.csv")
st.subheader("98% Test Accuracy When Predicting Parkinson's With Mobility Metrics")

# hour_to_filter = st.slider('hour', 0, 23, 17)
# parkinsons_df['endDate'] = pd.to_datetime(parkinsons_df['endDate'], errors='coerce')
filtered_data = healthy_df[healthy_df["sourceName"] == "Healthy"]

st.header("Double Support Time (%)")
st.write(
    "Double support time is the precentage of time during a walk that both feet are on the ground."
)
st.subheader("Healthy")
mean = filtered_data["double"].mean()
st.write(f"Average: {mean}")
st.line_chart(filtered_data["double"])


filtered_data2 = parkinsons_df[parkinsons_df["sourceName"] == "Parkinsons"]


st.subheader("Parkinson's")
mean = filtered_data2["double"].mean()
# st.write("Average: " + str(mean))
st.write(f"Average: {mean}")
st.line_chart(filtered_data2["double"])


st.header("Step Length (inches)")
st.write("Step length is the distance covered when a person takes one step.")
st.subheader("Healthy")
mean = filtered_data["length"].mean()
median = filtered_data["length"].median()
# st.write("Average: " + str(mean))
st.write(f"Average: {mean}")
st.line_chart(filtered_data["length"])


filtered_data2 = parkinsons_df[parkinsons_df["sourceName"] == "Parkinsons"]
# filtered_data2.to_csv('/Users/andreas/Desktop/ParkML/data/ParkinsonsData.csv')

st.subheader("Parkinson's")
mean = filtered_data2["length"].mean()
median = filtered_data2["length"].median()
# st.write("Average: " + str(mean))
st.write(f"Average: {mean}")
st.line_chart(filtered_data2["length"])


st.header("Walking Speed (mph)")
st.write("Walking speed is the measurement of how fast a person walks.")
st.subheader("Healthy")
mean = filtered_data["speed"].mean()
median = filtered_data["speed"].median()
# st.write("Average: " + str(mean))
st.write(f"Average: {mean}")
st.line_chart(filtered_data["speed"])


filtered_data2 = parkinsons_df[parkinsons_df["sourceName"] == "Parkinsons"]


st.subheader("Parkinson's")
mean = filtered_data2["speed"].mean()
median = filtered_data2["speed"].median()
# st.write("Average: " + str(mean))
st.write(f"Average: {mean}")
st.line_chart(filtered_data2["speed"])


class MultipleInputs:
    double: List[float]
    speed: List[float]
    length: List[float]


def load_regression_model():
    import_dir = Path("models/reg_model.sav")
    model = pickle.load(open(import_dir, "rb"))
    return model


def load_classifier_model():
    import_dir = Path("models/class_model.sav")
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
        "class_predictions": [float(i) for i in list(class_pred)],
    }


model = load_regression_model()

predictions = pd.DataFrame(multi_pred(filtered_data)["regression_predictions"])


st.header("Predicted Score")
st.write(
    "The predicted score is estimated using double support time, step length, and walking speed. A 0 indicates a healthier condition while a 1 indicates a poorer condition."
)
st.subheader("Healthy")
mean = predictions.mean()
median = predictions.median()
strMedian = str(float(median)).replace("dtype: float64", "")
st.write("Median: " + strMedian)
st.line_chart(predictions)


predictions = pd.DataFrame(multi_pred(parkinsons_df))

st.subheader("Parkinson's")
mean = predictions["regression_predictions"].mean()
median = predictions["regression_predictions"].median()
strMedian = str(median).replace("dtype: float64", "")
st.write("Median: " + strMedian)
st.line_chart(predictions["regression_predictions"])

healthyFiltered = predictions[predictions["class_predictions"] == 0]
parkinsonsFiltered = predictions[predictions["class_predictions"] == 1]

# TODO: change system for finding model accuracy
##st.header("Model Accuracy = " + str(len(parkinsonsFiltered) / len(filtered_data2)))


    
predictions = pd.DataFrame(multi_pred(df))
##predictions = [
    ##    "Parkinsons" if item == 1 else "Healthy" for item in list(predictions)
   ## ]
predicted = []
actual = []
df["sourceName"] = [
        1 if item == "Parkinsons" else 0 for item in list(df["sourceName"])
    ]
for p in predictions["class_predictions"]:
    predicted.append(str(float(p)))
    
for a in df["sourceName"]:
    actual.append(str(float(a)))


##st.text(len(predicted))
st.header("Model Accuracy = " + str(accuracy_score(np.array(predicted), np.array(actual))))


def reg_metrics(predictions):
        # Calculate the absolute errors
        errors = abs(predictions - df["sourceName"])
        return round(np.mean(errors), 2)

meanError = reg_metrics(predictions["regression_predictions"])
st.header("Regression Mean Error = " + str(meanError))
st.write("A zero indicates a perfect model, while a higher vaue indicates a weaker model")


    