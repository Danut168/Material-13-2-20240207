# TODO: Required Libraries
import pickle
import pandas as pd
import streamlit as st

# TODO: Load Dumped Files
label = pickle.load(file=open(file="./label.pkl", mode="rb"))
scaler = pickle.load(file=open(file="./scaler.pkl", mode="rb"))
value = pickle.load(file=open(file="./value.pkl", mode="rb"))
model = pickle.load(file=open(file="./model.pkl", mode="rb"))
st.title(body="Iris Web App", anchor=False)
st.header(body="User Input")
# TODO: Load Sample Data Set
# sample = pd.read_csv(filepath_or_buffer="./sample.csv")
# TODO: Upload Sample
uploaded_file = st.file_uploader(label="Upload sample", type="csv")
# TODO: Making Prediction
input_list = list()
for key in label["features"]:
    input_value = st.sidebar.number_input(label=key, value=value[key], step=0.1)
    input_list.append(input_value)
if uploaded_file is None:
    sample = pd.DataFrame(data=[input_list],columns=label["features"])
    # st.write(sample)
else:
    sample = pd.read_csv(filepath_or_buffer=uploaded_file)
df_test = sample.copy()
for key in label["features"]:
    df_test[key] = scaler[key].transform(X=sample[[key]].values)
X_test = df_test.values
y_pred = model.predict(X=X_test)
# st.dataframe(data=sample)
# st.dataframe(data=y_pred)
result = sample.copy()
result["prediction"] = y_pred
st.dataframe(data=result)
# print(X_test)
# print(y_pred)
