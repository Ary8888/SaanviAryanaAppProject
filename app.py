import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Pink Tax Predictor", layout="centered")

# Load model, encoder, and data
@st.cache_resource
def load_resources():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    df = pd.read_csv("cleaned_data.csv")
    return model, encoder, df

model, encoder, df = load_resources()

st.title(" Pink Tax Price Predictor")

# User inputs
brand = st.sidebar.selectbox("Brand", df['Brand'].unique())
item_type = st.sidebar.selectbox("Item Type", df['Item Type'].unique())
gender = st.sidebar.selectbox("Gender", df['Gender'].unique())

# Prediction
input_df = pd.DataFrame({'Brand': [brand], 'Item Type': [item_type], 'Gender': [gender]})
encoded_input = encoder.transform(input_df).toarray()
predicted_price = model.predict(encoded_input)[0]

st.subheader(" Predicted Price")
st.success(f"Estimated Price: **${predicted_price:.2f}**")

#  Price by Gender
st.subheader("Average Price by Gender")
avg = df.groupby("Gender")["Price"].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=avg, x="Gender", y="Price", palette="pastel", ax=ax)
ax.set_title("Average Price by Gender")
st.pyplot(fig)

#  Discrepancy by Item
st.subheader(" Price Differences by Category")
pivot = df.pivot_table(values="Price", index="Item Type", columns="Gender", aggfunc="mean")
pivot["Discrepancy ($F - $M)"] = pivot.get("F", 0) - pivot.get("M", 0)
st.dataframe(pivot.sort_values("Discrepancy ($F - $M)", ascending=False).style.format("{:.2f}"))
