import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessing tools
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoder_ohe.pkl", "rb") as f:
    ohe = pickle.load(f)
with open("vectorizer_tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("products.csv")
    df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)
    return df.dropna()

df = load_data()

# Streamlit UI
st.title("🎀 Pink Tax Detector")

brand = st.selectbox("Brand", df['Brand'].unique())
category = st.selectbox("Category", df['Category'].unique())
description = st.text_area("Product Description", "Basic white cotton t-shirt")

def make_input(gender):
    df_input = pd.DataFrame({
        'Brand': [brand],
        'Category': [category],
        'Gender': [gender]
    })
    X_cat = ohe.transform(df_input)
    X_desc = tfidf.transform([description])
    return hstack([X_cat, X_desc])

X_male = make_input("M")
X_female = make_input("F")

price_m = model.predict(X_male)[0]
price_f = model.predict(X_female)[0]

st.subheader("💵 Predicted Prices")
st.write(f"Male: ${price_m:.2f}")
st.write(f"Female: ${price_f:.2f}")

diff = price_f - price_m
if diff > 0:
    st.markdown(f"🔺 Pink Tax: ${diff:.2f} higher for women.")
elif diff < 0:
    st.markdown(f"🟢 Price is ${-diff:.2f} cheaper for women.")
else:
    st.markdown("⚖️ No price difference detected.")

st.subheader("📊 Average Price by Gender")
avg_prices = df.groupby("Gender")["Price"].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(data=avg_prices, x="Gender", y="Price", palette="pastel", ax=ax)
st.pyplot(fig)