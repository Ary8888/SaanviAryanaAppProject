import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

# Load data
df = pd.read_csv("products.csv")
df.dropna(inplace=True)
df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)

# Encode category data
cat_features = ['Brand', 'Category', 'Gender']
ohe = OneHotEncoder(handle_unknown='ignore')
X_cat = ohe.fit_transform(df[cat_features])

# Encode text data
tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(df['Product Description'])

# Final input
X = hstack([X_cat, X_text])
y = df['Price']

# Train model
model = Ridge()
model.fit(X, y)

# Save everything
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("encoder_ohe.pkl", "wb") as f:
    pickle.dump(ohe, f)
with open("vectorizer_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ All models saved!")

