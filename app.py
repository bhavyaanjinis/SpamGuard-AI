
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="SpamGuard AI", page_icon="🛡️")

st.title("🛡️ SpamGuard AI")
st.write("Real SMS dataset based spam detection")

@st.cache_data
def load_data():
    data = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )
    return data

try:
    data = load_data()
except Exception as e:
    st.error("Dataset load avvaledu. SMSSpamCollection file app.py unna folder lo undho check cheyyi.")
    st.stop()

data = data.dropna()
data["label"] = data["label"].str.lower().str.strip()
data = data[data["label"].isin(["ham", "spam"])]

st.write("Total SMS messages:", len(data))
st.write("Spam messages:", int((data["label"] == "spam").sum()))
st.write("Normal messages:", int((data["label"] == "ham").sum()))

X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.metric("Test Accuracy", f"{accuracy:.2%}")

report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0
)

st.subheader("Model Evaluation")
st.write(f"Spam precision: {report['spam']['precision']:.2%}")
st.write(f"Spam recall: {report['spam']['recall']:.2%}")
st.write(f"Spam F1-score: {report['spam']['f1-score']:.2%}")

st.caption(
    "Evaluation uses a held-out test split. Results may vary on new messages."
)

st.subheader("Test a Message")
message = st.text_area("Enter an SMS message:")

if st.button("Check Message"):
    if message.strip():
        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        confidence = max(probabilities)

        if prediction == "spam":
            st.error(f"⚠️ This message looks like SPAM. Model score: {confidence:.1%}")
        else:
            st.success(f"✅ This message looks like NOT SPAM. Model score: {confidence:.1%}")
    else:
        st.warning("Please enter a message first.")
