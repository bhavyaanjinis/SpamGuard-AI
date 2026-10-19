
# SpamGuard AI

A beginner-friendly machine learning application that
classifies text messages as Spam or Not Spam.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit

## Machine Learning
- CountVectorizer for text feature extraction
- Multinomial Naive Bayes classifier
- Train-test split for evaluation

## How to Run

1. Install the required libraries:

   pip install pandas scikit-learn streamlit

2. Run the application:

   streamlit run app.py

3. Enter a message and click Check Message.

## Dataset
A small, manually prepared dataset is used for initial
testing. Predictions may be inaccurate on unfamiliar messages.

## Future Improvements
- Use a larger labelled dataset
- Evaluate precision, recall and F1-score
- Test on unseen messages
