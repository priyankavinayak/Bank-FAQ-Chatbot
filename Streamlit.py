import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import fsspec 



# Load the preprocessed data
df=pd.read_csv("./BankFAQs.csv")
df1=pd.read_csv("./TobeMerged.csv",encoding='ISO-8859-1')

data1=pd.concat([df1,df])

# Define the TD-IDF vectorizer and fit it to the data
tdidf = TfidfVectorizer()
tdidf.fit(data1['Question'].str.lower())

# Define the support vector machine model and fit it to the data
svc_model = SVC(kernel='linear')
svc_model.fit(tdidf.transform(data1['Question'].str.lower()), data1['Class'])

# Define a function to get the answer to a given question
def get_answer(question):
    # Vectorize the question
    question_tdidf = tdidf.transform([question.lower()])
    
    # Calculate the cosine similarity between both vectors
    cosine_sims = cosine_similarity(question_tdidf, tdidf.transform(data1['Question'].str.lower()))

    # Get the index of the most similar text to the query
    most_similar_idx = np.argmax(cosine_sims)

    # Get the predicted class of the query
    predicted_class = svc_model.predict(question_tdidf)[0]
    
    # If the predicted class is not the same as the actual class, return an error message
    if predicted_class != data1.iloc[most_similar_idx]['Class']:
        return 'Could not find an appropriate answer.'
    
    # Get the answer and construct the response
    answer = data1.iloc[most_similar_idx]['Answer']
    response = f"Answer: {answer}"
    
    return response
import streamlit as st
import subprocess

def run_flask_app():
    # Run the Flask app in a separate process using subprocess
    flask_app_process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE)

    # Capture the output of the Flask app
    flask_output = flask_app_process.communicate()[0]

    return flask_output

# Create a placeholder where the output of the Flask app will be displayed
output_placeholder = st.empty()

# Run the Flask app and display the output in the placeholder
flask_output = run_flask_app()
output_placeholder.write(flask_output.decode())

# Create a streamlit app
def app():
    # Set the app title
    st.set_page_config(page_title="Bank FAQ Chatbot", page_icon=":bank:")

    # Add a title and description to the app
    st.title("Bank FAQ Chatbot")
    st.markdown("This app uses a machine learning model to answer frequently asked questions about banking.")

    # Create a text input for the user to ask a question
    question = st.text_input("Ask a question:")

    # Add a button to submit the question
    if st.button("Submit"):
        # Check if the user has entered a question
        if question == "":
            st.warning("Please enter a question.")
        else:
            # Call the get_answer function to predict the answer to the question
            answer = get_answer(question)

            # Display the answer to the user
            st.success(answer)

# Run the streamlit app
if __name__ == '_main_':
    app()
