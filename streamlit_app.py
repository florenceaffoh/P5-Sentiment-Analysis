import subprocess

subprocess.run(["pip", "install", "transformers"])
subprocess.run(["pip", "install", "torch"])

import time
import pandas as pd
from PIL import Image

import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch

# Define model paths
model_paths = {
    "Model 1-DistilBert": "Afia-manubea/FineTuned-DistilBert-Model",
    "Model 2-BertTweet": "Afia-manubea/FineTuned-BertTweet-Classification-Model"
}




def main():
    # Setup web page
    st.set_page_config(
        page_title="Covid Vaccine tweets Sentiment",
        layout="wide",
        menu_items={
            'About': "The source code for this application can be accessed on GitHub "
        }
    )

    # Initialize the tokenizer and model
    selected_model = st.sidebar.selectbox("Select Model", list(model_paths.keys()))
    tokenizer = AutoTokenizer.from_pretrained(model_paths[selected_model])
    model = AutoModelForSequenceClassification.from_pretrained(model_paths[selected_model])

    st.markdown("""
    <style type="text/css">
    blockquote {
        margin: 1em 0px 1em -1px;
        padding: 0px 0px 0px 1.2em;
        font-size: 20px;
        border-left: 5px solid rgb(230, 234, 241);
        # background-color: rgb(129, 164, 182);
    }
    blockquote p {
        font-size: 30px;
        color: #FFFFFF;
    }
    [data-testid=stSidebar] {
        background-color: rgb(129, 164, 182);
        color: #FFFFFF;
    }
    [aria-selected="true"] {
         color: #000000;
    }
    </style>
""", unsafe_allow_html=True)



    st.sidebar.title(' **Covid Vaccine tweets Sentiment**')
    st.sidebar.title(' **Navigation**')

    # Customize the width of the sidebar
    # st.sidebar.set_width(300)

    # Change the background color of the sidebar
    st.sidebar.markdown('<style>body {background-color: #f8f9fa;}</style>', unsafe_allow_html=True)

    page = st.sidebar.radio('Select a Page', ['Home', 'Predict', 'About'])

    if page == 'Home':
        home()

    elif page == 'Predict':
        predict_sentiment(tokenizer,model)

    elif page == 'About':
        about_info()

def home():



    # Add styled text using CSS
    st.markdown('<p style="color: blue; font-size: 20px;"></p>', unsafe_allow_html=True)
    st.markdown("## FINETUNED BERT SENTIMENT MODEL FOR CATEGORIZATION OF COVID TWEETS")

     # Open and display the image
    #img = Image.open('Covid-Tweets-Sentiment-Streamlit-App/p5/Covid-Tweets-Sentiment-Streamlit-App/pics/9019830.jpg')
    #st.image(img)

    # Specify the GitHub URL for the image
    image_url = 'https://huggingface.co/spaces/Afia-manubea/Covid-Tweets-Sentiment-Streamlit-App/raw/main/p5/Covid-Tweets-Sentiment-Streamlit-App/pics/9019830.jpg'

    # Display the image in Streamlit
    st.image(image_url, caption="Your Image Caption", use_column_width=True)


    st.write("""
     ### To ensure the best of **Experience** on the covid tweet sentiment App ,
    Navigate The App using the sidebar to:

    1. Predict sentiment on tweets
    2. Learn more about this project
    """)


def predict_sentiment(tokenizer,model):
    st.title("Classify Sentiment")

    # Add an information message
    st.info("Covid Tweets machine learning model to assess if a Twitter post related to vaccinations is positive, neutral, or negative. \nExample: 'Vaccine Who!, and where' ")

    # Getting user input
    tweet_text = st.text_input("Enter the COVID tweet:")

    if st.button("Predict"):
        if tweet_text:
            # Tokenize the input text
            inputs = tokenizer.encode_plus(tweet_text, return_tensors="pt", truncation=True, padding=True)

            # Forward pass through the model
            with torch.no_grad():
                output = model(**inputs)

            # Extract the predicted probabilities
            scores = torch.nn.functional.softmax(output.logits, dim=1).squeeze().tolist()

            # Define the sentiment labels
            labels = ["Negative", "Neutral", "Positive"]

            # Create a dictionary of sentiment scores
            scores_dict = {label: score for label, score in zip(labels, scores)}

            # Display predicted sales
            st.balloons()
            st.progress(10)
            with st.spinner('Wait for it...'):
              time.sleep(10)

            # Display results in a table
            st.write("Sentiment Scores:")
            df = pd.DataFrame(scores_dict.items(), columns=['Sentiment', 'Score'])
            st.table(df)


            # Get the index of the highest score
            max_index = scores.index(max(scores))

            # Get the corresponding sentiment label
            predicted_label = labels[max_index]

            # Display the predicted sentiment label
            st.success(f"Predicted Sentiment: {predicted_label}", icon="✅")

        else:
            st.warning("Please enter a COVID tweet to predict sentiment.")

def about_info():
    st.title("About")

     # Open and display the image
    #img = Image.open(r'Covid-Tweets-Sentiment-Streamlit-App\pics\8447107.jpg')
    #st.image(img)
    
    # Specify the GitHub URL for the image
    image_url = 'https://huggingface.co/spaces/Afia-manubea/Covid-Tweets-Sentiment-Streamlit-App/raw/main/p5/Covid-Tweets-Sentiment-Streamlit-App/pics/8447107.jpg'
    # Display the image in Streamlit
    st.image(image_url, caption="Your Image Caption", use_column_width=True)

    st.write("""
    Welcome to our Streamlit app! We are dedicated to providing a powerful solution that aims to help governments and public health actors monitor public sentiment towards COVID-19 vaccinations.
    By leveraging the capabilities of a fine-tuned BERT base uncased model from Hugging Face, we have developed a sentiment analysis tool specifically trained on COVID-19 vaccine tweets.

    Our platform allows users to enter a tweet and obtain an accurate sentiment label determined by our model. The sentiment labels include negative, neutral, or positive,
    providing valuable insights into public opinion surrounding COVID-19 vaccination efforts.

    The significance of monitoring public sentiment towards COVID-19 vaccines cannot be overstated. It plays a crucial role in shaping public health policies, vaccine communication strategies,
    and vaccination programs worldwide. By understanding the sentiments expressed in tweets related to COVID-19 vaccines, governments and public health actors can gain valuable
    insights into public perception, concerns, and attitudes.

    Developed by Team Charleston.
    More details about the project and methodology can be found [here](https://github.com/florenceaffoh/P5-Sentiment-Analysis.git).
    """)


if __name__ == '__main__':
    main()
