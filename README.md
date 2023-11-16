# P5-Sentiment-Analysis

## SENTIMENT ANALYSIS
Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique that involves extracting the sentiment or emotional tone expressed in a piece of text.
Sentiment analysis aims to classify text data into different categories or labels that represent sentiment, such as positive, negative, neutral, or even more specific emotions like happy, sad, angry, etc.

## COVID VACCINE TWEETS(DATASET)
Data for this project was obtained from the #ZindiWeekendz To Vaccinate or Not to Vaccinate: It’s not a Question Challenge found [here](https://zindi.africa/competitions/to-vaccinate-or-not-to-vaccinate.)
The data comes from tweets collected and classified through Crowdbreaks.org [Muller, Martin M., and Marcel Salathe. "Crowdbreaks: Tracking Health Trends Using Public Social Media Data and Crowdsourcing." Frontiers in public health 7 (2019).]. Tweets have been classified as pro-vaccine (1), neutral (0) or anti-vaccine (-1). The tweets have had usernames and web addresses removed.


The objective of this challenge was to develop a machine learning model to assess if a Twitter post related to vaccinations is positive, neutral, or negative. 
The end solution can help governments and other public health actors to monitor public sentiment towards COVID-19 vaccinations and help improve public health policy, vaccine communication strategies,
and vaccination programs across the world.

Variable definition:

tweet_id: Unique identifier of the tweet

safe_tweet: Text contained in the tweet. Some sensitive information has been removed like usernames and urls

label: Sentiment of the tweet (-1 for negative, 0 for neutral, 1 for positive)

agreement: The tweets were labeled by three people. Agreement indicates the percentage of the three reviewers that agreed on the given label. You may use this column in your training, but agreement data will not be shared for the test set.

Files available for download are:

Train.csv - Labelled tweets on which to train your model

Test.csv - Tweets that you must classify using your trained model

SampleSubmission.csv - is an example of what your submission file should look like. The order of the rows does not matter, but the names of the ID must be correct. Values in the 'label' column should range between -1 and 1.


## DESCRIPTION

This repository contains :

* An EDA notebook where the data was explored and cleaned before worked upon
  
* A dataset folder which contains all the data set we worked with bot original and cleaned
  
* Two notebook for the finetuning of both a DistilBert and a BERT model . This model and the evaluation details can be found on hugging face at
    1. [DistilBert Model](https://huggingface.co/Afia-manubea/FineTuned-DistilBert-Model)
       
    2. [BertTweet Model](https://huggingface.co/Afia-manubea/FineTuned-BertTweet-Classification-Model)
       
* Two inference notebook for both gradio and streamlit
  
* And finally, a streamlit_app.py and gradio_app.py which each contain codes for the building of streamlit and gradio user interface respectively.The two apps can be viewed through the following links
    1. [Gradio App](https://huggingface.co/spaces/Afia-manubea/Covid-Tweet-Sentiment-Gradio-App)
  
    2. [Streamlit App](https://huggingface.co/spaces/Afia-manubea/Covid-Tweets-Sentiment-Streamlit-App)




