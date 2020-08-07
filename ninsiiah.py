import streamlit as st
import tweepy
import yaml
import operator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import csv
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from nltk.corpus import stopwords
import gensim



# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

from collections import Counter
#import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer


with open("config.yml", "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
        
auth = tweepy.OAuthHandler(config['CONSUMER_KEY'], config['CONSUMER_SECRET']) 
auth.set_access_token(config['ACCESS_TOKEN'], config['ACCESS_TOKEN_SECRET'])

api = tweepy.API(auth)



image = Image.open('./0.jfif')

st.image(image, use_column_width=True)

st.header("Twitter Data Analysis")
image = Image.open('./2.png')
#image = image.resize((160,300),Image.ANTIALIAS)
st.image(image)
st.write("Enter a Twitter username in the text box without '@' at the beginning. Click the 'Get Tweets' button and get data and sentiment analysis of up to 3200 of the latest tweets. Press the stop button in right hand corner to stop the app. Press the reset button to reset the app.")

t = st.text_input("Enter a username to get tweets")

start = st.button("Get Tweets")
#med = st.button("Get User Tweets")
stop = st.button("Reset")

analyser = SentimentIntensityAnalyzer()


    
class StreamListener(tweepy.StreamListener):
    
    def on_status(self, status):
        if not stop:
            if hasattr(status, "retweeted_status"):
                pass
            else:
                try:
                    text = status.extended_tweet["full_text"]
                    score = analyser.polarity_scores(text)
                    st.write(text)
                    m = max(score.items(), key=operator.itemgetter(1))[0]
                    if m == 'neg':
                        st.error("The sentiment is: {}".format(str(score)))
                    elif m == 'neu':
                        st.warning("The sentiment is: {}".format(str(score)))
                    elif m == 'pos':
                        st.success("The sentiment is: {}".format(str(score)))
                    else:
                        st.info("The sentiment is: {}".format(str(score)))
                except AttributeError:
                    text = status.text
                    score = analyser.polarity_scores(text)
                    st.write(text)
                    m = max(score.items(), key=operator.itemgetter(1))[0]
                    if m == 'neg':
                        st.error("The sentiment is: {}".format(str(score)))
                    elif m == 'neu':
                        st.warning("The sentiment is: {}".format(str(score)))
                    elif m == 'pos':
                        st.success("The sentiment is: {}".format(str(score)))
                    else:
                        st.info("The sentiment is: {}".format(str(score)))
            return True
        else:
            exit()
            return False

def get(name):
    stuff = api.user_timeline(screen_name = name, include_rts = True)
    for status in stuff:
        st.info(status.author.description)

       
def stream_tweets(tag):
    listener = StreamListener(api=tweepy.API(wait_on_rate_limit=True, wait_on_rate_limit_notify=True))
    streamer = tweepy.Stream(auth=auth, listener=listener, tweet_mode='extended')
    query = [str(tag)]
    streamer.filter(track=query, languages=["en"])

    
def sent_to_words(words):
    for sentence in words:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  


def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method

	#initialize a list to hold all the tweepy Tweets
    alltweets = []	
	
    bar = st.progress(0)     
	#make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    

    n1 = 20

    bar.progress(20)
	#save most recent tweets
    alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
       # print("getting tweets before %s" % (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200, max_id=oldest)
		
		#save most recent tweets
        alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
        
        oldest = alltweets[-1].id - 1
        n1+=3
        bar.progress(n1)
        
    bar.progress(100) 
        #st.warning("Downloading %s tweets..." % (len(alltweets)))
        #progress_bar = st.progress(0)
        
    
        #progress_bar.progress(min(counter / length, 1.0))
        #for i in alltweets:
        #    print(i.user.screen_name, i.text, i.retweet_count, i.favorite_count,i.user.followers_count, i.user.friends_count, i.in_reply_to_status_id, i.in_reply_to_status_id_str)
        
	#transform the tweepy tweets into a 2D array that will populate the csv	
    outtweets = [[i.user.screen_name, i.created_at, i.text.encode("utf-8"),i.retweet_count, i.favorite_count,i.user.followers_count, i.user.friends_count] for i in alltweets]
    #status = alltweets[19]
    #print(alltweets)
   
    df = pd.DataFrame.from_records(outtweets)
    
    df.columns = ["screen_name","created_at", "text", "retweet_count", "favorite_count", "followers_count", "friends_count"]
    following = df["friends_count"]
    followers = df["followers_count"]
    df= df.drop("friends_count",1)
    df = df.drop("followers_count",1)
    df["text"] = df["text"].str.decode('utf-8') 
    df['weekday'] = df['created_at'].dt.dayofweek
    df['weekday'] = df['weekday'].astype(str)
    df['weekday'] = df['weekday'].str.replace('0','0: Monday')
    df['weekday'] = df['weekday'].str.replace('1','1: Tuesday')
    df['weekday'] = df['weekday'].str.replace('2','2: Wednesday')
    df['weekday'] = df['weekday'].str.replace('3','3: Thursday')
    df['weekday'] = df['weekday'].str.replace('4','4: Friday')
    df['weekday'] = df['weekday'].str.replace('5','5: Saturday')
    df['weekday'] = df['weekday'].str.replace('6','6: Sunday')
    
    df['text'] = df['text'].astype(str)

    df['Sentiment Analysis'] = df['text'].apply(lambda tweet: 'positive' if TextBlob(tweet).sentiment.polarity > 0  else ('neutral' if TextBlob(tweet).sentiment.polarity == 0 else 'negative'))
    
    words = df.text.tolist()
    
    #stop_words = stopwords.words('english')
    stop_words = ['from', 'com','the ','http','https', 'co', 'on', 'here', 'rt', 'of', 'to','is','www','the','you','an','via','it','in','are','let','subject','to', 'they','re', 'edu', 'use', 'the','https', 'will', 'thee', 'one', 'an', 'really', 'even', 'take','lot', 'nan','take','want' 'take'] + list(STOPWORDS)
    data_words = list(sent_to_words(words))
    list_words = []
    for i in data_words:
        for j in i:
            list_words.append(j)
            
    for token in list_words:
        if token in stop_words:
            list_words.remove(token)

    

#    if app_mode == "Show instructions":
#        st.sidebar.success('To continue select "Run the app".')
#    elif app_mode == "Show the source code":
#        readme_text.empty()
#        st.code(get_file_content_as_string("app.py"))

    st.title("Table of Tweets")

    
    st.dataframe(df)
    
    st.title("Basic Stats of Tweets")
    st.info("Number of Tweets: " + str(df.shape[0]))
    st.info("Number of Following: " + str(following[0]))
    st.info("Number of Followers: " + str(followers[0]))
    st.info("Total Number of Favorite Counts: " + str(sum(df["favorite_count"])))
    st.info("Average Number of Favorite Count Per Tweet: " + str((sum(df["favorite_count"]))/(df.shape[0])))
    #st.dataframe(df2)
    
    df_a =df.sort_values('favorite_count', ascending=False)
    
    
    
        
        
    
    df3 = df.groupby('weekday').count()
    
    df3['weekday'] = df3.index
    df3 = df3.rename(columns={'screen_name': 'num_tweets'})
   
    df4 = df.groupby(['weekday'])['favorite_count'].sum()
    df4= df4.to_frame() 
    df4['weekday'] = df4.index
    
 
    st.header("Number of Tweets Categorized by the Day of the Week")
    st.markdown("This gives a general overview of the number of tweets an account puts out by day of the week"
  )
    st.dataframe(df3[['num_tweets']])
    fig = px.scatter(df3, x="weekday", y="num_tweets")
    
    st.header("Scatter Diagram of the Number of Tweets Categorized by the Day of the Week")

    st.plotly_chart(fig)
    
    st.header("Pie Chart of the Number of Tweets Categorized by the Day of the Week")

    fig = go.Figure(data=[go.Pie(labels=df3['weekday'], values=df3['num_tweets'])])
    st.plotly_chart(fig)
    
   
   
   
    st.header("Number of Tweets Categorized by the Day of the Week")    
    st.markdown("This gives a general overview of the total number of favorite count of tweets of an account categorized by day of the week"
)
    st.dataframe(df4)
    fig = px.scatter(df4, x="weekday", y="favorite_count")
    st.header("Scatter Diagram of the Total Number of Favorite Count of Tweets Categorized by the Day of the Week")
    st.plotly_chart(fig)
    fig = go.Figure(data=[go.Pie(labels=df4['weekday'], values=df4['favorite_count'])])
    st.header("Pie Chart of the Total Number of Favorite Count of Tweets Categorized by the Day of the Week")
    st.plotly_chart(fig)
    
    df5 = df.groupby(['weekday'])['retweet_count'].sum()
    df5= df5.to_frame() 
    df5['weekday'] = df5.index 
    
        
    
    word_could_dict=Counter(list_words)
    wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)
    
    
    fig = px.scatter(df_a.head(10), x="created_at", y="favorite_count", color="Sentiment Analysis")
   
    
    st.header("Sentiment Analysis of Tweets")    
    st.markdown("This is a table of the tweets and their sentiment analysis.")
    
    st.dataframe(df_a)
    st.header("Scatter Diagram of the Top Tweets by Favorite Count and Colored by Sentiment Analysis.")
    st.plotly_chart(fig)
    
    df6 = df.groupby(['Sentiment Analysis']).count()
    
    
    fig = go.Figure(data=[go.Pie(labels=df6.index, values=df6['retweet_count'])])
    st.header("Pie Chart of Tweets Categorized by Sentiment Analysis.")
    st.plotly_chart(fig)   
    
    
    
    st.header("Word Cloud of Tweets")
    plt.figure(figsize=[20,10])
    plt.figtext(.5,.9,"Twitter WordCloud",  fontsize=60, ha='center', color='black')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    st.pyplot()
    
    
    
   # with open('data.json', 'w') as f:
   #    json.dump(status._json, f)
   # print(alltweets.Status._json)
    #print(outtweets)
    
if start:
    #stream_tweets(t)
    get_all_tweets(str(t))
    
#if med:
#    get(str(t))