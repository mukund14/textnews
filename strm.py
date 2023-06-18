#!/usr/bin/env python
# coding: utf-8

# In[417]:


api_key='d97a3d6087394870b4ccb6616fe18905'
#api_key='0af4ec9ff2a64d90ad6d8f92a49c9d20'


# In[418]:


import newsapi
from newsapi.newsapi_client import NewsApiClient
newsapi = NewsApiClient(api_key=api_key)
api_key='d97a3d6087394870b4ccb6616fe18905'
from pandas import json_normalize
import pandas as pd
pd.set_option('display.max_colwidth', -1)
  
import warnings
import streamlit as st
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup
import re
import requests
import glob 
import streamlit as st


# dataframe=requests.get("https://newsapi.org/v2/everything?q="+"Joe Biden"+"&apiKey="+api_key).json()
# json_normalize(dataframe).head(1)

# In[419]:


dataframe=requests.get("https://newsapi.org/v2/everything?q=us_stocks&apiKey="+api_key).json()


# In[420]:


from pandas import json_normalize
dfnew=json_normalize(dataframe)
pd.set_option('display.max_colwidth', -1)
import pandas as pd
import numpy as np
#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#for word embedding
import gensim
from gensim.models import Word2Vec


# In[421]:


import datetime
#datetime.datetime.now().date()
from datetime import datetime, timedelta
def date(base):
    date_list=[]
    yr=datetime.today().year
    if (yr%400)==0 or ((yr%100!=0) and (yr%4==0)):
        numdays=366
        date_list.append([base - timedelta(days=x) for x in range(366)])
    else:
        numdays=365
        date_list.append([base - timedelta(days=x) for x in range(365)])
    newlist=[]
    for i in date_list:
        for j in sorted(i):
            newlist.append(j)
    return newlist

def last_30(base):

    date_list=[base - timedelta(days=x) for x in range(30)]
    #newlist=[]
    #for i in sorted(date_list):
    #    newlist.append(j)
    return sorted(date_list)


def from_dt(x):
    from_dt=[]
    for i in range(len(x)):
        from_dt.append(last_30(datetime.today())[i-1].date())
        #to_dt=date(datetime.today())[i+1].date()
    return from_dt
        
def to_dt(x):
    to_dt=[]
    for i in range(len(x)):
        #from_dt=date(datetime.today())[i].date()
        to_dt.append(last_30(datetime.today())[i].date())
    return to_dt
from_list=from_dt(last_30(datetime.today()))
to_list=to_dt(last_30(datetime.today()))


# In[422]:


def text_from_urls(query):
    newd={}
    for (from_dt,to_dt) in zip(from_list,to_list):
        #all_articles = newsapi.get_everything(q=query,language='en',sort_by='relevancy',  page=1,   )
        d=json_normalize(dataframe['articles'])
 
 
        newdf=d[["url","source.name","title","content"]]
         
        dic=newdf.set_index(["source.name","title","content"])["url"].to_dict()

        for (k,v) in dic.items():
            #print(str(k[0])+str(k[1][5:10]))
            page = requests.get(v)
            html = page.content
            soup = BeautifulSoup(html, "lxml")
            text = soup.get_text()
            d2=soup.find_all("p")
            #for a in d2:
            newd[k]=re.sub(r'<.+?>',r'',str(d2)) 
    return newd.values()
#qry=input("Enter query for news search\n")
n=text_from_urls(dataframe)


# In[423]:


newd={}
#for (from_dt,to_dt) in zip(from_list,to_list):
        #all_articles = newsapi.get_everything(q=query,language='en',sort_by='relevancy',  page=1,   )
d=json_normalize(dataframe['articles'])
newdf=d[["url","source.name","title","content"]]
         
dic=newdf.set_index(["source.name","title","content"])["url"].to_dict()


# In[424]:


for (k,v) in dic.items():
            #print(str(k[0])+str(k[1][5:10]))
    page = requests.get(v)
    html = page.content
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text()
    d2=soup.find_all("p")
            #for a in d2:
    newd[k]=re.sub(r'<.+?>',r'',str(d2)) 


# In[425]:


n1=newd.values()


# In[426]:


def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)
#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
df_train=pd.DataFrame(n1)


# In[427]:


def text_from_urls(query):
    newd={}
    for (from_dt,to_dt) in zip(from_list,to_list):
        all_articles = newsapi.get_everything(q=query,language='en',sort_by='relevancy', page_size=1,page=1,   from_param=from_dt,to=to_dt)
        d=json_normalize(all_articles['articles'])
        newdf=d[["url","source.name","title","content"]]
        
        dic=newdf.set_index(["source.name","title","content"])["url"].to_dict()
        #print(dic)
        for (k,v) in dic.items():
            #print(str(k[0])+str(k[1][5:10]))
            page = requests.get(v)
            html = page.content
            soup = BeautifulSoup(html, "lxml")
            text = soup.get_text()
            d2=soup.find_all("p")
            #for a in d2:
            newd[k]=re.sub(r'<.+?>',r'',str(d2)) 
    return newd.values()
#qry=input("Enter query for news search")
#n=text_from_urls(qry)
#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)
#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
#df_train=pd.DataFrame(n)

#df_train.to_csv("news1_trainfile_"+str(qry)+".csv",index=False)


# In[428]:


df_train[0]=df_train[0].apply(lambda x: finalpreprocess(x))


# In[429]:


combineddf=df_train
df_train=combineddf


# In[430]:


df_train.rename(columns={0:'new'},inplace=True)
# df.rename(columns={'a': 'X
combineddf=df_train


# import nltk 
# df_train['new'] = df_train[0].apply(lambda x: finalpreprocess(x))
# df_train.drop(0,axis=1,inplace=True)

# In[431]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
import nltk
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_train['new'].values)
word_index = tokenizer.word_index


# This is fixed.

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_train['new'].values)
word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))


# In[432]:


d=dict(tokenizer.word_counts) 
df=pd.DataFrame.from_dict(d, orient='index')
df.reset_index(inplace=True)
#Setting top 30 features as ym
top30=df.sort_values(by=0,ascending=False).head(10)
top30.reset_index(inplace=True)
top30.rename(columns={'level_0':'index','index':'feature',0:'count'},inplace=True)
top30.drop('index',axis=1,inplace=True)

condensedlist=list(top30.feature)
top1000lst=condensedlist
top1000lst.sort()
#top1000lst
df3r4=pd.DataFrame(top1000lst).T
df3r4.columns=df3r4.loc[0]
df3r4.loc[1]=1
df3r4 = df3r4.iloc[1: , :]


# In[437]:



top1000lst
df_train10k=pd.DataFrame()
for i in top1000lst:
    if (i!='new') and (i!='dataset'):
        df_train[i]=0


# In[439]:


for i in top1000lst:
    df_train10k[i]=df_train[i]
df_train10k['new']=combineddf['new']


# In[440]:


#df_train10k=df_train[:5000]
df_train=df_train10k
df_train10k=df_train
df_train=df_train10k
df_train10k1=df_train10k[df_train10k['new']!=0]


# In[441]:


df_traincolfind=df_train.loc[:, ~df_train.columns.isin(['new', 'dataset'])]
for i1 in df_traincolfind.columns:
    for j,j1 in df_train.iterrows():
        if i1 in j1['new'].split(' '): 
       
            
            df_train[i1]=1
   
        
        else: 
            df_train[i1]=0
            


# df_train.to_csv("dftest.csv",index=False)

# df_traincolfindnew=df_train.loc[:, ~df_train.columns.isin(['new', 'dataset'])]

# df_train.to_csv("usstockstest.csv",index=False)

# df_train=pd.read_csv("usstockstest.csv")
# df_traincolfindnew=df_train.loc[:, ~df_train.columns.isin(['new', 'dataset'])]

# l=[]
# for i,i1 in df_traincolfindnew.iterrows():
#     y = np.hstack([i,i1])
#     l.append(y)

# In[443]:


X = tokenizer.texts_to_sequences(df_train['new'].values)
  
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
#print('Shape of data tensor:', X.shape)


# In[444]:



op1=df_train.iloc[:,1]
op2=df_train.iloc[:,2]
op3=df_train.iloc[:,3]
op4=df_train.iloc[:,4]
op5=df_train.iloc[:,5]
op6=df_train.iloc[:,6]
op7=df_train.iloc[:,7]
op8=df_train.iloc[:,8]
op9=df_train.iloc[:,9]
op10=df_train.iloc[:,10]


# In[445]:


op40=df_train.iloc[:,0:10]


# inputs = Input(shape=(250,), name='input')
# x = Dense(16, activation='relu', name='16')(inputs)
#  
# x = Dense(32, activation='relu', name='32')(x)
# output1 = Dense(1, name='cont_out')(x)
# output2 = Dense(1, activation='softmax', name='cat_out')(x)
#  
# inputs =  Input(shape=(250,), name='input')
# lstm = tf.keras.layers.LSTM(1)
# output = lstm(inputs)
# model = Sequential()
# model.add(LSTM(inputs=inputs, outputs=[output1, output2 ])
# #model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
# #model.add(SpatialDropout1D(0.5))
# #model.add(LSTM(100, dropout=0.6))
# #model.add(Dense(39,activation='sigmoid'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history = model.fit(X, {  'cont_out':op1, 'cat_out': op2}, epochs=10)

# In[446]:


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=250, name='input'))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.6, recurrent_dropout=0.2))
model.add(Dense(10, activation='softmax', name='op40'))


# In[447]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 7
batch_size = 16

history = model.fit(X, op40, epochs=epochs, batch_size=batch_size )


# In[449]:



new_tweet="""The S&P 500 climbed 53.67 points, or 1.23%, at 4,426.26 points. The Dow Jones Industrial Average rose 430.31 points, or 1.27%, at 34,415.46. The Nasdaq Composite gained 156.34 points, or 1.15%, at 13,782.82.

This Huge Jet Is Only Used By One Celebrity. Can You Guess Who?
investing.com
|
Sponsored 
Forget Expensive Solar Panels (Do This Instead)
Quote Wallet
|
Sponsored 
New issue of sovereign gold bond scheme to open next week. Key things to know
Live Mint
Use This Military Invention For 20 Minutes Per Day And Watch Your Stomach Shrink
Tactical X
|
Sponsored 
IKIO Lighting IPO: GMP signals strong debut of shares on listing date
Live Mint
Look For Any High School Yearbook, It's Free
Classmates
|
Sponsored 
The Serum That Takes Seconds To Apply, But Takes Years Off
Vibriance
|
Sponsored 
Salinas: These Unsold SUVs Now Almost Being Given Away (See Deals)
TopAnswersToday
|
Sponsored 
  by Taboola 
A report showed that US retail sales unexpectedly rose in May.

Another report showed that jobless claims were flat for the week ended June 10.

On Wednesday, Federal Reserve left interest rates unchanged at the 5%-5.25% range and indicated it may hike by at least half a percentage point this year to tame inflation. 

Microsoft Corp shares rose to a record high on Thursday on strong optimism about the prospects of artificial intelligence (AI). Its shares closed up 3.2% at $348.10 per share.


The yield on the 10-year treasury fell to 3.72% from 3.79% late Wednesday. The 2-year yield fell to 4.64% from 4.69%.

Currencies
The US dollar rose to 140.33 Japanese yen from 139.72. The euro rose to $1.0951 from $1.0834.

Canada
Canada stocks rose on Thursday as a rally in oil prices boosted energy shares.

The Toronto Stock Exchange's S&P/TSX composite index ended up 12.26 points, or 0.1%, at 20,027.35.

Europe
European shares fell on Thursday after the European Central Bank (ECB) increased its key rates.

The pan-European STOXX 600 index closed 0.1% lower after falling as much as 0.8% earlier in the day.

Britain’s FTSE 100 up 0.3% at 7,628.26. Germany’s Frankfurt DAX fell 0.1% at 16,290.12. France’s CAC 40 declined 0.5% at 7,290.91. 


Asia
Asian stocks surged on Thursday after the Federal Reserve paused monetary tightening and China’s central bank cut a key lending rate.

Hong Kong’s Hang Seng index up 2.2% at 19,828.92. China’s Shanghai Composite added 0.7% at 3,252.98.

Japan’s Tokyo - Nikkei 225 fell 0.1% at 33,485.49.

Australia’s S&P/ASX 200 index advanced 0.2% at 7,175.3. New Zealand's benchmark S&P/NZX 50 index rose 0.1% at 11,687.45.

 

Bullion
Gold for August delivery rose $1.80 to $1,970.70 an ounce. Silver for July delivery fell 16 cents to $23.95 an ounce.

Energy prices
Brent crude for August delivery rose $2.47 to $75.67 a barrel on Thursday. US crude oil benchmark for July delivery rose $2.35 to $70.62 a barrel.

Natural gas for July delivery rose 19 cents to $2.53 per 1,000 cubic feet.


KNOW YOUR INNER INVESTOR
Do you have the nerves of steel or do you get insomniac over your investments? Let’s define your investment approach.
TAKE THE TEST
Catch all the Business News, Market News, Breaking News Events and Latest News Updates on Live Mint. Download The Mint News App to get Daily Market Updates.
More
Updated: 16 Jun 2023, 02:14 AM IST
Topics
global market


Let us bring the summary of the day’s most important news directly to you with our newsletters!
Subscribe for free
You May Like
""".lower()
lst= finalpreprocess(new_tweet)
MAX_SEQUENCE_LENGTH=250
seq = tokenizer.texts_to_sequences(lst)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
# Add list of labels here labels=['']


# In[458]:


newlst=[]
def Average(lst):
    return sum(lst) / len(lst)
for i in pred:
    
    newlst.append(Average(i))
labels=[]
for i1 in top1000lst:
    labels.append(i1)
import operator 
zipped=list(zip(labels, newlst))
l=sorted(zipped,key = lambda x: x[1],  reverse=True)[:5]

print(l[0][0],'\n',l[1][0],'\n',l[2][0],'\n',l[3][0],'\n',l[4][0],)


# In[460]:



new_tweet="""SKIP NAVIGATION
logo

logo
MARKETS
BUSINESS
INVESTING
TECH
POLITICS
CNBC TV
INVESTING CLUB
PRO
MAKE IT
SELECT











USA
INTL
WATCH
LIVE
Search quotes, news & videos
WATCHLIST
SIGN IN
CREATE FREE ACCOUNT
U.S. inflation could return to 2.5% or 3% by year-end, advisory firm saysWATCH NOW
SHARE
Share Article via Facebook
Share Article via Twitter
Share Article via LinkedIn
Share Article via Email
SQUAWK BOX EUROPE
U.S. inflation could return to 2.5% or 3% by year-end, advisory firm says
Stephen Isaacs, chairman of the investment committee at Alvine Capital, says inflation is trending down and markets are focused more on this than on the Federal Reserve’s interest rate policy.
FRI, JUN 16 20236:27 AM EDT
TOP VIDEOS
WATCH NOW
You've seen the low water mark for retail sales, says JPMorgan's Matthew Boss
WATCH NOW
VIDEO03:58
You’ve seen the low water mark for retail sales, says JPMorgan’s Matthew Boss
WATCH NOW
I'm not sold the Federal Reserve will hike again, says SoFi's Liz Young
WATCH NOW
VIDEO03:44
I’m not sold the Federal Reserve will hike again, says SoFi’s Liz Young
WATCH NOW
BlackRock reportedly close to filing Bitcoin ETF application
WATCH NOW
VIDEO03:59
BlackRock reportedly close to filing Bitcoin ETF application
WATCH NOW
USGA CEO Mike Whan: I was 'surprised’ by PGA Tour-LIV Golf merger announcement
WATCH NOW
VIDEO04:09
USGA CEO Mike Whan: I was ‘surprised’ by PGA Tour-LIV Golf merger announcement
WATCH NOW
China expert Dennis Unkovic explains why Xi Jinping wants to meet Bill Gates
WATCH NOW
VIDEO04:01
China expert Dennis Unkovic explains why Xi Jinping wants to meet Bill Gates
LOAD MORE
logo
Subscribe to CNBC PRO
Licensing & Reprints
CNBC Councils
Select Personal Finance
CNBC on Peacock
Join the CNBC Panel
Supply Chain Values
Select Shopping
Closed Captioning
Digital Products
News Releases
Internships
Corrections
About CNBC
Ad Choices
Site Map
Podcasts
Careers
Help
Contact
News Tips
Got a confidential news tip? We want to hear from you.

GET IN TOUCH
Advertise With Us
PLEASE CONTACT US
CNBC Newsletters
Sign up for free newsletters and get more CNBC delivered to your inbox

SIGN UP NOW
Get this delivered to your inbox, and more info about our products and services. 

Privacy Policy
|
Do Not Sell My Personal Information
|
CA Notice
|
Terms of Service
© 2023 CNBC LLC. All Rights Reserved. A Division of NBCUniversal

Data is a real-time snapshot *Data is delayed at least 15 minutes. Global Business and Financial News, Stock Quotes, and Market Data and Analysis.

Market Data Terms of Use and Disclaimers
Data also provided by Reuters""".lower()
#new_tweet=st.text_input("Enter Label here")
lst= finalpreprocess(new_tweet)
seq = tokenizer.texts_to_sequences(lst)
padded = pad_sequences(seq, maxlen=250)
pred = model.predict(padded)
newlst=[]
def Average(lst):
    return sum(lst) / len(lst)
for i in pred:
    
    newlst.append(Average(i))
labels=[]
for i1 in top1000lst:
    labels.append(i1)
import operator 
zipped=list(zip(labels, newlst))
l=sorted(zipped,key = lambda x: x[1],  reverse=True)[:5]

print(l[0][0],'\n',l[1][0],'\n',l[2][0],'\n',l[3][0],'\n',l[4][0],)


# In[459]:


v=input("Enter URL here:")
page = requests.get(v)
html = page.content
soup = BeautifulSoup(html, "lxml")
text = soup.get_text()
d2=soup.find_all("p")
            #for a in d2:
new_tweet1=re.sub(r'<.+?>',r'',str(d2)) 
new_tweet=new_tweet1.lower()
#new_tweet=st.text_input("Enter Label here")
lst= finalpreprocess(new_tweet)
seq = tokenizer.texts_to_sequences(lst)
padded = pad_sequences(seq, maxlen=250)
pred = model.predict(padded)
newlst=[]
for i in pred:
    
    newlst.append(Average(i))
labels=[]
for i1 in top1000lst:
    labels.append(i1)
import operator 
zipped=list(zip(labels, newlst))
l=sorted(zipped,key = lambda x: x[1],  reverse=True)[:5]

print(l[0][0],'\n',l[1][0],'\n',l[2][0],'\n',l[3][0],'\n',l[4][0],)


# In[411]:


import warnings
warnings.filterwarnings("ignore")
def app_layout():
    st.title("Running news")
    
if __name__=='__main__':
    app_layout()


# In[416]:


get_ipython().system('streamlit run strm.py')


# In[ ]:




