import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, r2_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier 
from sklearn.neural_network import MLPRegressor 
from sklearn.tree import plot_tree 
import matplotlib.pyplot as plt 
from matplotlib import pyplot
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.dummy import DummyClassifier 
from numpy import mean
import pickle
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy
import re
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import boto3
plt.style.use('fivethirtyeight')

"""Importing Pickle Files"""

adaModel = pickle.load(open("Model_Folder/Normal/AdaModel.pkl", 'rb'))
mlpModel = pickle.load(open("Model_Folder/Normal/Multilayer_perceptron_classifier.pkl", 'rb'))
knnModel= pickle.load(open("Model_Folder/Normal/knn.pkl", 'rb'))
lrModel = pickle.load(open("Model_Folder/Normal/model_LogisticRegression.pkl", 'rb'))
nbModel = pickle.load(open("Model_Folder/Normal/naive_bayes_classifier.pkl", 'rb'))
svcModel=pickle.load(open("Model_Folder/Normal/model_SVC_classifier_kernel.pkl",'rb'))
rfModel = pickle.load(open("Model_Folder/Normal/rf.pkl", 'rb'))
scaler= pickle.load(open("Model_Folder/scaler.pkl", 'rb'))

"""Process Data"""

def process(dataset,scaler):
    if dataset['url'][0]==None:
        dataset['url']=0

    if dataset['geo_enabled'][0]==False:
        dataset['geo_enabled']=0
    else:
           dataset['geo_enabled']=1
           

    X = dataset.values
    if type(X[0][5])==str:
        X[0][5]=1
    else:
        X[0][5]=0
        
    if type(X[0][7])==str and X[0][7]!="":
        X[0][7]=1
    else:
         X[0][7]=0
        
    
    
    X=X.astype(np.float64)
    where_nans=np.isnan(X)
    X[where_nans]=0
    X=scaler.transform(X)
    return X

"""Configuring Twitter API"""
Api_Key = "0Spus1d0oea3ALOz3urj7Xaqt"  # consumer key
Api_Key_Secret = "8M6XufFUEvXhu4dfjxFI2QbEBRjluhWZ2sgWNIrlUmHV4OcMDs"  # consumer secret
Access_Token = "1035247680-h8IMZayZUmAq03OyKvsFtnporrR5Bz8EoCrzD6k"
Access_Token_Secret = "SwMuZgWRcJqIOvQTPKhBBMRwjZhpFIUD5coWOMvt2RqnC"

auth = tweepy.OAuthHandler(Api_Key, Api_Key_Secret)
auth.set_access_token(Access_Token, Access_Token_Secret)
api = tweepy.API(auth)


user = {
    'statuses_count': '',
    'followers_count': '',
    'friends_count': '',
    'favorite_count': '',
    'listed_count': '',
    'url': '',
    'geo_enabled': '',
    'description':' '
}

def getUserInfo(user_id):

    userInfo = api.get_user(screen_name=user_id)     
    user['statuses_count'] = userInfo.statuses_count
    user['followers_count'] = userInfo.followers_count
    user['friends_count'] = userInfo.friends_count
    user['favorite_count'] = userInfo.favourites_count
    user['listed_count'] = userInfo.listed_count
    user['url'] = userInfo.url
    user['geo_enabled'] = userInfo.geo_enabled
    user['description']=userInfo.description
    user_df = pd.DataFrame(user, index=[0])
    return user_df

"""Data Cleansing"""

def cleanData(text):
  text=re.sub(r"^\s+", "", text)
  text=re.sub(r'@[a-zA-Z0-9]+','',text)#mentions
  text=re.sub(r'#','',text)#remove trends #
  text=re.sub(r'RT[\s]+','',text)#retweet symbol
  text=re.sub(r'https?:\/\/\S+','',text)
  text=re.sub(r'[^A-Za-z0-9 ]+','',text)
  text=text.lower()
  emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
  text=re.sub(emoji, '', text)
  return text

def getSubjectivity(text):  #tells the subjectivity/opion
  return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
  return TextBlob(text).sentiment.polarity

def Analysis(score):
    if score==0:
      return 'Neutral'
    elif score>0:
      return 'Positive'
    else:
      return 'Negative'
from sklearn.metrics import accuracy_score
"""Predict Models"""
def checkAccount(username):
    df=getUserInfo(username)#DRrealumarriaz yaslamalj lokeshr61437607 INeerajKSIndian
    df=process(dataset=df,scaler=scaler)
    nbRes=nbModel.predict(df)
    mlpRes=mlpModel.predict(df)
    svcRes=svcModel.predict(df)
    lrRes=lrModel.predict(df)
    adaRes=adaModel.predict(df)
    rfRes=rfModel.predict(df)
    KnnRes=knnModel.predict(df)
    return[nbRes,mlpRes,svcRes,lrRes,adaRes,rfRes,KnnRes]

"""Checking Tweets of the User"""

def checkTweets(username):
  tweets=api.user_timeline(screen_name=username,count=200,lang='en',tweet_mode='extended')#Sabiha_Baluch ExposeAntiIndia
  tweetList=list()
  for tweet in tweets:#[0:100]:
    tweetList.append(tweet.full_text)
  df=pd.DataFrame(columns={'tweet'})
  df['tweet']=tweetList
  df['tweet']=df['tweet'].apply(cleanData)
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords
  stop = stopwords.words('english')
  df["tweet"]= df["tweet"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  df['tweet'].replace('', np.nan, inplace=True)
  df.dropna(subset=['tweet'], inplace=True)
  df['subjectivity']=df['tweet'].apply(getSubjectivity)
  df['polarity']=df['tweet'].apply(getPolarity)
  from wordcloud import WordCloud
  all_words = ' '.join([text for text in df.tweet])
  wordcloud = WordCloud(width= 800, height= 500,
                        max_font_size = 110,
                        collocations = False).generate(all_words)
  plt.figure(figsize=(10,7))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.savefig('WordCloud.png')
  #plt.show()
  df['Analysis']=df['polarity'].apply(Analysis)
  print(df.groupby(['Analysis'])['tweet'].count())
  df.groupby(['Analysis'])['tweet'].count().plot(kind="bar")
  plt.savefig('Graph.png')
  #plt.show()

"""Checking Image of the User"""

def checkImage(pictureUrl):
  client=boto3.client('rekognition',region_name='us-east-1',aws_access_key_id='AKIAZGPADAE6BEE5PI6X',aws_secret_access_key='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')
  result={'Celebrity':"",
        'Moderation':"",
        'Labels':"",
        'Face':""}
  import urllib.request
  from PIL import Image # $ pip install pillow
  image = Image.open(urllib.request.urlopen(pictureUrl))
  print(image.format, image.mode, image.size)
  print(image.filename)
  import matplotlib.pyplot as plt
  import matplotlib.image as mpimg
  imgplot = plt.imshow(image)
  plt.show()
  import io
  img_byte_arr = io.BytesIO()
  image.save(img_byte_arr, format='JPEG')
  img = image.save('profileImage.jpg')
  img_byte_arr = img_byte_arr.getvalue()
  celebList=[]
  response=client.recognize_celebrities(Image={'Bytes':img_byte_arr})
  for i in response['CelebrityFaces']:
    celebList.append(i['Name'])
  #print(celebList)

  moderationList=[]
  response1=client.detect_moderation_labels(Image={'Bytes':img_byte_arr})
  for i in response1['ModerationLabels']:
    #print(i['Name'])
    moderationList.append(i['Name'])
  #print(moderationList)

  labelList=[]
  response2=client.detect_labels(Image={'Bytes':img_byte_arr},MinConfidence=90)
  for i in response2['Labels']:
    #print(i['Name'])
    labelList.append(i['Name'])
  #print(labelList)
  faceList=[]
  response3=client.detect_faces(Image={'Bytes':img_byte_arr})
  for i in response3['FaceDetails']:
    for j in i['Landmarks']:
      #print(j['Type'])
      faceList.append(j['Type'])
  faceList   
  result['Celebrity']=celebList
  result['Moderation']=moderationList
  result['Labels']=labelList
  result['Face']=faceList
  return result


"""Getting Input of User"""
#username=str(input("Enter A valid Username: "))
userInfo=api.get_user(screen_name=username)
if userInfo.default_profile_image==False:
    pictureUrl=userInfo.profile_image_url
else:
    print("No Profile Picture")
if '.jpeg' in pictureUrl:
  pictureUrl=pictureUrl.replace("_normal.jpeg",".jpeg")
elif '.jpg' in pictureUrl:
  pictureUrl=pictureUrl.replace("_normal.jpg",".jpg")
elif '.png' in pictureUrl:
  pictureUrl=pictureUrl.replace("_normal.png",".png")
print(pictureUrl)
resList=checkAccount(username)
#res=format(sum(resList) / len(resList),".3f")

print(resList)
#res

checkTweets(username)
#feature list
result=checkImage(pictureUrl)
#print(result)