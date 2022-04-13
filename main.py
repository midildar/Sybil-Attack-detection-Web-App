from typing import Final
from flask import Flask , render_template , request , url_for , redirect , flash
from flask_sqlalchemy import SQLAlchemy
from matplotlib.pyplot import text
from numpy import average
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin , LoginManager, login_manager , login_user ,login_required , logout_user , current_user
import tweepy , os 
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
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
import networkx as nx
plt.style.use('fivethirtyeight')

"""Importing Pickle Files"""

adaModel = pickle.load(open("Model_Folder/SMOTE/AdaModel3.pkl", 'rb'))
mlpModel = pickle.load(open("Model_Folder/SMOTE/Multilayer_perceptron_classifier3.pkl", 'rb'))
knnModel= pickle.load(open("Model_Folder/SMOTE/knn3.pkl", 'rb'))
lrModel = pickle.load(open("Model_Folder/SMOTE/model_LogisticRegression3.pkl", 'rb'))
nbModel = pickle.load(open("Model_Folder/SMOTE/naive_bayes_classifier3.pkl", 'rb'))
svcModel=pickle.load(open("Model_Folder/SMOTE/model_SVC_classifier_kernel3.pkl",'rb'))
rfModel = pickle.load(open("Model_Folder/SMOTE/rf3.pkl", 'rb'))
scaler= pickle.load(open("Model_Folder/scaler.pkl", 'rb'))


imageFolder = os.path.join('static', 'images')
app = Flask(__name__)
app.config.from_pyfile('config.cfg')
db = SQLAlchemy(app)
login_manager = LoginManager(app)
mail = Mail(app)
safe_url= URLSafeTimedSerializer('secretkey')
login_manager.login_view = 'login'
app.config['images'] = imageFolder
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


Api_Key = "input your key "  # consumer key
Api_Key_Secret = "input your key"  # consumer secret
Access_Token = "input your key"
Access_Token_Secret = "input your key"
Bearer_Token="input your key"
auth = tweepy.OAuthHandler(Api_Key, Api_Key_Secret)
auth.set_access_token(Access_Token, Access_Token_Secret)
api = tweepy.API(auth)
api1=tweepy.Client(bearer_token=Bearer_Token, consumer_key=Api_Key, consumer_secret=Api_Key_Secret, access_token=Access_Token, access_token_secret=Access_Token_Secret, wait_on_rate_limit=False)

reslist = list()
result = dict()

class User(UserMixin, db.Model):
   id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
   email = db.Column(db.String(30), unique=True, nullable=False)
   password = db.Column(db.String(30), nullable=False)
   name = db.Column(db.String(50), nullable=False)
   status = db.Column(db.Boolean , default=False)


@login_manager.user_loader
def load_user(user_id):
   return User.query.get(int(user_id))

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

@app.route('/login')
def login():
   
    return render_template('login.html')

@app.route('/signup')
def register():

   return render_template('register.html')

@app.route('/home')
def index():

   return render_template('home.html')

@app.route('/')
def initial():

   return render_template('home.html')

@app.route('/dashboard')
@login_required
def dashboard():

   return render_template('index.html',names  = current_user.name)


@app.route('/dashboard', methods=['post'])
@login_required
def analyze():

   userid = request.form.get('twittername')
   print(userid)

   try:
      user = api.get_user(id=userid)
      import math
      req=api1.get_users_followers(id=api.get_user(screen_name =userid).id)#1035247680) #apnay followers dekhnay hoon
      #print(req.meta['result_count'])
      sybilDetection=[0,0,0]
      if(req.meta['result_count']!=0):
         resdf=generateResult(userid,req)
         FINAL=resdf['Result'][0]
         print(FINAL)
         image3=os.path.join(app.config['images'],"Graph1.png")
         Genuine=len(resdf[resdf['Color']=='green'].index.tolist())
         Fake=len(resdf[resdf['Color']=='yellow'].index.tolist())
         Spambot=len(resdf[resdf['Color']=='red'].index.tolist())
         sybilDetection=[Fake,Genuine,Spambot]
      else:
         FINAL=math.ceil(generateResult1(userid))
         image3=os.path.join(app.config['images'],"defaultGraph1.png")
      if FINAL>1:
         FINAL1='Fake'
      li= additionalFeature(userid)
      #reslist = li[0]
      result = li[0]
      tweetRes=li[1]

      FINAL = int(((FINAL - 0) * 100) )/(2-0)
      print(f'Checking Result of Model:{FINAL}')
      FINAL1 = "Genuine"

      if(FINAL>=50 and FINAL<=70):
         FINAL1='In-active User'
      if(FINAL>70):
         FINAL1='Fake'

      

      if type(result)!=str:
         if result["Celebrity"] != None and len(result["Celebrity"]) != 0:   
            if user.verified==True:
               CELEB= result["Celebrity"]
            else:
               CELEB= f"Fan Account (Celebrity: {result['Celebrity']}"
         else:
            CELEB= 'NOT A CELEBRITY'
         
         if result["Labels"] != None and len(result["Labels"]) != 0:
            FACED= result["Labels"]
         else:
            FACED= 'Face Not Detected'
         image1 = os.path.join(app.config['images'],"profileImage.png")
      else:
         CELEB='Not Available'
         FACED="Not Available"
         image1=os.path.join(app.config['images'],"defaultImage.jpg")
      print (reslist)
      print (result)
      image2 = os.path.join(app.config['images'],"WordCloud.png")
         
      return render_template('index.html',names  = current_user.name,sybil=sybilDetection,IMAGE1=image1,IMAGE2=image2,IMAGE3=image3,FACED=FACED,TWEET=tweetRes,nameescreen=user.screen_name, namee =user.name ,FINAL=FINAL ,FINAL1 = FINAL1, CELEB=CELEB)
   except Exception as e:
      print(e)
      flash('Inavlid Username')
      return redirect(url_for('dashboard'))
   





   


@app.route('/signup', methods=['post'])
def user_signup():
   # code to validate and add user to database goes here
   email = request.form.get('email')
   name = request.form.get('username')
   password = request.form.get('pass')

   user = User.query.filter_by(email=email).first() # if this returns a user, then the email already exists in database

   if user: # if a user is found, we want to redirect back to signup page so user can try again
      flash('Email Address already exists')
      return redirect(url_for('register'))

   user = User.query.filter_by(name=name).first() # if this returns a user, then the email already exists in database

   if user: # if a user is found, we want to redirect back to signup page so user can try again
      flash('Username already exists')
      return redirect(url_for('register'))
   
   # create a new user with the form data. Hash the password so the plaintext version isn't saved.
   new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))

   Confirmation_token(email)

   # add the new user to the database
   db.session.add(new_user)
   db.session.commit()
   flash('Email verification link has been sent to your email. Kindly verify your email')
   return render_template('login.html')

def Confirmation_token(email):
   token = safe_url.dumps(email, salt='email-confirm')
   msg = Message('Confirm Email', sender='sybildetection@gmail.com', recipients=[email])
   link = url_for('confirm_email', token=token, _external=True)
   msg.body = 'Your link is {}'.format(link)
   mail.send(msg)

@app.route('/confirm_email/<token>')
def confirm_email(token): 
    try:
        email=safe_url.loads(token, salt='email-confirm', max_age=600)
        user= User.query.filter_by(email=email).first()

        if user: # if a user is found, we want to redirect back to signup page so user can try again
            user.status=True
            db.session.commit()
            flash('User Email verification completed.')
            return redirect(url_for('dashboard'))
    except SignatureExpired:
        return '<h1>The token is expired. Try Again !</h1>'
   


@app.route('/login', methods=['post'])
def user_login():

   
   name = request.form.get('username')
   password = request.form.get('pass')
   remember = request.form.get('remember')
   user = User.query.filter_by(name=name).first()

   # check if the user actually exists
   # take the user-supplied password, hash it, and compare it to the hashed password in the database
   if not user or not check_password_hash(user.password, password):
      flash('Please check your login details and try again.')
      return redirect(url_for('login')) # if the user doesn't exist or password is wrong, reload the page

   # if the above check passes, then we know the user has the right credentials
   login_user(user, remember=remember)
   return redirect(url_for('dashboard'))

@app.route('/login_validated' , methods=['POST'])
def login_validation():

   username = request.form.get('twittername')
   password = request.form.get('pass')

   return "The email {} and the password {}".format(username,password)

@app.route('/profile')
@login_required
def profile():
   return render_template('profile.html', names  = current_user.name)

@app.route('/logout')
@login_required
def logout():
   logout_user()
   return redirect(url_for('login'))








def generateResult1(username):
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

    

    user = {
      'statuses_count': '',
      'followers_count': '',
      'friends_count': '',
      'favorite_count': '',
      'listed_count': '',
      'url': '',
      'geo_enabled': '',
      'description':' '}

    def getUserInfo(user_id):
        userInfo = api.get_user(id=user_id)     
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

 
    #from sklearn.metrics import accuracy_score
    #"""Predict Models"""
    #def checkAccount(username):
    df=getUserInfo(username)#DRrealumarriaz yaslamalj lokeshr61437607 INeerajKSIndian
    df=process(dataset=df,scaler=scaler)
    nbRes=nbModel.predict(df)
    #mlpRes=mlpModel.predict(df)
    #svcRes=svcModel.predict(df)
    lrRes=lrModel.predict(df)
    #adaRes=adaModel.predict(df)
    #rfRes=rfModel.predict(df)
    KnnRes=knnModel.predict(df)
    print(nbRes+lrRes+KnnRes)
    return((nbRes+lrRes+KnnRes)/3)

def generateResult(username,req):
    user=api.get_user(screen_name =username).id
    #user="1035247680"#user_list = ["1035247680"]#5b29cca230e348a 1035247680
    follower_list = []
    name_list=[]
    #req=api1.get_users_followers(id=user)#1035247680) #apnay followers dekhnay hoon
    if(req.data!=None):
      for p in req.data:
         follower_list.append(p.id)
         name_list.append(p.username)
        
    graphdf = pd.DataFrame(columns=['sourceUsername','targetUsername']) #Empty DataFrame
    graphdf['target'] = follower_list #Set the list of followers as the target column
    graphdf['sourceUsername']=api.get_user(id =username).screen_name
    graphdf['source'] = user
    graphdf['targetUsername']=name_list
    print(graphdf)
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
    def getUserInfo(user_id):
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
        userInfo = api.get_user(id=user_id)     
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
    
    def checkAccount(sourceName,targetName):
        li=list(targetName)
        li.insert(0,sourceName)
        colorList=list()
        resList=[]
        for name in li:
            df=getUserInfo(name)#DRrealumarriaz yaslamalj lokeshr61437607 INeerajKSIndian
            df=process(dataset=df,scaler=scaler)
            nbRes=nbModel.predict(df)[0]
            lrRes=lrModel.predict(df)[0]
            KnnRes=knnModel.predict(df)[0]
            temp=(nbRes+lrRes+KnnRes)/3
            if(temp<1):
                colorList.append('green')
            elif(temp<1.5 and temp>=1):
                colorList.append('yellow')
            else:
                colorList.append('red')
            resList.append(temp)#(nbRes+rfRes+KnnRes)/3)
        return resList,colorList
    df=pd.DataFrame()
    df['Result'],df['Color']=checkAccount(graphdf['sourceUsername'][0],graphdf['targetUsername'])
    df.head()
    G = nx.from_pandas_edgelist(graphdf,'sourceUsername','targetUsername') #Turn df into graph
    pos = nx.spring_layout(G) #specify layout for visual
    f, ax = plt.subplots(figsize=(15, 15))
    ax.grid(False)
    plt.style.use('ggplot')
    nodes = nx.draw_networkx_nodes(G, pos,node_color = df.Color)#,width=10,alpha=0.8)
    nodes.set_edgecolor('k')
    #nx.draw(G, node_color = df.Color)
    nx.draw_networkx_labels(G, pos)#,width=5,font_size=10)
    nx.draw_networkx_edges(G, pos)#, width=5.0, alpha=0.2)
    plt.savefig('static/images/Graph1.png')
    return df

def additionalFeature(username):
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
        plt.figure(figsize=(15,15))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('static/images/WordCloud.png')
        #plt.show()
        df['Analysis']=df['polarity'].apply(Analysis)
        countList=df.groupby(['Analysis'])['tweet'].count()
        #print(countList)
        #print(type(countList))
        plt.figure(figsize=(15,15))
        df.groupby(['Analysis'])['tweet'].count().plot(kind="bar") 
        plt.savefig('static/images/Graph2.png')
        countList=countList.tolist()
        return countList
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
        #plt.show()
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img = image.save('static/images/profileImage.png')
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
   userInfo=api.get_user(id=username)
   if userInfo.default_profile_image==False:
        pictureUrl=userInfo.profile_image_url
        if '.jpeg' in pictureUrl:
            pictureUrl=pictureUrl.replace("_normal.jpeg",".jpeg")
        elif '.jpg' in pictureUrl:
            pictureUrl=pictureUrl.replace("_normal.jpg",".jpg")
        elif '.png' in pictureUrl:
            pictureUrl=pictureUrl.replace("_normal.png",".png")
        print(pictureUrl)
        result=checkImage(pictureUrl)
   else:
        print("No Profile Picture")
        result="No Profile Picture"
   
       #resList=checkAccount(username)
       #res=format(sum(resList) / len(resList),".3f")

       #print(resList)
       #res

   countList=checkTweets(username)
    #print(countList)
    #print(result)
   return[result,countList]
    #feature list
   
   #print(result)



if __name__ == '__main__':
   app.run(debug=False)  
