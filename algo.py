from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
import traceback
from string import punctuation
from nltk.data import path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.tree import DecisionTreeClassifier
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import col
from xgboost import XGBClassifier
import tkinter as tk

current_directory = os.path.dirname(os.path.abspath(__file__))
print(current_directory)
os.environ["QT_PLUGIN_PATH"] = os.path.join(current_directory, 'Env\Library\plugins')
path.append(os.path.join(current_directory, 'Dependencies\\nltk_data'))
main = tkinter.Tk()
main.title("Garbage Data Filtering for Social Netwoking Sties")
main.geometry("2000x2000")

accuracy = []
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1 and len(word) < 20]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END, f"{filename} Dataset Loaded\n\n")
    pathlabel.config(text=f"{filename} Dataset Loaded")
    file_path = os.path.join(current_directory, 'Data', 'dataset.csv')
    dataset = pd.read_csv(file_path)
    text.insert(END, str(dataset.head()))
    text.update_idletasks()
    label = dataset.groupby('Label').size()
    label.plot(kind="bar",color='#5B9A8B')
    plt.title("SNS Graph 0 (Garbage), 1 (Advertisement) & 2 (Definite Data)")
    plt.show()

def dataClassifier():
    global dataset, word_meme
    text.delete('1.0', END)
    textdata.clear()
    labels.clear()
    for i in range(len(dataset)):
        msg = dataset.at[i, 'Tweets']
        label = dataset.at[i, 'Label']
        msg = str(msg)
        msg = msg.strip().lower()
        labels.append(label)
        clean = cleanPost(msg)
        textdata.append(clean)
    word_meme = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    weight = word_meme.fit_transform(textdata).toarray()
    df = pd.DataFrame(weight, columns=word_meme.get_feature_names())
    df['target'] = labels
    file_path = os.path.join(current_directory, 'Data', 'weight.csv')
    df.to_csv(file_path, index=False)
    text.insert(END, "Words Morphological Weights\n\n")
    text.insert(END, str(df))

def setupSparkEnvironment():
    # Backup existing environment variables
    # Backup existing environment variables
    backup_env_variables = {
        "HADOOP_HOME": os.environ.get("HADOOP_HOME", ""),
        "SPARK_HOME": os.environ.get("SPARK_HOME", ""),
        "JAVA_HOME": os.environ.get("JAVA_HOME", ""),
    }
    # Update these paths based on your project structure
    subfolder_name = 'Dependencies'
    hadoop_home = "Hadoop"
    spark_home = "spark"
    java_home = "java"

    os.environ["HADOOP_HOME"] = os.path.join(current_directory, subfolder_name, hadoop_home)
    # print(os.environ["HADOOP_HOME"])
    os.environ["SPARK_HOME"] = os.path.join(current_directory, subfolder_name, spark_home)
    # print(os.environ["SPARK_HOME"])
    os.environ["JAVA_HOME"] = os.path.join(current_directory, subfolder_name, java_home)
    # print(os.environ["JAVA_HOME"])
    return backup_env_variables

def naiveBayesTraining():
    text.delete('1.0', END)
    accuracy.clear()
    backup_env_variables = setupSparkEnvironment()
    
    try:
        # create spark object using HDFS hadoop big data processing
        spark = SparkSession.builder.appName("HDFS").getOrCreate()
        sparkcont = SparkContext.getOrCreate(SparkConf().setAppName("HDFS"))
        # logs = sparkcont.setLogLevel("ERROR")
        file_path = os.path.join(current_directory, 'Data', 'weight.csv')
        print(file_path)
        # read dataset weight file
        df = spark.read.option("header", "true").csv("file:///"+file_path, inferSchema=True)
        temp = df.toPandas()
        # extract columns from dataset
        required_features = df.columns
        # convert dataset into spark compatible format
        assembler = VectorAssembler(inputCols=required_features, outputCol='features', handleInvalid="skip")
        transformed_data = assembler.transform(df)
        indexer = StringIndexer(inputCol="target", outputCol="indexlabel", handleInvalid="skip")
        transformed_data = indexer.fit(transformed_data).transform(transformed_data)
        # split dataset into train and test where 0.8 refers to 80% training data 0.2 means 20 testing data
        (training_data, test_data) = transformed_data.randomSplit([0.8, 0.2])
        # training spark naive bayes
        nb = NaiveBayes(modelType="multinomial", featuresCol='features', labelCol='indexlabel')
        nb_model = nb.fit(training_data)
        # predicting on test data
        predictions = nb_model.transform(test_data)
        # calculating accuracy
        evaluator = MulticlassClassificationEvaluator(labelCol='indexlabel', metricName="accuracy")
        acc = evaluator.evaluate(predictions) * 100
        accuracy.append(acc)
        text.insert(END, "Spark Naive Bayes Data Classifier Accuracy : "+str(acc)+"\n\n")
    except Exception as e:
        traceback.print_exc()
        text.insert(END, f"Error: {e}")
    finally:
        # Restore original environment variables
        for var, value in backup_env_variables.items():
            os.environ[var] = value

        # Clean up created HADOOP_HOME if it was not in the original environment
        if "HADOOP_HOME" not in backup_env_variables:
            del os.environ["HADOOP_HOME"]
        if "SPARK_HOME" not in backup_env_variables:
            del os.environ["SPARK_HOME"]
        if "JAVA_HOME" not in backup_env_variables:
            del os.environ["JAVA_HOME"]

def runRandomForest():
    global X_train, X_test, y_train, y_test, rf
    file_path = os.path.join(current_directory, 'Data', 'weight.csv')
    dataset = pd.read_csv(file_path)
    dataset = dataset.values
    X = dataset[:, 0:dataset.shape[1]-1]
    Y = dataset[:, dataset.shape[1]-1]
    print(X)
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test)
    random_acc = accuracy_score(y_test, predict) * 100
    text.insert(END, "Extension Random Forest Classifier Accuracy : "+str(random_acc)+"\n\n")
    accuracy.append(random_acc)

def runDecisionTree():
    global X_train, X_test, y_train, y_test
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    predict = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, predict)* 100
    text.insert(END, "Extension Decision Tree Classifier Accuracy : "+str(dt_acc)+"\n\n")
    accuracy.append(dt_acc)

def runXGBoost():
    global X_train, X_test, y_train, y_test
    xg = XGBClassifier()
    xg.fit(X_train, y_train)
    predict = xg.predict(X_test)
    xg_acc = accuracy_score(y_test, predict) * 100
    text.insert(END, "Extension XGBoost Classifier Accuracy : "+str(xg_acc)+"\n\n")
    accuracy.append(xg_acc)

def dataAnalyzer():
    text.delete('1.0', END)
    global rf, word_meme
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=f"{filename} Dataset Loaded")
    testData = pd.read_csv(filename)
    print(testData)
    for i in range(len(testData)):
        msg = testData.at[i, 'Tweets']
        tweet = str(msg)
        tweet = tweet.strip().lower()
        tweet = cleanPost(tweet)
        testReview = word_meme.transform([tweet]).toarray()
        print(testReview.shape)
        predict = rf.predict(testReview)
        print(predict.shape)
        predict = predict[0]
        if predict == 0:
            text.insert(END, "Tweet = "+str(msg)+"\n")
            text.insert(END, "PREDICTED AS =========> Garbage Tweet\n\n")
        if predict == 1:
            text.insert(END, "Tweet = "+str(msg)+"\n")
            text.insert(END, "PREDICTED AS =========> Advertisement Tweet\n\n")
        if predict == 2:
            text.insert(END, "Tweet = "+str(msg)+"\n")
            text.insert(END, "PREDICTED AS =========> Definite Tweet\n\n")


def graph():
    global accuracy
    height = accuracy
    bars = ('Spark Naive Bayes', 'Extension Random Forest', 'Extension Decision Tree', 'Extension XGBoost')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color='#5B9A8B')
    plt.xticks(y_pos, bars, color='#5B9A8B')
    # plt.plot(kind="bar",color='#5B9A8B')
    plt.title("Algorithms Comparison Graph")
    plt.show()



def close():
    main.destroy()
#Header
font = ('times', 18, 'bold')
title = Label(main, text='Garbage Data Filtering for Social Netwoking Sties')
title.config(bg='#F7E987', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5, y=5)


#buttons
font1 = ('times', 15, 'bold')

uploadButton = Button(main, text="Upload SNS Dataset", command=uploadDataset)
uploadButton.place(x=70, y=200)
uploadButton.config(font=font1)  
uploadButton.config(bg='#F5F5F5')

clsButton = Button(main, text="Data Classifier Generator", command=dataClassifier)
clsButton.place(x=70, y=250)
clsButton.config(font=font1)
clsButton.config(bg='#F5F5F5')

sparkButton = Button(main, text="Data Classifier using Spark Naive Bayes", command=naiveBayesTraining)
sparkButton.place(x=70, y=300)
sparkButton.config(font=font1)
sparkButton.config(bg='#F5F5F5')

rfButton = Button(main, text="Random Forest", command=runRandomForest)
rfButton.place(x=70, y=350)
rfButton.config(font=font1)
rfButton.config(bg='#F5F5F5')

dtButton = Button(main, text="Decision Tree", command=runDecisionTree)
dtButton.place(x=70, y=400)
dtButton.config(font=font1)
dtButton.config(bg='#F5F5F5')

xgboostButton = Button(main, text="XGBoost", command=runXGBoost)
xgboostButton.place(x=70, y=450)
xgboostButton.config(font=font1)
xgboostButton.config(bg='#F5F5F5')

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=70, y=500)
graphButton.config(font=font1)
graphButton.config(bg='#F5F5F5')

analyzerButton = Button(main, text="Data Analyzer", command=dataAnalyzer)
analyzerButton.place(x=70, y=550)
analyzerButton.config(font=font1)
analyzerButton.config(bg='#F5F5F5')

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=70, y=600)
exitButton.config(font=font1)
exitButton.config(bg='#F5F5F5')




#TextBox
font1 = ('times', 14, 'bold')

text=Text(main,height=25,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450, y=150)
text.config(font=font1)
pathlabel = Label(main)

#Upload data text box
pathlabel.config(bg='#1F4172', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=580, y=100)

#main background
main.config(bg='#5B9A8B')
main.mainloop()