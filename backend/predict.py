import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
import pickle
from generateTrainingMatrix import do_the_job
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], str(int(y[i]))+'%', ha = 'center')


def predict():
    csv_input = pd.read_csv('./static/realtime_csv.csv')
    csv_input['Right AUX'] = np.random.randint(-100, 100, csv_input.shape[0])
    csv_input.to_csv('./static/regen.csv', index=False)
    expanded,header=do_the_job('./static/regen.csv')
    header=header.split(',')
    # print(expanded.shape,len(header))
    data=pd.DataFrame(expanded,columns=header)
    data.drop(['label'],axis=1,inplace=True)
    # print(data.shape,data.columns)
    # print(keras.__version__)
    # print(os.listdir())
    # print(os.path.isfile('model.keras'))
    
    model = Sequential()
    model.add(LSTM(units=256, input_shape=(1, 2548)))
    model.add(Dense(units=3, activation='softmax'))
    model.load_weights('model_weights.weights.h5')


    with open('labelEncoder.pickle','rb') as f:
        label=pickle.load(f)
        
    # data = pd.read_csv('./static/realtime_csv.csv')
    

    data.replace([np.inf, -np.inf], 0, inplace=True)
    X_test=data
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    y_pred = model.predict(X_test)
    y_pred=np.array(list(map(lambda x: np.argmax(x), y_pred)))
    y_pred=label.inverse_transform(y_pred)
    # print(y_pred)
    d={}
    for i in y_pred:
        if i not in d:
            d[i]=1
        else:
            d[i]+=1
    # print(d)
    total=sum(list(d.values()))
    for i in d:
        d[i]=d[i]/total*100
    # print(", ".join([f"{i}: {round(d[i],2)}%" for i in d]))
    plt.bar(range(len(d)), list(d.values()), align='center',color=['blue','green','red'])
    addlabels(range(len(d)), list(d.values()))

    plt.xticks(range(len(d)), ['Happiness','Neutral','Sadness'])

    plt.savefig('../src/components/mood analysis')
if __name__=="__main__":
    predict()