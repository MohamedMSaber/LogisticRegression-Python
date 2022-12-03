import pandas as pd
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
import warnings
from sklearn.utils import shuffle
warnings.filterwarnings('ignore')



# Load Data
data = pd.read_csv("customer_data.csv")
#print(data)


# Split Dataset
X = data[['age','salary']]
y = data['purchased']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0, train_size=.75)
#print(X_train,X_test,y_train,y_test)


#Normalization

minValue = X['age'].min()
maxValue = X['age'].max()
X.loc[:,0] = (X.iloc[:,0] - minValue) / (maxValue - minValue)
minValue = X['salary'].min()
maxValue = X['salary'].max()
X.loc[:,1] = (X.iloc[:,1] - minValue) / (maxValue - minValue)
#print(X.head(5))

class logisticRegression():
    #constructor
    def __init__(self , learnRate , itr) :
        self.learnRate = learnRate
        self.itr = itr
        W = None
        b = 0

    #training Model
    def train(self , X_train , Y_train):
        #  m = no of rows  , n = no of cols
        self.m , self.n = X_train.shape #(m = 400, n = 2)

        # weight initialization        
        self.W = np.zeros( self.n )  #return matrix of no of cols(Features)     
        self.b = 0        
        

        # Gradient Descent 
        for _ in range( self.itr) :
            predLinear = (X_train.dot(self.W)+self.b)  # exp = e power x   , X.dot = matrix multi  (320*2) * (2*1) = (320*1) + bias   y =  x0 x1 01 + b
            pred = self.sigmoidFun(predLinear)
    
            # Gradient Calc
            #print(pred.shape , Y_train.shape , X_train.shape)
            dW = ((1/self.m ) * np.dot(X_train.T , (pred - Y_train)) )            # (320*2)=> x_trainT (2*320)  * (320*1) = (2*1)
            db = ((1/self.m)*np.sum(pred - Y_train))    

            # New Weights
            self.W = self.W - self.learnRate * dW    
            self.b = self.b - self.learnRate * db
            
        
    #sigmiod Fun
    def sigmoidFun(self,x):
        return 1/(1+np.exp(-x))
       

        # H Fun
    def predictY(self , X_test):
        predLinear = (X_test.dot(self.W)+self.b)  
        pred = self.sigmoidFun(predLinear)
        Y = np.where(pred >= 0.5 , 1 , 0)
        return Y

    def accuracy(self,Y_pred,Y_test):
        return np.sum(Y_pred==Y_test)/len(Y_test)

    
        

def main():
    

    data = pd.read_csv("customer_data.csv")
    data["age"] = (data["age"] - np.min(data["age"]))/(np.max(data["age"]-np.min(data["age"])))
    data["salary"] = (data["salary"] - np.min(data["salary"]))/(np.max(data["salary"]-np.min(data["salary"])))
    data = data.drop_duplicates()
    data = shuffle(data)

    X = data[['age','salary']]
    Y = data['purchased']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, train_size=0.80)

    model = logisticRegression(learnRate= 0.1 ,   itr= 10000)

    model.train(X_train , Y_train)
    Y_Pre = model.predictY(X_test)
    print (Y_Pre)

    accuracy   = model.accuracy(Y_Pre , Y_test)*100
    print (accuracy)

if __name__ == "__main__" :	
	main()






