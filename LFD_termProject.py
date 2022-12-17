from sklearn.linear_model import BayesianRidge, Ridge, Lasso, LogisticRegression 
from sklearn.multioutput import MultiOutputRegressor

from sklearn.neighbors import LocalOutlierFactor

from sklearn.feature_selection import GenericUnivariateSelect, RFE
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA


from sklearn.metrics import mean_squared_error as mse

import pickle
import numpy as np
import pandas as pd
import random as r

r.seed(1)

import warnings
warnings.filterwarnings("ignore")

# import xgboost for learning model

class Regression:
    def __init__(self):
        # Defining BayesianRidge Regressor for multiple output. All hyperparameters are default.
        self.models = [BayesianRidge(), Ridge(), Lasso()]
        
    def loadData(self,X_train_path, y_train_path, x_test_path):
        self.X_train_LR = pd.read_csv(X_train_path)
        self.y_train_HR = pd.read_csv(y_train_path)
        self.X_test_LR = pd.read_csv(x_test_path)

        # Determining outliers in the data set by applying LocalOutlierFactor on the data
    def preprocessData(self):
        self.X_train_LR = pd.get_dummies(self.X_train_LR)
        clf = LocalOutlierFactor(n_neighbors=2)
        outX_train, outy_train = clf.fit_predict(self.X_train_LR), clf.fit_predict(self.y_train_HR)
        self.X_train_LR = np.delete(np.asarray(self.X_train_LR),np.argwhere(outX_train + outy_train != 2),axis=0)
        self.y_train_HR = np.delete(np.asarray(self.y_train_HR),np.argwhere(outX_train + outy_train != 2),axis=0)
        
        # Fit method
    def fit(self):
        for model in self.models:
            print('\033[96m' + "STARTING ..." + '\033[0m')
            print('\033[93m' + "MODEL:" + str(model) + '\033[0m')
            model.fit(self.X_train_LR,self.y_train_HR)
            print('\033[92m' + "FINISHED!" + '\033[0m')
    
    def saveModel(self,out_name):
        pickle.dump(self.models, open(out_name, 'wb'))
        
    def loadModel(self,model_name):
        self.models = pickle.load(open(model_name, 'rb'))
        
        # Writing test predictions to csv
    def outCsv(self,out_file_name, selected_model = 0):
        self.X_test_LR = pd.get_dummies(self.X_test_LR)
        predicted = self.models[selected_model].predict(self.X_test_LR)  #direct output
        meltedDF = pd.DataFrame(predicted).to_numpy().flatten()
        ID_column = []
        for i in range(meltedDF.shape[0]):
            ID_column.append(i) 
        ID_column_frame = pd.DataFrame(ID_column, columns = ['ID'] )
        predicted_frame = pd.DataFrame(meltedDF, columns = ['Predicted'])
        pd.concat([ID_column_frame,predicted_frame],axis=1).to_csv(out_file_name,index=False)


class crossValidation:
    def __init__(self,model):
        self.model = model
        self.kf = KFold(n_splits=5, shuffle=True, random_state=1)
        self.mse = []
        #self.transformer = PCA(n_components=10) #n_components = min(num of samples, num of features)
        #self.transformer.fit(pd.get_dummies(pd.read_csv("train_features.csv")).values)
        # PCA method for data Preprocessing
    def PCA(self,X_train,X_test):
        return self.transformer.transform(X_train),self.transformer.transform(X_test)       
        
        # 5 fold cv
    def cv5(self,X,y):
        X = pd.get_dummies(X).values
        for train_index,test_index in self.kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            #X_train_, X_test_ = self.PCA(X_train,X_test)

            reg = self.model
            reg.fit( X_train,y_train)
            
            predictions = reg.predict(X_test)
            
            self.mse.append(mse(predictions,y_test))
        print(str(model),": Mean MSE Score:", np.average(self.mse))


if __name__ == "__main__":
    
    Regressor = Regression()
    Regressor.loadData("train_features.csv","train_targets.csv","test_features.csv")
    Regressor.preprocessData()

    # Training Model
    Regressor.fit()
    #Regressor.saveModel('BayesianRidge_regressor.pkl')
    
    # Loading Model
    #Regressor.loadModel('BayesianRidge_regressor.pkl')
    
    # Write predictions
    Regressor.outCsv("test.csv")

    for model in Regressor.models:
        cv = crossValidation(model)
        cv.cv5(pd.read_csv("train_features.csv"),pd.read_csv("train_targets.csv").values)

