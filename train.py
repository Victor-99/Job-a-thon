import numpy as np
import pandas as pd
import pickle

import imblearn as imbl

from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder

from preprocess import *


def main():
    csv=pd.read_csv(r'./Data/train_Df64byy.csv')
    csv.set_index('ID',inplace=True)
    print(csv.head())
    y=csv["Response"]
    x=csv.drop("Response",axis=1)
    preprocess_csv=preprocess(x)
    print(preprocess_csv.head)
    print(preprocess_csv.info())
    print(preprocess_csv.describe())

    oversample=imbl.over_sampling.SMOTE()
    x_data,y_data=oversample.fit_resample(preprocess_csv,y)

    x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,shuffle=True)
    print('Training_data shape: {} {}'.format(x_train.shape,y_train.shape))


    cat_cols=['City_Code','Accomodation_Type','Reco_Insurance_Type','Reco_Policy_Cat']
    ord_cols=['Health Indicator','Holding_Policy_Duration']
    reg_cols=['Reco_Policy_Premium']

    pipeline=full_pipe=ColumnTransformer([
                    ('Categorical_columns',OneHotEncoder(sparse=False),cat_cols),
                    ('Ordinal_columns',OrdinalEncoder(),ord_cols),
                    ('Regression_columns',StandardScaler(),reg_cols)
    ])

    print(x_train.head())
    print(x_test.head())
    x_train=pipeline.fit_transform(x_train)
    print("Saving Pipeline")
    pickle.dump(pipeline,open('./Model/pipeline.pkl','wb'))
    x_test=pipeline.transform(x_test)

    model=CatBoostClassifier(silent=True)
    print("Fitting model:")
    model.fit(x_train,y_train)
    print("Model trained")
    print("Generating roc scores:")
    y_pred=model.predict(x_test)
    print("Roc score: {}".format(roc_auc_score(y_test,y_pred)))
    print("Saving model")
    pickle.dump(model,open('./Model/trained_model.pkl','wb'))
    print("Model Saved")


if __name__=='__main__':

    main()
