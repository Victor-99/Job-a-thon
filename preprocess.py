import pandas as pd
import numpy as np


def replace_binary(data_val,binary_cols=None):
    for col in binary_cols:
        data_val.replace(data_val[col].unique(),[0,1],inplace=True)
    return data_val

def rep_city(data_val):
    city_code=data_val['City_Code'].unique()
    rep_by=[i for i in range(1,len(city_code)+1)]
    data_val['City_Code'].replace(city_code,rep_by,inplace=True)
    return data_val

def rep_mode(data_val):
    cols=['Health Indicator','Holding_Policy_Type','Holding_Policy_Duration']
    data_val["Holding_Policy_Duration"]=data_val["Holding_Policy_Duration"].apply(pd.to_numeric)
    for i in cols:
        mode_value=data_val[i].mode()
        data_val[i].fillna(mode_value[0],inplace=True)
    return data_val


def rep_health(data_val):
    health=data_val['Health Indicator'].unique()
    bar=[x for x in range(len(health),0,-1)]
    data_val['Health Indicator'].replace(health,bar,inplace=True)
    data_val['Holding_Policy_Duration'].replace('14+',15,inplace=True)
    return data_val



def preprocess(csv):
    bin_cols=['Accomodation_Type','Reco_Insurance_Type','Is_Spouse']
    csv_bin=replace_binary(csv,binary_cols=bin_cols)
    csv_rep=rep_city(csv_bin)
    rep_h=rep_health(csv_rep)
    final_data_val=rep_mode(rep_h)

    final_data_val=final_data_val.apply(np.int64)

    return final_data_val


if __name__=='__main__':
    preprocess(csv)
