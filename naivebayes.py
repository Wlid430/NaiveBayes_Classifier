import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import operator


def Naive_Bayes(Testgroup):
    # Importing dataset
    Dataset = pd.read_csv(f"./Test_Group{Testgroup}.csv")

    #import training dataset
    #training_Data = pd.read_csv("./Training_data1.csv")

    #testing_Data = pd.read_csv("./Test_Data1.csv")

    #Split Data in training and testing data
    train, test = train_test_split(Dataset, test_size=0.2)

    # Convert categorical variable to numeric
    #data["Typecleaned"]=np.where(data["Type"]=="Laptop",0,1)

    # Cleaning training dataset of NaN
    training_Data_Clean = train[[
        #-----
        "PurchaseID",
        "EmployeeID",
        "LoanItemID"
    ]].dropna(axis=0, how='any')

    # Cleaning testing dataset of NaN
    testing_Data_Clean = test[[
        #-----
        "PurchaseID",
        "EmployeeID",
        "LoanItemID"
    ]].dropna(axis=0, how='any')


    # Instantiate the classifier
    gnb = GaussianNB()
    used_features =[
        #-----
        #"PurchaseID"
       # "EmployeeID",
        "LoanItemID",
        #"ID",
        #"Brand",
        #"CPUrating",
        #"RAM",
        #"GPUrating"	

    ]

    # Train classifier
    gnb.fit(
        training_Data_Clean[used_features].values,
        training_Data_Clean["LoanItemID"]
    )

    #prediction algorithm testing data
    item_Predictions = gnb.predict(test[used_features])

    print(test)
    # Print results
    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
          .format(
              testing_Data_Clean.shape[0],
              (testing_Data_Clean["LoanItemID"] != item_Predictions).sum(),
              100*(1-(testing_Data_Clean["LoanItemID"] != item_Predictions).sum()/testing_Data_Clean.shape[0])
    ))


    
    ### Object serialization

    ## Importing dataset2
    #data2 = pd.read_csv("../resources/products.csv")
    #data2.rename(columns={'ID':'LoanItemID'}, inplace=True)

    ## Importing dataset3
    ## Convert categorical variable to numeric
    ##ata["Typecleaned"]=np.where(data["Type"]=="Laptop",0,1)

    ##importing dataset3
    #data3 = pd.read_csv("../resources/profiles.csv")
    #data3.rename(columns={'Id':'LoanItemID'}, inplace=True)

    recommendation = gnb.predict(testing_Data_Clean[used_features]).tolist()


    rating = {}
    for item in recommendation:
        occur = recommendation.count(item)
        rating[item] = int(occur)

    return rating



    print(recommendation)


Naive_Bayes(2)