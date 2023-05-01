import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def bunch2df(bunch) :
    '''This function converts a bunch dataset into a pandas dataframe.
    Input :
        bunch : dataset in a bunch format
    Output
        pandas dataframe.
    '''
    list_feature = bunch['feature_names']
    df = pd.DataFrame(data = bunch.data, columns=list_feature)
        
    target_name = 'target'
    if 'target_names' in bunch :
        list_target = bunch['target_names']
        target_name = 'target_names'
        df['target'] = pd.Series(bunch[target_name][category] for category in bunch['target'])
    else :
        list_target = bunch['target']
        df['target'] = pd.Series(bunch['target'])
        
    return df
#-------------------------------------------------------------------------------   

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def class1_binary_percent(arr, round_value=2) :
    '''Computes percentage of positive values from an binary array, means, 
    array containing 2 kind of differents values.
    Input:
        - Array containing binary values
        - round_value : number of digits for the returned percentage; default value is 2
    Output:
        - Returns a percentage
    '''
    nb_0 = pd.DataFrame(data=arr).value_counts().iloc[0]
    nb_1 = pd.DataFrame(data=arr).value_counts().iloc[1]
    return round(100*nb_1/(nb_0+nb_1),round_value)
#-------------------------------------------------------------------------------
    
#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------
def get_X_y(df):
    '''Return 2 arrays : X for features and y for target.
    Target is supposed to be the last column.
    '''
    #Select all columns for independants variables
    x_column = df.columns[:-1].values
    
    # Select column from target
    y_column = df.columns[-1:].values

    # Extract values respectivaly from independants variables and target
    X = df[x_column].values
    y = df[y_column].values
    
    return X,y
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
#-------------------------------------------------------------------------------

def binary_training_and_testing(df, pct_train_size, classifier, target_name0, target_name1, is_verbose=False) :
    '''What is the purpose of this function ?
    '''
    # Extract X and y from dataframe
    X, y = get_X_y(df)
    
    # Split dataset into train and test datasets, with the train fraction sized to pct_train_size
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=pct_train_size, random_state=13, stratify=y)

    # Train classfii
    classifier.fit(X_train, y_train.ravel())
    
    # Predictions for train and test datasets
    #y_train_predict = classifier.predict(X_train)
    #y_test_predict = classifier.predict(X_test)
    
    # Score predictions
    score_train = round(100*classifier.score(X_train, y_train),2)
    score_test  = round(100*classifier.score(X_test, y_test),2)
    
    if is_verbose is True :
        print("Classfier precision for train= {}%".format(score_train))
        print("Classifier precision for test= {}%".format(score_test))
    else:
        pass
    return score_train, score_test
#-------------------------------------------------------------------------------

