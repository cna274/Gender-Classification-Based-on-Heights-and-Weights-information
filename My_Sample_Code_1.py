
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd


# In[ ]:

'''Use this class to pre process the humungous ugly data to machine learning algorithm understandable matrix form and
meaningfull. Output will be of form n_samples,n_features plus the labels. Following things can be done using this 
class. Add missing values, standardise the data, binarize the data, output meaningfull label column'''
       
def reading_data(data_path, data_format='csv'):
    
    if data_format == 'csv':
        df = pd.read_csv(data_path)
    elif data_format == 'tsv':
        df = pd.read_csv(data_path, sep='\t')
    return df
    
def changing_dtypes(df,dtype='integer'):
    df = pd.to_numeric(df,errors = 'coerce', downcast = dtype)     
    #print (df)
    return df

def fill_with(df,col_name,col_value = 'numbers', fill_with = 'mean'):
    miss_value = 'n'
    if col_value == 'numbers':
        for values in df[col_name].unique():
            #print (values)
            try:
                dummy = int(values)
                #print (dummy)
            except:
                miss_value = values
                #print ('Hi')
        print (miss_value)
        if fill_with == 'mean':
            mean_value = np.mean(int(df[col_name][df[col_name] != miss_value]))
            df[col_name][df[col_name] == miss_value] = mean_value

        if fill_with == 'median':
            mean_value = np.median(df[col_name][df[col_name] != miss_value])
            df[col_name][df[col_name] == miss_value] = mean_value

    else:
        values = df[col_name].unique()
        miss_value = list(map(lambda s: s if not s in values else None, col_value))
        print (miss_value)
        keys = list(map(lambda s: df[col_name][df[col_name]== s].count(),df[col_name].unique()))
        fill_value = values[keys.index(max(keys))]
        df[col_name][df[col_name] == miss_value] = fill_value
    return df


# In[ ]:

'''ML algorithms will have high dimensional feature vectors, its hard to visualize the data, in high dimension 
as you cannot plot them, hence dimensionality reduction. This class reduce, high dimension to specified dimension
This reduced dimension data can be used to plot and run ML algorithm. Auditing the data gives some insight 
about the dataset, Plotting serves this requirement'''

from sklearn.decomposition import PCA
def principal_component(df,dimen = 2):
    pca = PCA(n_components= dimen)
    new_values = pca.fit_transform(df)
    return new_values


# In[ ]:

'''Heart of problem, Now you have nice looking dataset in matrix form, lets get some meaning out of it by running 
algorithm. here scikit-learn module is used to run the algorithm. Output can be either regression or classification
various hyper parameter are used as input while instansiating the algorithm, user has to know what algorithm to run, 
based on the requirement and type of dataset'''

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
def cv_optimize(clf, parameters, train_data, train_label, n_folds=5):
    gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(train_data, train_label)
    print ("BEST PARAMS", gs.best_params_)
    best = gs.best_estimator_
    return best
    
def do_classify(clf, parameters, indf, featurenames, targetname, standardize=False, train_size=0.8):
    subdf=indf[featurenames]
    if standardize:
        subdfstd=(subdf - subdf.mean())/subdf.std()
    else:
        subdfstd=subdf
    X=subdfstd.values
    y=(indf[targetname].values)
    train_data, test_data, train_label,test_label = train_test_split(X, y, train_size=train_size)
    clf = cv_optimize(clf, parameters, train_data, train_label)
    clf=clf.fit(train_data, train_label)
    training_accuracy = clf.score(train_data, train_label)
    test_accuracy = clf.score(test_data, test_label)
    #print (training_accuracy)
    #print (test_accuracy)
    print ("Accuracy on training data: " ,training_accuracy)
    print ("Accuracy on test data:     " , test_accuracy)
    return clf, train_data, test_data, train_label,test_label


# In[ ]:

'''Okay, Now you ran an algorithm. whats next ?. lets evaluate how well does our model works on unseen data. This 
class, outputs the mentioned metrics. which helps developer to understand and improve the trained model. 
Some of them will be confusion marix, accuracy, f1 score, precision and recall '''

def confusion_matrix(clf,test_label,test_data):
    from sklearn.metrics import confusion_matrix
    print (confusion_matrix(test_label, clf.predict(test_data)))
    
def accuracy(clf,test_label,test_data):
    from sklearn.metrics import accuracy_score
    print (accuracy_score(test_label, clf.predict(test_data)))
    
def f1_score(clf,test_label,test_data):
    from sklearn.metrics import f1_score
    print (f1_score(test_label, clf.predict(test_data)))
    
def precision(clf,test_label,test_data):
    from sklearn.metrics import precision_score
    print (precision_score(test_label, clf.predict(test_data)))
    
def recall(clf,test_label,test_data):
    from sklearn.metrics import recall_score
    print (recall_score(test_label, clf.predict(test_data)))


# In[ ]:

#np.shape(ytrain)
recall(clf=bestcv,test_data=ytrain,test_label=ytest)


# In[ ]:

'''This method is used, to get the accuracy from the predicted model. Technique is called 'cross validation'
One more metric evaluation method'''
from sklearn.metrics import accuracy_score
def cv_score(clf, x, y, score_func=accuracy_score):
    result = 0
    nfold = 5
    for train, test in KFold(y.size, nfold):
        clf.fit(x[train], y[train]) 
        result += score_func(clf.predict(x[test]), y[test]) 
    return result / nfold # average
        


# In[ ]:

df=reading_data("C:/Users/ssk150530/Desktop/python/01_heights_weights_genders.csv")
df.head()


# In[ ]:

df.Height.dtype


# In[ ]:

df['Gender'] = list(map(lambda x: 1 if x == 'Male' else 2, df['Gender']))


# In[ ]:

principal_component(df[['Height','Weight']],dimen=1)


# In[ ]:

df['Gender']


# In[ ]:

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:

bestcv, Xtrain, ytrain, Xtest, ytest = do_classify(KNeighborsClassifier(), 
                                                   {"n_neighbors": np.arange(1,40,2)}, 
                                                   df, ['Height','Weight'], 
                                                   'Gender', 'check' )


# In[ ]:

test_label = list(map(lambda x: 1 if x == 'Male' else 2, ytest))
#test_label


# In[ ]:

bestcv


# In[ ]:

precision(clf=bestcv,test_data=ytrain,test_label=ytest)

recall(clf=bestcv,test_data=ytrain,test_label=ytest)

confusion_matrix(clf=bestcv,test_data=ytrain,test_label=ytest)

accuracy(clf=bestcv,test_data=ytrain,test_label=ytest)

f1_score(clf=bestcv,test_data=ytrain,test_label=ytest)


# In[ ]:



