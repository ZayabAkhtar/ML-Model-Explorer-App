# Importing Libraries
import pandas as pd
import numpy as np
import streamlit as st 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

# Streamlit App
st.write("""
        # Explore Different ML Models and Datasets
        lets see which one is best....""")

# Selecting Dataset and Classifier
dataset_name=st.sidebar.selectbox(
    "Select Dataset",
    ["Iris", "Breast Cancer", "Wine"]
)

classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ('KNN','SVM','Random Forest')
)

# Loading Dataset
def get_dataset(dataset_name):
    """
    Loads a dataset based on the selected dataset name.
    
    Parameters:
    dataset_name (str): Name of the dataset to load.
    
    Returns:
    X (array): Feature matrix.
    y (array): Target vector.
    """
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X=data.data
    y=data.target
    return X,y

X,y=get_dataset(dataset_name)

# Displaying Dataset Information
st.write('Shape of dataset:',X.shape)
st.write('Number of Classes:',len(np.unique(y)))

# Adding Classifier Parameters
def add_parameter_ui(classifier_name):
    """
    Adds parameters for the selected classifier.
    
    Parameters:
    classifier_name (str): Name of the classifier.
    
    Returns:
    params (dict): Dictionary containing the classifier parameters.
    """
    params=dict()
    if classifier_name == "SVM":
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C
    elif classifier_name == 'KNN':
        K=st.sidebar.slider('K',1,15)
        params['K']=K
    else:
        max_depth=st.sidebar.slider('max_depth',2,15)
        params['max_depth']=max_depth
        n_estimators = st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators
    return params

params= add_parameter_ui(classifier_name)

# Training Classifier
def get_classifier(classifier_name,params):
    """
    Returns a classifier object based on the selected classifier name and parameters.
    
    Parameters:
    classifier_name (str): Name of the classifier.
    params (dict): Dictionary containing the classifier parameters.
    
    Returns:
    clf (object): Classifier object.
    """
    clf=None
    if classifier_name == "SVM":
        clf=SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf=KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],
                                max_depth=params['max_depth'],random_state=1234)
    return clf

clf=get_classifier(classifier_name,params)

# Training and Evaluating Classifier
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

acc=accuracy_score(y_test,y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =',acc)

# Visualizing Dataset with PCA
pca=PCA(2)
X_projected=pca.fit_transform(X)

x1=X_projected[:,0]
x2=X_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,
            c=y,alpha=0.8,
            cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)