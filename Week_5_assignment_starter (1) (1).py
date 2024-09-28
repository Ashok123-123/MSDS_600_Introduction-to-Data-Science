#!/usr/bin/env python
# coding: utf-8

# # DS Automation Assignment

# Using our prepared churn data from week 2:
# - use pycaret to find an ML algorithm that performs best on the data
#     - Choose a metric you think is best to use for finding the best model; by default, it is accuracy but it could be AUC, precision, recall, etc. The week 3 FTE has some information on these different metrics.
# - save the model to disk
# - create a Python script/file/module with a function that takes a pandas dataframe as an input and returns the probability of churn for each row in the dataframe
#     - your Python file/function should print out the predictions for new data (new_churn_data.csv)
#     - the true values for the new data are [1, 0, 0, 1, 0] if you're interested
# - test your Python module and function with the new data, new_churn_data.csv
# - write a short summary of the process and results at the end of this notebook
# - upload this Jupyter Notebook and Python file to a Github repository, and turn in a link to the repository in the week 5 assignment dropbox
# 
# *Optional* challenges:
# - return the probability of churn for each new prediction, and the percentile where that prediction is in the distribution of probability predictions from the training dataset (e.g. a high probability of churn like 0.78 might be at the 90th percentile)
# - use other autoML packages, such as TPOT, H2O, MLBox, etc, and compare performance and features with pycaret
# - create a class in your Python module to hold the functions that you created
# - accept user input to specify a file using a tool such as Python's `input()` function, the `click` package for command-line arguments, or a GUI
# - Use the unmodified churn data (new_unmodified_churn_data.csv) in your Python script. This will require adding the same preprocessing steps from week 2 since this data is like the original unmodified dataset from week 1.

# In[1]:


get_ipython().system('conda create -n msds python=3.10.14 -y')
get_ipython().system('conda activate msds')
get_ipython().system('pip install --upgrade pycaret')


# In[15]:


import pandas as pd
df = pd.read_csv('C:/Users/gudip/Downloads/NewFeature_churn_data.csv', index_col='customerID')
df


# In[2]:


import sys
print(sys.version)


# Importing PyCaret 
# 

# In[7]:


from pycaret.classification import setup, compare_models, predict_model, save_model, load_model


# PyCaret's classification module wraps all the steps of the machine learning workflow through easy-to-use functions. The setup function provides initialization of the environment and preparation of data, compare_models evaluates several algorithms in search of the best, and predict_model is used for making predictions. save_model or load_model enables efficient management of trained models.
# 
# 
# Setting up our AutoML

# In[16]:


automl = setup(df, target='Churn')


# In[17]:


best_model = compare_models(sort='AUC')


# Checking the best model

# In[18]:


best_model


# If we care most about AUC, which is a bit higher-0.8289 for Gradient Boosting Classifier (gbc) compared to 0.8273 for Logistic Regression-then gbc is an appropriate choice. The models are similarly accurate, but gbc has the highest AUC. This in turn suggests that gbc has better general classification performance, especially in terms of distinguishing the classes.

# In[19]:


df.iloc[-2:-1].shape


# The predict_model(best_model, df.iloc[-2:-1]) call uses the learned model for prediction of the second last row in the dataframe. It takes as input just this one instance of data. The result includes the original features with the prediction class and probabilities.

# In[20]:


predict_model(best_model, df.iloc[-2:-1])


# We can see , two new columns are generated in the prediction process: prediction_label-Yes/No, which will be for customer churning, and prediction_score for the probability of the customer churning. The 0.5131 score indicates the model predicts likelihood of churn as "Yes" since it is greater than or equals 0.5.

# Saving the best Model and Loading it 

# In[35]:


save_model(best_model, 'best_model')


# <h3>Making a Python module<h3.

# In[58]:


import pandas as pd
from pycaret.classification import load_model, predict_model

def load_data(filepath):
   
    df = pd.read_csv(filepath, index_col='customerID')
    return df

def make_predictions(df):
   
    model = load_model('best_model') 
    predictions = predict_model(model, data=df)

    # Check the column names
    print("Predictions DataFrame Columns:")
    print(predictions.columns)

    # Rename 'prediction_label' to 'Churn_prediction' if it exists
    if 'prediction_label' in predictions.columns:
        predictions.rename(columns={'prediction_label': 'Churn_prediction'}, inplace=True)
        
        # Replace values in the new column
        predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
        
        return predictions[['Churn_prediction', 'prediction_score']]
    else:
        raise KeyError("The 'prediction_label' column was not found in the predictions DataFrame")

if __name__ == "__main__":
    # Load the new churn data
    new_data = load_data('C:/Users/gudip/Downloads/new_churn_data.csv')
    predictions = make_predictions(new_data)
    # True values for the new data
    true_values = [1, 0, 0, 1, 0]
    
    # Print predictions and true values
    print('Predictions:')
    print(predictions)
    print('True Values:')
    print(true_values)
    print('Predictions:')
    print(predictions)


    


# # Summary

# First It installed the required packages, imported the pycaret packages, and then configured the AutoML. Various classifiers were evaluated to predict churn. Among these,If we care most about AUC, which is a bit higher-0.8289 for Gradient Boosting Classifier (gbc) compared to 0.8273 for Logistic Regression-then gbc is an appropriate choice. It saved the best model as best_model.pkl along with its preprocessing pipeline. It performs predictions on a new dataset, new_churn_data.csv. This model predicted for all instances "No Churn". The true values were [1, 0, 0, 1, 0], and indeed, the model did not predict any of these as churning cases. This tends to depict how much more refinement of the model would be necessary to bring up good prediction capabilities. All in all, this process shows training a model, saving, and then applying it on the prediction for churning.

# In[ ]:




