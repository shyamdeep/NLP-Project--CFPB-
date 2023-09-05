#!/usr/bin/env python
# coding: utf-8

# In[4]:


import joblib
import re
import streamlit as st
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components


# In[ ]:


st.write("# Customer Complaints Classification")


# In[ ]:


complaint_text = st.text_input("Enter a complaint for classification")


# In[3]:


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


# In[8]:


model = joblib.load('complaints_classifier.joblib')


# In[16]:


def classify_complaint(model, complaint):
    label = model.predict([complaint])[0]
    product_dict ={0:'credit_reporting',1:'debt_collection',2:'mortgages_and_loans', 
                3:'credit_card',4:'retail_banking'}
    
    complaint_prob = model.predict_proba([complaint])
    return {'label': product_dict[label], 'complaint_prob': complaint_prob[0][label]}


# In[ ]:


if complaint_text != '':
    result = classify_complaint(model, complaint_text)
    st.write(result)
    
    explain_pred = st.button('Explain Predictions')
    if explain_pred:
        with st.spinner('Generating explanations'):
            class_names = ['credit_reporting','debt_collection','mortgages_and_loans','credit_card','retail_banking']
            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(complaint_text,model.predict_proba, num_features=10,top_labels=2)
            components.html(exp.as_html(), height=3500)





