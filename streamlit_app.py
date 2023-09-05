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


# In[3]:


def classify_complaint(model, complaint):
    label = model.predict([complaint])[0]
    complaint_prob = model.predict_proba([complaint])
    return {'label': label, 'complaint_prob': complaint_prob[0][label]}


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


# In[7]:


#complaint_text = "applied property order able view several property order able use make consumer credit transaction purchase home filing report agency back think pertaining agency bank america recieved respone letter call time saying america mortgage never responded asociate ask come back bofa reapply went pa spoke asociate name show response letter showing previously said letter mean thing bank often send letter like customer said could assist put touch mortgage department said could assist said score low bank assist getting told federal law deny credit told denying simply stating bofa policy refuse give completed mortgage application anyway spoke phone told credit score low bofa policy unable assist extension consumer credit"


# In[12]:


#model.predict([complaint_text])[0]


# In[15]:


#model.predict_proba([complaint_text])[0][2]


# In[ ]:




