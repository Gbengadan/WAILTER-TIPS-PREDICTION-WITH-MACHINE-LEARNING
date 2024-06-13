#!/usr/bin/env python
# coding: utf-8

# # wailter tip prediction
# 

# In[84]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# In[85]:


df = pd.read_csv(r"C:\Users\DELL\Downloads\Requirements (1)\tips.csv")
df


# In[86]:


df.head()


# In[87]:


#lets look at the factor at that affect wailter tip


# In[88]:


figure = px.scatter(data_frame = df, x="total_bill", y="tip", size ="size",color="day", trendline ="ols")
figure.show()


# In[89]:


figure = px.scatter(data_frame = df, x="total_bill", y="tip", size ="size",color="sex", trendline ="ols")
figure.show()


# In[90]:


figure = px.scatter(data_frame = df, x="total_bill", y="tip", size ="size",color="time", trendline ="ols")
figure.show()


# In[52]:


#Now let's see the tips given to the wailters according to days, to find out which day the most tips are given to the waiters.


# In[53]:


figure = px.pie(df,
               values = 'tip',
               names='day', hole = 0.5)
figure.show()


# In[54]:


#it shows that saturday get the most tips


# In[55]:


#now lets look at the number of tips given to the waiters by gender
figure = px.pie(df,
               values = 'tip',
               names='sex', hole = 0.5)
figure.show()


# In[56]:


#the highest gender tip given to the wailter is male


# In[ ]:





# In[57]:


#now lets look at the number of tips given to the waiters by smoker


# In[58]:


figure = px.pie(df,
               values = 'tip',
               names='smoker', hole = 0.5)
figure.show()


# In[59]:


#therefore none smoker gave highest tips


# In[60]:


#now lets look at the number of tips given to the waiters by time of the day


# In[61]:


figure = px.pie(df,
               values = 'tip',
               names='time', hole = 0.5)
figure.show()


# In[62]:


figure = px.pie(df,
               values = 'tip',
               names='size', hole = 0.5)
figure.show()


# In[63]:


#thats how i analyse all the factors that affecting wailter tips


# ## Waiter Tips Prediction Model

# In[64]:


df["sex"] = df["sex"].map({"Female": 0, "Male": 1})
df["smoker"] = df["smoker"].map({"No": 0, "Yes": 1})
df["day"] = df["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
df["time"] = df["time"].map({"Lunch": 0, "Dinner": 1})
df.head()


# In[76]:


x =np.array(df[["total_bill","sex", "smoker", "day", "time", "size"]])

y =np.array(df["tip"])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 42)


# In[79]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)


# In[80]:


# Features = [["total_bill", "sex", "smoker", "day", "time", "size"]]
features = np.array([[24.50, 1, 0, 0, 1, 4]])
model.predict(fea)


# In[ ]:




