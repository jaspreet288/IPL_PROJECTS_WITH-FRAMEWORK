#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

balling_df = pd.read_csv(r"C:\Users\jaspr\Downloads\ipl.csv")
print(balling_df)
match_df = pd.read_csv(r"C:\Users\jaspr\Downloads\ipl match 2008-.csv")
print(match_df)

print(match_df.head)
print(balling_df.head)
df = match_df.isnull().sum()
print(df)
df2 = balling_df.isnull().sum()
print(df2)
print(match_df.columns)

print("matche played so far:", match_df.shape[0])
print("\n cities played at:", match_df['city'].unique())
print("\n team participated:", match_df['team1'].unique())

match_df["season"] = pd.DatetimeIndex(match_df["date"]).year

print(match_df.head)

match_per_season = match_df.groupby(['season'])['id'].count().reset_index().rename(columns={'id':'matches'})
print(match_per_season)

sns.countplot(match_df['season'])
plt.xticks(rotation=90, fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('season', fontsize=10)
plt.ylabel('Count', fontsize=10)
plt.title('Total matches played in each season', fontsize = 10, fontweight = "bold")


# In[5]:


season_data=match_df[['id','season']].merge(balling_df, left_on = 'id', right_on = 'id', how = 'left').drop('id', axis = 1)
season_data.head()


# In[7]:


season=season_data.groupby(['season'])['total_runs'].sum().reset_index()
p=season.set_index('season')
ax = plt.axes()
ax.set(facecolor = "grey")
sns.lineplot(data=p,palette="magma") 
plt.title('Total runs in each season',fontsize=12,fontweight="bold")
plt.show()


# In[9]:


runs_per_season=pd.concat([match_per_season,season.iloc[:,1]],axis=1)
runs_per_season['Runs scored per match']=runs_per_season['total_runs']/runs_per_season['matches']
runs_per_season.set_index('season',inplace=True)
runs_per_season


# In[10]:


toss=match_df['toss_winner'].value_counts()
ax = plt.axes()
ax.set(facecolor = "grey")
sns.set(rc={'figure.figsize':(15,10)},style='darkgrid')
ax.set_title('No. of tosses won by each team',fontsize=15,fontweight="bold")
sns.barplot(y=toss.index, x=toss, orient='h',palette="icefire",saturation=1)
plt.xlabel('# of tosses won')
plt.ylabel('Teams')
plt.show()


# In[13]:


ax = plt.axes()
ax.set(facecolor = "grey")
sns.countplot(x='season', hue='toss_decision', data=match_df,palette="magma",saturation=1)
plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=15)
plt.xlabel('\n Season',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.title('Toss decision across seasons',fontsize=12,fontweight="bold")
plt.show()


# In[14]:


match_df['result'].value_counts()


# In[15]:


match_df.venue[match_df.result!='runs'].mode()


# In[17]:


match_df.venue[match_df.result!='wickets'].mode()


# In[18]:


toss = match_df['toss_winner'] == match_df['winner']
plt.figure(figsize=(10,5))
sns.countplot(toss)
plt.show()


# In[19]:


plt.figure(figsize=(12,4))
sns.countplot(match_df.toss_decision[match_df.toss_winner == match_df.winner])
plt.show()


# In[20]:


player = (balling_df['batsman']=='SK Raina')
df_raina=balling_df[player]
df_raina.head()


# In[21]:


df_raina['dismissal_kind'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,rotatelabels=True)
plt.title("Dismissal Kind",fontweight="bold",fontsize=15)
plt.show()


# In[22]:


def count(df_raina,runs):
    return len(df_raina[df_raina['batsman_runs']==runs])*runs


# In[23]:


print("Runs scored from 1's :",count(df_raina,1))
print("Runs scored from 2's :",count(df_raina,2))
print("Runs scored from 3's :",count(df_raina,3))
print("Runs scored from 4's :",count(df_raina,4))
print("Runs scored from 6's :",count(df_raina,6))


# In[24]:


match_df[match_df['result_margin']==match_df['result_margin'].max()]


# In[25]:


runs = balling_df.groupby(['batsman'])['batsman_runs'].sum().reset_index()
runs.columns = ['Batsman', 'runs']
y = runs.sort_values(by='runs', ascending = False).head(10).reset_index().drop('index', axis=1)
y


# In[26]:


ax = plt.axes()
ax.set(facecolor = "grey")
sns.barplot(x=y['Batsman'],y=y['runs'],palette='rocket',saturation=1)
plt.xticks(rotation=90,fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('\n Player',fontsize=15)
plt.ylabel('Total Runs',fontsize=15)
plt.title('Top 10 run scorers in IPL',fontsize=15,fontweight="bold")


# In[28]:


ax = plt.axes()
ax.set(facecolor = "black")
match_df.player_of_match.value_counts()[:10].plot(kind='bar')
plt.xlabel('Players')
plt.ylabel("Count")
plt.title("Highest MOM award winners",fontsize=15,fontweight="bold")


# In[ ]:




