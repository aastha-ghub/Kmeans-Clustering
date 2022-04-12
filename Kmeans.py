#!/usr/bin/env python
# coding: utf-8

#   <tr>
#         <td>
#             <div align="left">
#                 <font size=25px>
#                     <b>Kmeans Clustering
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>

# ## Problem Statement:
# A key challenge for e-commerce businesses is to analyze the trend in the market to increase their sales. The trend can be easily observed if the companies can group the customers; based on their activity on the e-commerce site.  This grouping can be done by applying different criteria like previous orders, mostly searched brands and so on. The machine learning clustering algorithms can provide an analytical method to cluster customers with similar interests.

# ## Data Definition:
# 
# Input variables:
# 
# 1) **Cust_ID** Unique numbering for customers
# 
# 2) **Gender:** Gender of the customer
# 
# 
# 3) **Orders:** Number of orders placed by each customer in the past
# 
# 
# Remaining 35 features contains the number of times customers have searched them

# ## Content
# 
# 1. **[Import Packages](#import_packages)**
# 2. **[Read Data](#Read_Data)**
# 3. **[Understand and Prepare the Data](#data_preparation)**
#     - 3.1 - [Data Types and Dimensions](#Data_Types)
#     - 3.2 - [Distribution of Variables](#data_prepartion)
#     - 3.3 - [Statistical Summary](#Statistical_Summary)
#     - 3.4 - [Missing Data Treatment](#Missing_Data_Treatment)
#     - 3.5 - [Visualization](#Visualization)
# 4. **[K-means Clustering](#modeling)**
#     - 4.1 - [Prepare the data](#preparation_of_data)
#     - 4.2 - [Build a Model with Multiple K](#model_k)
# 5. **[Retrieve the Clusters](#retrieve_clusters)**
# 6. **[Clusters Analysis](#cluster)**
#     - 6.1 - [Analysis of Cluster_1](#cluster_1)
#     - 6.2 - [Analysis of Cluster_2](#cluster_2)
#     - 6.3 - [Analysis of Cluster_3](#cluster_3)
#     - 6.4 - [Analysis of Cluster_4](#cluster_4)
# 7. **[Conclusion](#conclusion)**

# <a id='import_packages'></a>
# ## 1. Import Packages

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set default setting of seaborn
sns.set()


# <a id='Read_Data'></a>
# ## 2. Read the Data

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>Read the data using read_excel() function from pandas<br> 
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[24]:


# read the data
raw_data = pd.read_excel('cust_data.xlsx', index_col=0)

# print the first five rows of the data
raw_data.head()


# <a id='data_preparation'></a>
# ## 3. Understand and Prepare the Data
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>The process of data preparation entails cleansing, structuring and integrating data to make it ready for analysis. <br><br>
#                         Here we will analyze and prepare data :<br>
#                         1. Check dimensions and data types of the dataframe <br>
#                         2. Study summary statistics<br> 
#                         3. Check for missing values<br>
#                         4. Visualization<br>
#                         5. Study correlation<br>
#                                        </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Data_Types'></a>
# ## 3.1 Data Types and Dimensions

# In[25]:


# check the data types for variables
raw_data.info()


# In[26]:


# get the shape
print(raw_data.shape)


# **We see the dataframe has 37 columns and 30000 observations**

# <a id='dis'></a>
# ## 3.2 Distribution of Variables
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> Check the distribution of the variables <br><br>
#                         1. Distribution of orders placed by customers<br>
#                         2. Distribution of gender of the customer<br>
#                 </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. Distribution of orders placed by customers**
# 
# Check the distribution for the number of orders placed by the customers in the past

# In[27]:


# use 'countplot' to plot barplot for orders
sns.countplot(data = raw_data, x = 'Orders')

# set the axes and plot labels
# set the font size using 'fontsize'
plt.title('Distribution of Orders', fontsize = 15)
plt.xlabel('No. of Orders', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

plt.show()


# <table align='left'>
#     <tr>
#         <td width='8%'>
#             <img src='note.png'>
#         </td>
#         <td>
#             <div align='left', style='font-size:120%'>
#                     <b>It can be easily seen that most of the customers have no past orders 
#                 </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **2. Distribution of of gender of the customer**
# 
# Plot the barplot to get the count for the gender 

# In[28]:


# use 'countplot' to gender-wise calculate the number of customers 
sns.countplot(data= raw_data, x = 'Gender')

# set the axes and plot labels
# set the font size using 'fontsize'
plt.title('Distribution of Gender', fontsize = 15)
plt.xlabel('Gender', fontsize = 15)
plt.ylabel('No. of Customers', fontsize = 15)

# use below code to print the values in the graph
# 'x' and 'y' gives the position of the text
# 's' is the text 
plt.text(x = -0.1, y = raw_data.Gender.value_counts()[1] + 20, s = str(round((raw_data.Gender.value_counts()[1])*100/len(raw_data.Gender),2)) + '%')
plt.text(x = 0.9, y = raw_data.Gender.value_counts()[0] + 20, s = str(round((raw_data.Gender.value_counts()[0])*100/len(raw_data.Gender),2)) + '%')
plt.show()


# <table align='left'>
#     <tr>
#         <td width='8%'>
#             <img src='note.png'>
#         </td>
#         <td>
#             <div align='left', style='font-size:120%'>
#                     <b>There are more female customers in the data than the male customers<br><br>
#                         It can be seen that the variable 'Gender' has lesser observations (percent-wise only 90.92% observations) than the total number of observations. This inconsistency is because of the existence of missing values; we deal with this issue in section 4.5 
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Statistical_Summary'></a>
# ## 3.3 Statistical Summary
# Here we take a look at the summary of each attribute. This includes the count, mean, the min and max values as well as some percentiles for numeric variables.

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> In our dataset we have numerical variables. Now we check for summary statistics of all the variables<br>
#                         For numerical variables, we use .describe().
#           <br>
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **1. For numerical variables, use .describe()**

# In[29]:


# data frame with numerical features
raw_data.describe()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <b>The above output illustrates the summary statistics of the numeric variables.<br>
#                         The customers have placed 4 orders on an average with minimum zero orders and maximum of 12.<br>
#                         From the summary output, it can be seen that the considered dataset is sparse; since, for all the variables with brand searches, 75% of the observations are 0
#                 </b>     
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **2. For categorical variables, use .describe(include=object)**

# In[30]:


# summary of the categorical variables
raw_data.describe(include = object)

# Note: If we pass 'include=object' to the .describe() function returns descriptive statistics for categorical variables only


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <b>The summary contains information about the total number of observations, number of unique classes, the most occurring class and frequency of the same.<br> It can be seen that the mode of the variable 'Gender' is F with 22054 observations
#                 </b>     
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# <a id='Missing_Data_Treatment'></a>
# ## 3.4. Missing Data Treatment
# If the missing values are not handled properly we may end up drawing an inaccurate inference about the data. Due to improper handling, the result obtained will differ from the ones where the missing values are present.
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>In order to get the count of missing values in each column, we use the in-built function .isnull().sum()
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[31]:


# sorting variables based on null values
# 'ascending = False' sorts values in the descending order
Total = raw_data.isnull().sum().sort_values(ascending=False)          

# percentage of missing values
Percent = (raw_data.isnull().sum()/raw_data.isnull().count()*100).sort_values(ascending=False)   

# create a dataframe using 'concat' function 
# 'keys' is the list of column names
# 'axis = 1' concats along the columns
missing_data = pd.concat([Total, Percent], axis=1, keys=['Total', 'Percent'])    
missing_data


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <b> Only the variable 'Gender' has 9% of missing values 
#                 </b>     
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **We plot a heatmap for visualization of missing values**

# In[32]:


# plot heatmap to check null values
# 'cbar = False' does not show the color axis 
sns.heatmap(raw_data.isnull(), cbar=False)

# set the axes and plot labels
# set the font size using 'fontsize'
plt.title('Heatmap for Missing Values', fontsize = 15)
plt.xlabel('Variables', fontsize = 15)
plt.ylabel('Cust_ID', fontsize = 15)

plt.show()


# **The horizontal lines in the heatmap correspond to the missing values**

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>How to deal with missing data?<br><br>
# Drop data<br>
# a. Drop the whole row<br>
# b. Drop the whole column<br><br>
# Replace data<br>
# a. Replace it by mean<br>
# b. Replace it by frequency<br>
# c. Replace it based on other functions<br><br>
# Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns are empty enough to drop entirely. We have some freedom in choosing which method to replace data 
#                 </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# #### Replace missing values in 'Gender'
# 
# 'Gender' is a categorical variable with categories, 'M' and 'F'. We have 2724 customers whose gender is not known to us. To deal with this, we perform dummy encoding for the variable  

# In[33]:


# create dummies against 'gender'
data = pd.get_dummies(raw_data,columns=['Gender'])     

# head() to display top five rows
data.head()


# In[34]:


# check the dimensions after dummy encoding
data.shape


# **We see the dataframe has 38 columns and 30000 observations**

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
# <b>Gender_F' and 'Gender_M' are the two newly created variables that do not possess any missing value. The customer for which both the columns have '0' value indicates that the gender is not known </b>     </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[35]:


# recheck the null values
data.isnull().sum()


# There are no missing values present in the data.

# <a id='Visualization'></a>
# ## 3.5. Visualization
# 

# **PDF's of features**

# In[36]:


fig = data.hist(figsize = (18,18))


# <a id='modeling'></a>
# # 4. K-means Clustering
# 

# Centroid-based clustering algorithms cluster the data into non-hierarchical clusters. Such algorithms are efficient but sensitive to initial conditions and outliers. K-means is the most widely-used centroid-based clustering algorithm

# <a id='preparation_of_data'></a>
# ## 4.1 Prepare the Data
# 
# Feature scaling is used to transform all the variables in the same range. If the variables are not in the same range, then the variable with higher values can dominate our final result. 
# 
# The two most discussed scaling methods are normalization and standardization. 
# 
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> We consider only the brand names to segment the customers. Thus, drop the variables 'Orders', 'Gender_F', 'Gender_M' and scale the remaining variables
#                       </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[37]:


# 'features' contain only the brand names
features = data.drop(['Orders', 'Gender_F', 'Gender_M'], axis=1)

# head() to display top five rows
features.head()


# **Scale the data**

# In[41]:


from sklearn.preprocessing import StandardScaler
# instantiate and fit 'StandardScaler' function
scale = StandardScaler().fit(features)       

# scale the 'features' data
features = scale.transform(features)                


# In[43]:


# create a dataframe of the scaled features 
features_scaled = pd.DataFrame( features, columns= data.columns[1:36])

# head() to display top five rows
features_scaled.head()


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>Thus, we have scaled all the features in the data and stored it in a dataframe named 'features_scaled'</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# <a id='model_k'></a>
# ## 4.2 Build a Model with Multiple K
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> We build our models using the silhouette score method. 
# The silhouette is a method of interpretation and validation of consistency within clusters of data
#                       </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **We do not know how many clusters give the most useful results. So, we create the clusters varying K, from 4 to 8 and then decide the optimum number of clusters (K) with the help of the silhouette score**

# In[46]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# create a list for different values of K
n_clusters = [4, 5, 6, 7, 8]

# use 'for' loop to build the clusters
# 'random_state' returns the same sample each time you run the code  
# fit and predict on the scaled data
# 'silhouette_score' function computes the silhouette score for each K
for K in n_clusters :
    cluster = KMeans (n_clusters= K, random_state= 10)
    predict = cluster.fit_predict(features_scaled)
    
    score = silhouette_score(features_scaled, predict, random_state= 10)
    print ("For n_clusters = {}, silhouette score is {})".format(K, score))


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>The optimum value for K is associated with the high value of the 'silhouette score'. From the above output it can be seen that, for K = 4, the silhouette score is highest. Thus, we build the clusters with K = 4</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# In[47]:


# building a K-Means model for K = 4
model = KMeans(n_clusters= 4, random_state= 10)

# fit the model
model.fit(features_scaled)


# **Now, explore these 4 clusters to gain some insights about the clusters**

# <a id='retrieve_clusters'></a>
# # 5. Retrieve the Clusters
# 
# 

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> As we have built the 4 clusters, now we want to know which customers belong to which cluster. 'model.labels_' can give the cluster number in which the customer belongs
#                       </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# 
# </table>

# In[48]:


data_output = data.copy(deep = True)
# add a column 'Cluster' in the data giving cluster number corresponding to each observation
data_output['Cluster'] = model.labels_

# head() to display top five rows
data_output.head()


# **We have added a column 'cluster' in the dataframe describing the cluster number for each observation**

# #### Check the size of each cluster

# In[49]:


# 'return_counts = True' gives the number observation in each cluster
np.unique(model.labels_, return_counts=True)                


# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b>Plot a barplot to visualize the cluster sizes</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 

# In[51]:


# use 'seaborn' library to plot a barplot for cluster size
sns.countplot(data= data_output, x = 'Cluster')

# set the axes and plot labels
# set the font size using 'fontsize'
plt.title('Cluster Sizes', fontsize = 15)
plt.xlabel('Clusters', fontsize = 15)
plt.ylabel('No. of Customers', fontsize = 15)

# add values in the graph
# 'x' and 'y' assigns the position to the text
# 's' represents the text on the plot
plt.text(x = -0.18, y =2000, s = np.unique(model.labels_, return_counts=True)[1][0])
plt.text(x = 0.9, y =2000, s = np.unique(model.labels_, return_counts=True)[1][1])
plt.text(x = 1.85, y =2000, s = np.unique(model.labels_, return_counts=True)[1][2])
plt.text(x = 2.85, y =2000, s = np.unique(model.labels_, return_counts=True)[1][3])

plt.show()


# **The first cluster is the largest cluster containing 22573 customers**

# #### Cluster Centers
# 
# The cluster centers can give information about the variables belonging to the clusters
# 

# In[53]:


# form a dataframe containing cluster centers
# 'cluster_centers_' returns the co-ordinates of a cluster center 
centers = pd.DataFrame(model.cluster_centers_, columns=  data_output.columns[1:36])      


# In[54]:


# head() to display top five rows
centers.head()


# **Now, extract the variables in each of the clusters and try to name each of the cluster based on the variables**

# <a id='cluster'></a>
# # 6 Clusters Analysis
# 

# <a id='cluster_1'></a>
# ## 6.1 Analysis of Cluster_1

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                     <b> Here we analyze the first cluster:<br><br>
#                         1. Check the size of a cluster <Br>
#                         2. Sort the variables belonging to the cluster <br>
#                         3. Compute the statistical summary for observations in the cluster  
#                      </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# Sort all the variables based on value for the cluster center (i.e., the variable with the highest value of the cluster center will be on top of the sorted list) and store the first ten variables as a list

# In[55]:


# sort the variables based on cluster centers
cluster_1 = sorted(zip(list(centers.iloc[0,:]), list(centers.columns)), reverse = True)[:10]     


# **1. Check size of the cluster**

# In[56]:


# size of a cluster_1
np.unique(model.labels_, return_counts=True)[1][0]


# There are 22587 customers in this cluster. This is the largest cluster among all the clusters

# **2. Sort variables belonging to the cluster**

# In[57]:


# retrieve the top 10 variables present in the cluster
cluster1_var = pd.DataFrame(cluster_1)[1]
cluster1_var


# <table align="left">
#    <tr>
#         <td width="8%">
#            <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>Most of the customers belonging to this cluster have searched for electronics, apparels as well as grocery brands like HP, Apple, Prada, Reebok, Pillsbury, Bertolli, and so on. Thus, we can segment this cluster under 'Basket class'</b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# **3. Compute the statistical summary for observations in the cluster**

# In[58]:


# get summary for observations in the cluster
# consider the number of orders and customer gender for cluster analysis
data_output[['Orders', 'Gender_F', 'Gender_M', 'Cluster']][data_output.Cluster == 0].describe()


# The proportion of both male and female customers is proportionate in this cluster as compared to the overall gender proportion in the dataset

#  <a id='cluster_2'></a>
# ## 6.2 Analysis of Cluster_2

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> Here we analyze the second cluster:<br><br>
#                         1. Check the size of a cluster <Br>
#                         2. Sort the variables belonging to the cluster <br>
#                         3. Compute the statistical summary for observations in the cluster         </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[61]:


# sort the variables based on cluster centers
cluster_2 = sorted(zip(list(centers.iloc[1,:]), list(centers.columns)), reverse = True)[:10]     


# **1. Check the size of a cluster**

# In[62]:


# size of a cluster_2
np.unique(model.labels_, return_counts=True)[1][1]


# 561 customers belong to cluster_2. This is the smallest cluster

# **2. Sort variables belonging to the cluster**

# In[63]:


# retrieve the top 10 variables present in the cluster
cluster2_var = pd.DataFrame(cluster_2)[1]
cluster2_var        


# <table align="left">
#    <tr>
#         <td width="8%">
#            <img src="note.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>It can be observed that most of the customers in this cluster have searched for electronics brands like Bosch, Samsung, OnePlus and so on. Thus, we can segment this cluster under 'Electronics'
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 
# 
# 
# 
# 

# **3. Compute the statistical summary for observations in the cluster**

# In[65]:


# get summary for observations in the cluster
# consider the number of orders and customer gender for cluster analysis
data_output[['Orders', 'Gender_F', 'Gender_M', 'Cluster']][data_output.Cluster == 1].describe()


# This cluster contains highest male population among all the clusters. But, there is high deviation in both the genders

# <a id='cluster_3'></a>
# ## 6.3 Analysis of Cluster_3

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> Here we analyze the third cluster:<br><br>
#                         1. Check the size of a cluster <Br>
#                         2. Sort the variables belonging to the cluster <br>
#                         3. Compute the statistical summary for observations in the cluster         </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[66]:


# sort the variables based on cluster centers
cluster_3 = sorted(zip(list(centers.iloc[2,:]), list(centers.columns)), reverse = True)[:10]   


# **1. Check the size of a cluster**

# In[67]:


# size of cluster_3
np.unique(model.labels_, return_counts=True)[1][2]


# This cluster contains 1267 customers

# **2. Sort variables belonging to the cluster**

# In[68]:


# retrieve the top 10 variables present in the cluster
cluster3_var = pd.DataFrame(cluster_3)[1]
cluster3_var             


# <table align="left">
#    <tr>
#         <td width="8%">
#            <img src="note.png">
#         </td>
#        <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This cluster contains the customers who have searched for food brands like Nestle, Buskin-Robbin's,  Pillsbury, and so on. Thus, we can segment this cluster under 'Grocery' 
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 
# 
# 
# 
# 
# 
# 

# **3. Compute the statistical summary for observations in the cluster**

# In[70]:


# get summary for observations in the cluster
# consider the number of orders and customer gender for cluster analysis
data_output[['Orders', 'Gender_F', 'Gender_M', 'Cluster']][data_output.Cluster == 2].describe()


# It can be observed that there is a majority of female customers (with mean 0.82) in this cluster. 

# <a id='cluster_4'></a>
# ## 6.4 Analysis of Cluster_4

# <table align="left">
#     <tr>
#         <td width="8%">
#             <img src="key.png">
#         </td>
#         <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b> Here we analyze the fourth cluster:<br><br>
#                         1. Check the size of a cluster <Br>
#                         2. Sort the variables belonging to the cluster <br>
#                         3. Compute the statistical summary for observations in the cluster         </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[71]:


# sort the variables based on cluster centers
cluster_4 = sorted(zip(list(centers.iloc[3,:]), list(centers.columns)), reverse=True)[:10]   


# **1. Check the size of a cluster**

# In[72]:


# size of cluster_4
np.unique(model.labels_, return_counts=True)[1][3]


# This cluster contains 5585 customers

# **2. Sort variables belonging to the cluster**

# In[73]:


# retrieve the top 10 variables present in the cluster
cluster4_var = pd.DataFrame(cluster_4)[1]
cluster4_var             


# <table align="left">
#    <tr>
#         <td width="8%">
#            <img src="note.png">
#         </td>
#        <td>
#             <div align="left", style="font-size:120%">
#                 <font color="#21618C">
#                     <b>This cluster contains the customers who have searched for clothing brands like Scabal, Jordan, Dior, H&M, and so on. Thus, we can segment this cluster under 'Apperals' 
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>
# 
# 
# 
# 
# 
# 
# 
# 

# **3. Compute the statistical summary for observations in the cluster**

# In[74]:


# get summary for observations in the cluster
# consider the number of orders and customer gender for cluster analysis
data_output[['Orders', 'Gender_F', 'Gender_M', 'Cluster']][data_output.Cluster==3].describe()


# This cluster contains highest female population and lowest male population among all the clusters

# <a id='conclusion'></a>
# # 7. Conclusion

# <table align="left">
#    <tr>
#       <td width="8%">
#            <img src="key.png">
#        </td>
#          <td>
#         <div align="left", style="font-size:120%">
#               <font color="#21618C">
#                 <b>In this case study, we have grouped the customers' dataset into 4 clusters based on the brands they have searched on e-commerce sites. We have used the silhouette score method to find the optimum number of clusters and decided k = 4 as the best pick after analyzing the silhouette score.<br><br>
# After applying the K-means algorithm with an optimized number of clusters, we segment the customers under 'Grocery', 'Apparels', 'Electronics', and 'Basket class' categories. These clusters give information about the interest of the customer in the different brands. This type of segmentation can help the e-commerce companies, to know the customer's choices and they can provide more accurate recommendations to the customers                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# In[ ]:





# In[ ]:




