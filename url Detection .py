
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('host_detections.csv', names=['host','detections','detection_count'], header=1, encoding='utf-8')


# ### Cleanup - Blacklist Data
# Drop the duplicates on the *df* dataframe, for column *host*

# This next section cleans up the detections column. It removes the text formatting and puts the information into a Python list, and places the Python list back into the dataframe in place of the text. It also creates a multi-dimensional list that represents the the various blacklists and if there was a hit for the domain *1* or not *0*. 

# In[2]:


df.count()


# In[3]:


df.drop_duplicates(subset='host', inplace=True)


# In[4]:


df = df.replace(" ", "") 
df = df.replace("u'", "")
df = df.replace("'", "")
df = df.replace("[", "")
df = df.replace("]", "")


# In[5]:


df.head()


# In[6]:


df.count()


# Join the resulting multi-dimensional list to the "side" of the existing dataframe. 
# 
# You can see the host **02b123c.netsolhost.com** has 0 detections, and has *0*s in place for all of the blacklist values. Where **0lilioo0l0o00lilil.info** has 7 detections and a *1* in place of it's detections (e.g. hpHosts).

# In[9]:


import warnings
df = df.join(pd.DataFrame(index=df.index, columns=black_list_sources))
df = df.fillna(0)
for i in df.index:
    for x in df.xs(i)['detections']:
        df.ix[i, x] = 1
df.head()


# ### File Input - VirusTotal
# The data is in a file named *mal_domains.csv* and has columns: *host*, *count*, and *detections*. This data has been pre-processed to save some pain on parsing and assembling massive amounts of JSON data.

# In[7]:


av_domains = pd.read_csv('mal_domains.csv', names=['host','count','detections'], encoding='utf-8')


# ### Cleanup - VirusTotal
# Similar to the above we clean-up the detections column.

# In[8]:


av_list_sources = set()
def get_list(x):
    detections = []
    if not (len(x) == 1 and int(x) == 0):
        x = x.replace(" ", "")
        x = x.replace("u'", "")
        x = x.replace("'", "")
        x = x.replace("[", "")
        x = x.replace("]", "")
        [av_list_sources.add(i) for i in x.split(',') if len(i) > 1]
        [detections.append(i) for i in x.split(',') if len(i) > 1]
    return detections
av_domains.detections = av_domains.detections.apply(lambda x: get_list(x))
av_domains.head()


# In[9]:


av_domains.count()


# A little massaging is necessary here because there are blacklists and AV engines that have the same name. This renames the columns and places an *av_* prefix to the name ensuring there are no duplicates, and has the extra advantage of allow easy distinction in analysis.
# 
# Also, join the AV dataframe to the blacklist one created above.

# In[10]:


new_cols = av_domains.columns - ['host']
new_cols = ['av_' + x for x in new_cols.tolist()]
df = df.join(pd.DataFrame(index=df.index, columns=new_cols))


# This is where the expansion, and then filling in of values, *1* for detection and *0* for no detection, happens.

# In[11]:


for i in df.index:
    host = df.xs(i)['host']
    avs = av_domains[av_domains['host'] == host]['detections']
    if len(avs) > 0:
        for a in avs.values.tolist()[0]:
            df.ix[i, 'av_' + a] = 1
        df.ix[i, 'av_count'] = av_domains[av_domains['host'] == host]['count'].values[0]
        df.ix[i, 'av_detections'] = av_domains[av_domains['host'] == host]['detections'].values


# In[12]:


df.av_detections = df.av_detections.apply(lambda x: [] if isinstance(x, float) or len(x) < 1 else x[0])


# In[13]:


df = df.fillna(0)
#del df['None']


# For consistency's sake, set all of the columns but *host*, *detections*, and *av_detections* to type **int**

# In[14]:


int_cols = list(df.columns - ['host','detections','av_detections'])
df[int_cols] = df[int_cols].astype(int)


# Take a look at the resulting dataframe, you'll see a similar structure to the one above.
# 
# The cell below shows how to print the dimensions of the dataframe, in this case it has 346 rows and 97 columns (e.g. dimensions). This is due to the selection clause, it looks for domains that have zero AV results against it, and more than one blacklist hit.
# 
# Try reversing the query *av_count* > 0 and *detection_count* == 0.

# In[17]:


a = df[(df['av_count'] == 0) & (df['detection_count'] > 0)].head()


# In[18]:


a


# In[22]:


a.shape


# In[23]:


b = df[(df['av_count'] > 0) & (df['detection_count'] == 0)].head()


# In[24]:


b


# In[25]:


b.shape


# In your exploration you might have run across an IP address or 2, let's split these up into two different dataframes. This will allow and apples-to-apples comparison.

# In[26]:


domains = df[~df.host.str.contains("^\d+\.\d+\.\d+\.\d+$")]
ips = df[df.host.str.contains("^\d+\.\d+\.\d+\.\d+$")]


# How many elements (rows) are in each dataframe (*domains*, *ips*)?

# In[27]:


domains.shape


# In[28]:


ips.shape


# ### Analysis
# The cell below pulls out the list of features what we want to use. In this case it's all of the columns that don't (or appear not to) add any value to the analysis. The hostname is what is being analyzed, the *detections* and *av_detections* are sparse text that can't be use in this lab, and the counts should be summed-up/accounted for by the presence or lack of a qualifying detection event (AV or blacklist).

# In[33]:


cols = list(domains.columns - ['host','detections','av_detections','av_count','detection_count'])


# ### K-Means Clustering
# K-Means works on a fairly simple idea. You provide the algorithm with **K**, the number of clusters you think are in the dataset. The algorithm will attempt to find points that have the minimum distance to the other points, the centroids dictate the center of the cluster.
# 
# Below, the **K** for K-means was set to two. There are many ways to determine an optimal K, but for this exercise we're only interested in two labels, good and bad. By doing this we can guide the algorithm into picking two centers and giving us a "good" group and a "bad" group of domains.
# 
# The data is clustered two times. One time with both the blacklist and AV features, and another time with just the blacklist features. The labels for the clusters are stored in *bl_vt_labels* and *bl_labels* respectively. This allows an easy way to reference the labels without re-clustering the data later on.
# 
# You should add a third cluster section that stores the labels in *vt_labels*, and is only a cluster of columns from the AV set. Remember the AV results are prefixed with *av_* making the columns easy to pick out.

# In[30]:


#Initial labeling of the data with 2 different datasets (URLS + VT, and just URLS)
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

X = domains.as_matrix(cols)

k_clusters = 2
kmeans = KMeans(n_clusters=k_clusters)
kmeans.fit(X)
bl_vt_labels = kmeans.labels_

# Blacklist only columns
bl_cols = [x for x in cols if not 'av_' in x]
X = domains.as_matrix(bl_cols)

k_clusters = 2
kmeans = KMeans(n_clusters=k_clusters)
kmeans.fit(X)
bl_labels = kmeans.labels_

# VirusTotal only columns
vt_cols = [x for x in cols if 'av_' in x]
X = domains.as_matrix(vt_cols)

k_clusters = 2
kmeans = KMeans(n_clusters=k_clusters)
kmeans.fit(X)
vt_labels = kmeans.labels_


# Check your work! Make sure to print out at least a few elements of *vt_labels*.

# In[19]:


print bl_labels[:10]
print bl_vt_labels[:10]
print vt_labels[:10]


# Remember, the algorithm doesn't know what's malicious or not, so don't place any inherent value in a label of *1* or *0*. It's only a label of what group the algorithm thinks the data belongs in. Although, you as an analyst, might be able to infer if it's in the malicious or benign cluster.
# 
# ----
# 
# Below is a way to spot check domains, explore a couple more on your own. You can see what blacklists and AV engines, if any, are associated with the domain.

# In[20]:


d = "0lilioo0l0o00lilil.info"
print "Domain %s has bl_label: %d" %(d, bl_labels[domains[domains['host'] == d].index[0]])
print "Domain %s has bl_vt_label: %d" %(d, bl_vt_labels[domains[domains['host'] == d].index[0]])
R = zip(list(domains.columns), domains[domains['host'] == d].values.tolist()[0])
for r in R:
    if r[1] == 1:
        print r


# In[21]:


d = "02b123c.netsolhost.com"
print "Domain %s has bl_label: %d" %(d, bl_labels[domains[domains['host'] == d].index[0]])
print "Domain %s has bl_vt_label: %d" %(d, bl_vt_labels[domains[domains['host'] == d].index[0]])
R = zip(list(domains.columns), domains[domains['host'] == d].values.tolist()[0])
for r in R:
    if r[1] == 1:
        print r


# In[22]:


d = "0hb.ru"
print "Domain %s has bl_label: %d" %(d, bl_labels[domains[domains['host'] == d].index[0]])
print "Domain %s has bl_vt_label: %d" %(d, bl_vt_labels[domains[domains['host'] == d].index[0]])
R = zip(list(domains.columns), domains[domains['host'] == d].values.tolist()[0])
for r in R:
    if r[1] == 1:
        print r


# In[23]:


len(cols)


# ### PCA
# PCA is used for dimensionality reduction, one of the major advantages of this is being able to visualize data. Our current dataset has 92 features/dimensions, which unless you have super powers is pretty hard to visualize. One awesome use of PCA is to reduce these dimensions down into something that we as mortals can see.
# 
# The first exercise is reducing all 92 dimensions down to three for easy and pretty graphing. The colors in the graph are set by the labels from the K-Means clustering above.
# 
# Do the same as the cell below but one set of graphs for the blacklist only data and one set of graphs for the VirusTotal only data. What kinds of patterns emerge?
# 
# **Hint** don't forget to use the right labels for the right columns.

# In[24]:


import pylab
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

pylab.rcParams['figure.figsize'] = (16.0, 5.0)

X = PCA(n_components=3).fit_transform(domains.as_matrix(cols))
colors = ['green' if x == 1 else 'red' for x in bl_vt_labels]

figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
fig.suptitle("Exploding Tacos!")
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], alpha=.5, color=colors, s=50)
ax.set_title("Kmeans Clusters")
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlim(-5,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
ax.scatter(X[:,0], X[:,1], X[:,2], alpha=.5, color=colors, s=50)
ax.set_title("KMeans Clusters (zoomed in)")
plt.show()


# In[25]:


pylab.rcParams['figure.figsize'] = (16.0, 5.0)

X = PCA(n_components=3).fit_transform(domains.as_matrix(bl_cols))
colors = ['green' if x == 1 else 'red' for x in bl_labels]

figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], alpha=.5, color=colors, s=50)
ax.set_title("Kmeans Clusters")
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlim(-5,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
ax.scatter(X[:,0], X[:,1], X[:,2], alpha=.5, color=colors, s=50)
ax.set_title("KMeans Clusters (zoomed in)")
plt.show()


# In[26]:


pylab.rcParams['figure.figsize'] = (16.0, 5.0)

X = PCA(n_components=3).fit_transform(domains.as_matrix(vt_cols))
colors = ['green' if x == 1 else 'red' for x in vt_labels]

figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
fig.suptitle("Exploding Tacos!")
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], alpha=.5, color=colors, s=50)
ax.set_title("Kmeans Clusters")
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlim(-5,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
ax.scatter(X[:,0], X[:,1], X[:,2], alpha=.5, color=colors, s=50)
ax.set_title("KMeans Clusters (zoomed in)")
plt.show()


# ### 2D
# Now that you're a wiz at reducing various dimensions to three, it's possible to reduce down to two and graph that. Perhaps some more or different structure will pop out at you.
# 
# Once again the blacklist and VirusTotal scenario is done for you, do the same as above and examine the blacklist only and VirusTotal cases in 2D.

# In[27]:


colors = ['green' if x == 1 else 'red' for x in bl_vt_labels]
DD = PCA(n_components=2).fit_transform(domains.as_matrix(cols))
figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(DD[:,0], DD[:,1], s=50, alpha=.5, color=colors)
ax.set_title("Raw Data 2D")
plt.show()


# In[28]:


colors = ['green' if x == 1 else 'red' for x in bl_labels]
DD = PCA(n_components=2).fit_transform(domains.as_matrix(bl_cols))
figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(DD[:,0], DD[:,1], s=50, alpha=.5, color=colors)
ax.set_title("Raw Data 2D")
plt.show()


# In[29]:


colors = ['green' if x == 1 else 'red' for x in vt_labels]
DD = PCA(n_components=2).fit_transform(domains.as_matrix(vt_cols))
figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(DD[:,0], DD[:,1], s=50, alpha=.5, color=colors)
ax.set_title("Raw Data 2D")
plt.show()


# ### 1D
# Our last stop on this journey is 1D. The insights gained by visualizing the data in both three and two dimensions can be pretty helpful. As the beginning of the lab stated our goal is to create some kind of ranking or prioritization of the domains which is just a one-dimensional task. We'll cheat a little bit since looking at a list of numbers isn't that pretty. We'll cheat a bit for the graphing and plot our points along the X-axis with a Y value of 0 for each point.
# 
# The case of all the features has been provided for you, repeat the process for blacklist only and AV only.

# In[30]:


import numpy as np

colors = ['green' if x == 1 else 'red' for x in bl_vt_labels]
D = PCA(n_components=1).fit_transform(domains.as_matrix(cols))
print len(D)
DD = np.ndarray(shape=(len(D),2), dtype=float, order='F')
for i in range(0,len(D)):
    DD[i] = [D[i], 0.0]

figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(DD[:,0], DD[:,1], s=50, color=colors)
ax.set_title("Line 'em up!")
plt.show()


# In[31]:


colors = ['green' if x == 1 else 'red' for x in bl_labels]
D = PCA(n_components=1).fit_transform(domains.as_matrix(bl_cols))
print len(D)
DD = np.ndarray(shape=(len(D),2), dtype=float, order='F')
for i in range(0,len(D)):
    DD[i] = [D[i], 0.0]

figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(DD[:,0], DD[:,1], s=50, color=colors)
ax.set_title("Line 'em up!")
plt.show()


# In[32]:


colors = ['green' if x == 1 else 'red' for x in vt_labels]
D = PCA(n_components=1).fit_transform(domains.as_matrix(vt_cols))
print len(D)
DD = np.ndarray(shape=(len(D),2), dtype=float, order='F')
for i in range(0,len(D)):
    DD[i] = [D[i], 0.0]

figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(DD[:,0], DD[:,1], s=50, color=colors)
ax.set_title("Line 'em up!")
plt.show()


# ## Scaled Data
# One of the final things we can do with this information is scale the feature returned by PCA in this instance. This shifts the data so all values are between zero and one. Giving a really nice scale.
# 
# The case of both AV and blacklist is once again provided, perform the same operation/graph for AV only and blacklist only.

# In[33]:


D = PCA(n_components=1).fit_transform(domains.as_matrix(cols))
D = [(x - D.min())/(D.max() - D.min()) for x in D]
DD = np.ndarray(shape=(len(D),2), dtype=float, order='F')
for i in range(0,len(D)):
    DD[i] = [D[i], 0.0]

figsize(12,8)
fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(DD[:,0], DD[:,1], s=50, alpha=.5, color=colors)
ax.set_title("Normalized/Scaled between 0 and 1")
plt.show()


# ## Putting It All Together
# 
# After doing all that work to attempt to order and group data, it's time to make use of the results. Remember, that the labels *0* and *1* are arbitrary so it will take assigning the values back and you interpreting the data to understand what's going on.
# 
# Here's one of the ways to assign and look at domains. This is just for the AV and blacklist results, so you should do the same with the other labels/values.
# 
# When does this seem to work, when does it seem to fail? How valuable do you think this kind of technique is?

# In[34]:


D = PCA(n_components=1).fit_transform(domains.as_matrix(cols))
D = [(x - D.min())/(D.max() - D.min()) for x in D]
domains['bl_vt_scaled'] = D
domains[['host','bl_vt_scaled']].head()


# In[35]:


domains[domains['host'] == '0td4nbde7.ttl60.com'][['detections','detection_count','av_detections','av_count']]


# In[36]:


domains[domains['bl_vt_scaled'] == 1]['host'].unique()


# In[37]:


domains[domains['host'] == 'turningsbyterry.com'][['detections','detection_count','av_detections','av_count']]


# In[38]:


domains[domains['bl_vt_scaled'] == 0]['host'].unique()


# In[39]:


domains[domains['host'] == 'download.yspbrsz.net'][['detections','detection_count','av_detections','av_count']]


# In[40]:


D = PCA(n_components=1).fit_transform(domains.as_matrix(vt_cols))
D = [(x - D.min())/(D.max() - D.min()) for x in D]
domains['vt_scaled'] = D
domains[['host','vt_scaled']].head()


# In[41]:


domains[domains['host'] == '02b123c.netsolhost.com'][['av_detections','av_count']]


# In[42]:


domains[domains['host'] == '0td4nbde7.ttl60.com'][['av_detections','av_count']]


# In[43]:


domains[domains['vt_scaled'] == 1]['host'].unique()


# In[44]:


domains[domains['host'] == 'ww.turningsbyterry.com'][['av_detections','av_count']]


# In[45]:


domains[domains['vt_scaled'] == 0]['host'].unique()


# In[46]:


domains[domains['host'] == '1385065244.listentoy.com'][['av_detections','av_count']]


# In[47]:


D = PCA(n_components=1).fit_transform(domains.as_matrix(bl_cols))
D = [(x - D.min())/(D.max() - D.min()) for x in D]
domains['bl_scaled'] = D
domains[['host','bl_scaled']].head()


# In[48]:


domains[domains['host'] == '0lilioo0l0o00lilil.info'][['detections','detection_count']]


# In[49]:


domains[domains['host'] == '0n1u4og97yt85sy8faitxwt.addirectory.org'][['detections','detection_count']]


# In[50]:


domains[domains['bl_scaled'] == 1]['host'].unique()


# In[51]:


domains[domains['host'] == 'storylootybuz.com'][['detections','detection_count']]


# In[52]:


domains[domains['bl_scaled'] == 0]['host'].unique()


# In[53]:


domains[domains['host'] == '1174378403-6.ichers.ru'][['detections','detection_count']]

