import numpy as np
import textblob as tb
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import tflearn
#from tflearn.datasets import
import nltk

def handleClean():
    """
    Handles the stemmed and cleaned tweets
    :return: list of tweets
    """
    CLEANDATA = "data_clean_stemmed_withoutRT.csv"
    cleandata = pd.read_csv(CLEANDATA, index_col=0)

    stemmedtweets = []

    for tweet in cleandata["text"]:
        tweet = tb.TextBlob(tweet) # preferred format
        stemmedtweets.append(tweet)
    return stemmedtweets

# fetch
DATAFILE = "data_original.csv"
data = pd.read_csv(DATAFILE, index_col=0)

n = len(data)
tweets = []
vec = []    # clustering vectors

positive = 0
negative = 0
subjectivity50 = 0
subjectivity70 = 0
for tweet in data["text"]:
    tweet = tb.TextBlob(tweet) # preferred format
    tweets.append(tweet)
    vec.append([tweet.polarity, tweet.subjectivity])

    #percentages
    if( tweet.polarity > 0):
        positive += 1
    if( tweet.subjectivity > 0.5):
        subjectivity50 += 1
    if( tweet.subjectivity >= 0.7):
        subjectivity70 += 1
    elif (tweet.polarity < 0):
        negative += 1

print("Positive tweets: ", round(positive/n, 3)," Negative tweets: ", round(negative/n, 3), "Neutral tweets: ", round(positive/n - negative/n, 3) )
print("Amount of subjective tweets: ", round(subjectivity50/n, 3), "Amount of very subjective tweets: ", round(subjectivity70/n, 3))

def findK():
    """
    finds optimal k with elbow method
    """
    distortions = []
    K = range(1,10)
    for k in K:
        vecFitted, clustLabels, cent, kmeans = cluster(k, vec)
        distortions.append(sum(np.min(cdist(vecFitted, cent, 'euclidean'), axis=1)) / vecFitted.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
#seems k = 4 is an elbow


def cluster(n_clusters, vec):
    """
    :param n_clusters: number of clusters
    :param vec: vectors to be clustered
    :return: clustered vec by Kmeans
    """
    vec = np.asarray(vec)
    model = KMeans(n_clusters)
    vecFitted = vec
    model.fit(vecFitted)
    clustLabels = model.predict(vecFitted)
    cent = model.cluster_centers_

    kmeans = pd.DataFrame(clustLabels)
    #vecFitted.insert((vec.shape[1]),'kmeans',kmeans)
    return vecFitted, clustLabels, cent, kmeans


def plotOriginal():
    # plotting original data
    npvec = np.asarray(vec) #need np array for slicing
    plt.scatter(npvec[:, 0], npvec[:, 1])
    plt.title("Scatterplot of original data")
    plt.xlabel("Polarity")
    plt.ylabel("Subjectivity")
    plt.show()


def clusterDensity(kmeans):
    """
    :param kmeans: takes in the data assigned to each cluster
    :return: amount of tweets in each cluster, and the density of each cluster.
    """
    clusters = [0, 0, 0, 0]
    tot = len(kmeans)
    for row in range(len(kmeans)):
        index = kmeans.iat[row, 0]
        clusters[index] += 1
    print("Number of tweets in each cluster: ", clusters)
    print("densities in clusters: c0 = ", round(clusters[0] / tot,3), " c1 = ", round(clusters[1] / tot,3), " c2 = ",
          round(clusters[2] / tot,3), " c3 = ", round(clusters[3] / tot),3)
    # print(kmeans)


def findMostCommon(dict):
    """
    Plots the most common words of each cluster
    """
    freqlist = sorted(dict.values())  # amount of times a word is used
    keys = []
    values = np.zeros(14)
    for i in range(14, 0, -1):
        value = freqlist[-i]
        values[i - 1] = value
        for item in dict.items():
            if item[1] == value:
                keys.append(item[0])

    keys = list(set(reversed(keys)))

    if len(keys) != len(values):
        cut = len(values)
        keys = keys[:cut]

    return (keys, values)


def wordFrequencyClusters(kmeans):
    """
    :return: The most common words of each cluster.
    """
    stemmed = handleClean()
    clustersTweets = ['', '', '', '']

    for row in range(len(kmeans)):
        index = kmeans.iat[row, 0]  # finds correct cluster
        twe = stemmed[row]
        twe = twe.replace('[', ' ')
        twe = twe.replace(']', ' ')
        clustersTweets[index] += str(twe)

    """
    Now all stemmed tweets will be sorted into the clusters they belong, and we can do wordcount 
    """
    word_counts0 = tb.TextBlob(clustersTweets[0]).word_counts  # these are dictionaries
    word_counts1 = tb.TextBlob(clustersTweets[1]).word_counts
    word_counts2 = tb.TextBlob(clustersTweets[2]).word_counts
    word_counts3 = tb.TextBlob(clustersTweets[3]).word_counts

    words0, freq0 = findMostCommon(word_counts0)
    plt.bar(words0, freq0)
    plt.ylabel("Frequency of cluster 0")
    plt.show()
    words1, freq1 = findMostCommon(word_counts1)
    plt.bar(words1, freq1)
    plt.ylabel("Frequency of cluster 1")
    plt.show()
    words2, freq2 = findMostCommon(word_counts2)
    plt.bar(words2, freq2)
    plt.ylabel("Frequency of cluster 2")
    plt.show()
    words3, freq3 = findMostCommon(word_counts3)
    plt.bar(words3, freq3)
    plt.ylabel("Frequency of cluster 3")
    plt.show()


def plotClustered():
    # plotting clustered
    vecFitted, clustLabels, cent, kmeans = cluster(4, vec)
    clusterDensity(kmeans)
    fig = plt.scatter(vecFitted[:,0],vecFitted[:,1],c=kmeans[0],s=50)
    plt.colorbar(fig)
    plt.title("Scatterplot of clustered data")
    plt.xlabel("Polarity")
    plt.ylabel("Subjectivity")
    plt.show(fig)
    wordFrequencyClusters(kmeans)


if __name__ == '__main__':
    plotOriginal() #Here is the original data with sentiment and polarity analysis
    findK() #finds how many clusters we need
    plotClustered() # we now plot the clustered dataset in the sentiment and polarity graph.  And how many tweets in each cluster ?
                    # Also shows the words frequency of each cluster.
