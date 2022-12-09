import collections
import itertools
import random
import re
import sys
import nltk
import heapq
import numpy as np
import pandas as pd
import json
from sklearn.cluster import AgglomerativeClustering
from math import comb
from statistics import mean
import matplotlib.pyplot as plt

# Section 1-------------------------------------------------------------------------------------------------------------
"""
In this section of the code, 
the Data is imported into Python and cleaned.
"""

with open('D:\\EBOOK Econometrie\\TVs-all-merged.json') as file:
    data = json.load(file)

titles = []
shop = []
modelID = []

for v in data:
    for i in data[v]:
        titles.append(i['title'])
        modelID.append(i['modelID'])
        shop.append(i['shop'])

replacement1 = {'Inch': 'inch', 'inches': 'inch', '"': 'inch', '-inch': 'inch', ' inch': 'inch', 'inch': 'inch'}
replacement2 = {'Hertz': 'hz', 'hertz': 'hz', 'Hz': 'hz', 'HZ': 'hz', ' hz': 'hz', '-hz': 'hz', 'hz': 'hz'}
pattern = r"([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)"

n = len(titles)
for i in range(n):
    for key, value in replacement1.items():
        titles[i] = titles[i].replace(key, value)
    for key, value in replacement2.items():
        titles[i] = titles[i].replace(key, value)
    titles[i] = titles[i].lower()
    titles[i] = re.sub(r'[()]', '', titles[i])
    tuple_list = re.findall(pattern, titles[i])
    modelWord = ''
    for j in tuple_list:
        modelWord = modelWord + j[0] + ' '
    titles[i] = modelWord

# Section 2-------------------------------------------------------------------------------------------------------------
"""
In this section of the code,
all the functions and methods are defined
including the replication method for different bootstraps.
"""

# This function takes model words as creates binary vectors for each product and puts them into a matrix.
def binaryMatrix(titles, modelID):
    wordCount = {}
    for product in titles:
        tokens = nltk.word_tokenize(product)
        for token in tokens:
            if token not in wordCount.keys():
                wordCount[token] = 1
            else:
                wordCount[token] += 1

    # sort the model words from most frequent to least frequent to make it easier to see in a dataframe
    # if the binary representations are correct for each product.
    most_common_words = heapq.nlargest(len(wordCount), wordCount, key=wordCount.get)

    vector_of_products = []
    for product in titles:
        sentence_tokens = nltk.word_tokenize(product)
        product_binary_vector = []
        for token in most_common_words:
            if token in sentence_tokens:
                product_binary_vector.append(1)
            else:
                product_binary_vector.append(0)
        vector_of_products.append(product_binary_vector)

    vector_of_products = np.asarray(vector_of_products)
    df = pd.DataFrame(np.transpose(vector_of_products), index=most_common_words, columns=modelID)
    data_matrix = df.to_numpy()
    return data_matrix, df, len(most_common_words)


# This function computes the jaccard similarity for 2 binary vectors.
def jaccard(a, b):
    union_of_vectors = np.logical_or(a, b)
    intersection_of_vectors = np.logical_and(a, b)
    measure = intersection_of_vectors.sum() / float(union_of_vectors.sum())
    return measure


# This function takes a number as input and checks if it is a prime.
def isPrime(n):
    for i in range(2, n):
        if (n % i) == 0:
            return False
    return True


# This function finds the nearest prime that is larger then the input number.
def nearestPrime(k):
    potential_prime = k
    while isPrime(potential_prime) is False:
        potential_prime = potential_prime + 1
    return potential_prime

# Hash function
def h(a, b, x, m):
    return (a + b * x) % m


# This function mainly takes a binary matrix and number of hashrows as input, then generates random hash functions
# according to the given hashrow number and prime. And returns a signature matrix as result of Min-Hashing.
def minHashing(data_matrix, prime, seed, number_of_hashrows):
    r, c, i = len(data_matrix), len(data_matrix[0]), number_of_hashrows  # int((0.4 * len(data_matrix)))
    random.seed(seed)
    a_list = random.sample(range(1, prime - 1), i)
    b_list = random.sample(range(0, prime - 1), i)
    m_value = prime

    signature_matrix = []
    for h_i in range(i):
        signature_matrix.append([sys.maxsize] * c)

    for rows in range(r):
        hashvalues_vector = []
        for h_i in range(i):
            hashvalue = h(a_list[h_i], b_list[h_i], rows + 1, m_value)
            hashvalues_vector.append(hashvalue)
        for columns in range(c):
            if data_matrix[rows][columns] == 0:
                continue
            for h_i in range(i):
                if hashvalues_vector[h_i] < signature_matrix[h_i][columns]:
                    signature_matrix[h_i][columns] = hashvalues_vector[h_i]
    return signature_matrix


# This function takes the signature matrix and the number of bands as input and uses Locality-Sensitive hashing
# to return a list with candidate pairs.
def LSH(sig_matrix, b):
    n = len(sig_matrix)
    p = len(sig_matrix[0])
    assert (b <= n)  # we cannot have more bands than hashes
    buckets = collections.defaultdict(set)
    bands = np.array_split(sig_matrix, b, axis=0)
    for i, band in enumerate(bands):
        for product_index in range(p):
            product_hash = list(band[:, product_index])
            minhash_vector = tuple(product_hash + [str(i)])  # adding the number of the band because
            buckets[minhash_vector].add(product_index)  # we do not want accidental collision
    candidate_pairs = set()  # between different bands.
    for bucket in buckets.values():
        bucket_length = len(bucket)
        if bucket_length > 1:
            for pairs in itertools.combinations(bucket, 2):
                candidate_pairs.add(pairs)
    return candidate_pairs, len(candidate_pairs)


# This function returns the dissimilarity matrix based on the candidate pairs given by LSH and put distance between
# products from same webshops on max.int
def distanceMatrix(lsh_list, shop, data_matrix):
    p = len(shop)
    lsh_matrix = np.zeros(shape=(p, p))
    for values in lsh_list:
        lsh_matrix[values[0]][values[1]] = 1
        lsh_matrix[values[1]][values[0]] = 1

    dissimilarity_matrix = []
    for i in range(p):
        dissimilarity_matrix.append([sys.maxsize] * p)

    for i in range(p):
        for j in range(p):
            if i != j and shop[i] != shop[j] and lsh_matrix[i][j] == 1:
                dissimilarity = 1 - jaccard(data_matrix[:, i], data_matrix[:, j])
                dissimilarity_matrix[i][j] = dissimilarity
    return dissimilarity_matrix


# This function takes the labels of the assigned clusters for all products as input and
# prints the found clusters with model ids in a nice way.
def printClusters(labels, modelID):
    for label in range(labels.max() + 1):
        cluster_products = list(np.where(labels == label))
        cluster_len = np.count_nonzero(labels == label)
        print("cluster ", label, " has ", cluster_len, " products: ", list(modelID[i] for i in cluster_products[0]))


# This function calculates the total pair of duplicates for a given list of model ids.
def totalDuplicates(modelID):
    duplicates_total = 0
    n = len(modelID)
    for i in range(n):
        for j in range(i + 1, n):
            if modelID[i] == modelID[j]:
                duplicates_total = duplicates_total + 1
    return duplicates_total


# This function computes the performance measures.
def validationMetrics(labels, modelID, duplicates_total):
    number_of_comparisons_cluster = 0
    duplicates_found_cluster = 0

    for label in range(labels.max() + 1):
        cluster_products = list(np.where(labels == label))
        cluster_len = np.count_nonzero(labels == label)
        number_of_comparisons_cluster = number_of_comparisons_cluster + comb(cluster_len, 2)
        for i in range(cluster_len):
            for j in range(i + 1, cluster_len):
                if modelID[cluster_products[0][i]] == modelID[cluster_products[0][j]]:
                    duplicates_found_cluster = duplicates_found_cluster + 1
    precision = duplicates_found_cluster / number_of_comparisons_cluster
    recall = duplicates_found_cluster / duplicates_total
    f1 = (2 * precision * recall) / (precision + recall)

    return duplicates_found_cluster, number_of_comparisons_cluster, precision, recall, f1


# This function will return randomly drawn bootstrap samples from a given input of products.
def Bootstrap(titles, shop, modelID):
    n = len(titles)
    full_indices = list(range(n))

    bootstrap = np.random.choice(full_indices, size=n, replace=True)
    bootstrap_indices = np.unique(bootstrap)

    titles_bootstrap = list(titles[i] for i in bootstrap_indices)
    modelID_bootstrap = list(modelID[i] for i in bootstrap_indices)
    shop_bootstrap = list(shop[i] for i in bootstrap_indices)

    return titles_bootstrap, modelID_bootstrap, shop_bootstrap


# This function will run the LSH and hierachical clustering for a given number of bootstraps
# and returns the performance measures for each bootstrap in a list.
def Replicate(number_of_bootstraps, titles, shop, modelID, bands):
    sample_size = []
    F1 = []
    PQ = []
    PC = []
    fraction_of_Comparison = []
    b = bands

    for i in range(number_of_bootstraps):
        titles_bootstrap, modelID_bootstrap, shop_bootstrap = Bootstrap(titles, shop, modelID)
        bootstrap_size = len(titles_bootstrap)
        sample_size.append(bootstrap_size)
        data_matrix_bootstrap, df_bootstrap, number_of_modelwords = binaryMatrix(titles_bootstrap, modelID_bootstrap)

        prime = nearestPrime(number_of_modelwords)
        signature_matrix_bootstrap = minHashing(data_matrix_bootstrap, prime, 123, 720)
        df2_bootstrap = pd.DataFrame(signature_matrix_bootstrap, columns=modelID_bootstrap)

        sig_matrix_bootstrap = df2_bootstrap.to_numpy()
        lsh_list_bootstrap, number_of_comparisons = LSH(sig_matrix_bootstrap, b)

        duplicates = []
        for i in lsh_list_bootstrap:
            if modelID_bootstrap[i[0]] == modelID_bootstrap[i[1]]:
                duplicates.append(modelID_bootstrap[i[0]])

        count_duplicates = len(np.unique(duplicates))
        total_possible_comparison = comb(bootstrap_size, 2)
        duplicates_total = totalDuplicates(modelID_bootstrap)

        fraction_of_Comparison.append(number_of_comparisons / total_possible_comparison)
        PQ.append(count_duplicates / number_of_comparisons)
        PC.append(count_duplicates / duplicates_total)

        dissimilarity_matrix = distanceMatrix(lsh_list_bootstrap, shop_bootstrap, data_matrix_bootstrap)
        df3_bootstrap = pd.DataFrame(dissimilarity_matrix, columns=modelID_bootstrap, index=modelID_bootstrap)
        dist_matrix_bootstrap = df3_bootstrap.to_numpy()

        hierachical_clustering_bootstrap = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                                                   linkage="complete",
                                                                   distance_threshold=0.4)
        clusters_bootstrap = hierachical_clustering_bootstrap.fit(dist_matrix_bootstrap)
        labels_bootstrap = clusters_bootstrap.labels_
        clusters_duplicates, clusters_comparison, precision, recall, f1 = validationMetrics(labels_bootstrap,
                                                                                            modelID_bootstrap,
                                                                                            duplicates_total)
        F1.append(f1)

    return sample_size, fraction_of_Comparison, PC, PQ, F1


# Section 3-------------------------------------------------------------------------------------------------------------
"""
In this part of the code,
one bootstrap sample will be performed.
And some validation metrics including all the clusters will be printed in order to give the user insight into 
how the clusters are formed for 1 bootstrap.
"""

# Using bootstrap to get bootstrap sample
titles_bootstrap, modelID_bootstrap, shop_bootstrap = Bootstrap(titles, shop, modelID)
bootstrap_size = len(titles_bootstrap)
b = 122
print("Below are the results for one bootstrap sample with only around 5-6% of total comparison:")
print()
print("Bootstrap sample size is: ", bootstrap_size)


# Optimize on bootstrap sample
data_matrix_bootstrap, df_bootstrap, number_of_modelwords = binaryMatrix(titles_bootstrap, modelID_bootstrap)
print("number of model words: ", number_of_modelwords)
print()

prime = nearestPrime(number_of_modelwords)
signature_matrix_bootstrap = minHashing(data_matrix_bootstrap, prime, 123, 720)
df2_bootstrap = pd.DataFrame(signature_matrix_bootstrap, columns=modelID_bootstrap)

sig_matrix_bootstrap = df2_bootstrap.to_numpy()
lsh_list_bootstrap, number_of_comparisons = LSH(sig_matrix_bootstrap, b)

duplicates = []
for i in lsh_list_bootstrap:
    if modelID_bootstrap[i[0]] == modelID_bootstrap[i[1]]:
        duplicates.append(modelID_bootstrap[i[0]])

count_duplicates = len(np.unique(duplicates))
total_possible_comparison = comb(bootstrap_size, 2)
duplicates_total = totalDuplicates(modelID_bootstrap)

print("total possible comparisons: ", total_possible_comparison)
print("number of comparisons: ", number_of_comparisons)
print("fraction of comparison is: ", number_of_comparisons / total_possible_comparison)
print()
print("total duplicates: ", duplicates_total)
print("duplicates found: ", count_duplicates)
print("pair completeness is: ", count_duplicates / duplicates_total)
print()
print("pair quality is: ", count_duplicates / number_of_comparisons)
print()

dissimilarity_matrix = distanceMatrix(lsh_list_bootstrap, shop_bootstrap, data_matrix_bootstrap)
df3_bootstrap = pd.DataFrame(dissimilarity_matrix, columns=modelID_bootstrap, index=modelID_bootstrap)
dist_matrix_bootstrap = df3_bootstrap.to_numpy()

hierachical_clustering_bootstrap = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                                           linkage="complete",
                                                           distance_threshold=0.4)
clusters_bootstrap = hierachical_clustering_bootstrap.fit(dist_matrix_bootstrap)
labels_bootstrap = clusters_bootstrap.labels_
printClusters(labels_bootstrap, modelID_bootstrap)
clusters_duplicates, clusters_comparison, precision, recall, f1 = validationMetrics(labels_bootstrap, modelID_bootstrap,
                                                                                    duplicates_total)
print()
print("Number of duplicates found: ", clusters_duplicates)
print("Number of comparisons with clustering: ", clusters_comparison)
print("precision is: ", precision)
print("recall is: ", recall)
print("F1 measure is: ", f1)
print("---------------------------------------------------------------------------------------------------------------")

# Section 4-------------------------------------------------------------------------------------------------------------
"""
In this part of the code,
simulation over the defined number of bootstraps will be run for a variation of fraction of comparisons.
And the average scores over all bootstraps are returned.
"""


# The code below calculates the average performance measures over a defined number of bootstraps and stores them in
# a list for different number of bands. A higher number of bands leads to more fraction of comparison.
# Note that the size of bootstrap sample will be around 1000 and 720 rows are used for minhashing,
# hence the number of bands should not exceed 720.

print("Below are the average performance measures over 5 bootstraps for different fraction of comparisons")

bootstrap = 5
number_of_bands = 360
average_bootstrap_size = []
average_fraction_of_comparison = []
average_PC = []
average_PQ = []
average_F1 = []

for i in range(0, number_of_bands+1, 60):
    band = i
    if band == 0: # number of bands should always be greater then 0.
        band = 1

    bootstrap_size, fraction_of_Comparison, PC, PQ, F1_bootstrap = Replicate(number_of_bootstraps=bootstrap, titles=titles,
                                                                         shop=shop, modelID=modelID, bands=band)
    average_bootstrap_size.append(mean(bootstrap_size))
    average_fraction_of_comparison.append(mean(fraction_of_Comparison))
    average_PC.append(mean(PC))
    average_PQ.append(mean(PQ))
    average_F1.append(mean(F1_bootstrap))

for i in range(len(average_fraction_of_comparison)):
    for j in range(i+1, len(average_fraction_of_comparison)):
        if average_fraction_of_comparison[i] > average_fraction_of_comparison[j]:
            average_fraction_of_comparison[j], average_fraction_of_comparison[i] = average_fraction_of_comparison[i],\
                                                                                   average_fraction_of_comparison[j]
            average_PC[j], average_PC[i] = average_PC[i], average_PC[j]
            average_PQ[j], average_PQ[i] = average_PQ[i], average_PQ[j]
            average_F1[j], average_F1[i] = average_F1[i], average_F1[j]

print("Average fraction of comparison: ",average_fraction_of_comparison)
print("Average pair completeness: ", average_PC)
print("Average pair quality: ", average_PQ)
print("Average F1-measure", average_F1)

"""
plt.plot(average_fraction_of_comparison, average_PC)
plt.ylabel('Pair completeness')
plt.xlabel('Fraction of comparisons')
plt.show()

plt.plot(average_fraction_of_comparison, average_PQ)
plt.ylabel('Pair quality')
plt.xlabel('Fraction of comparisons')
plt.show()

plt.plot(average_fraction_of_comparison, average_F1)
plt.ylabel('F1-measure')
plt.xlabel('Fraction of comparisons')
plt.show()
"""
