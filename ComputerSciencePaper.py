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


def binaryMatrix(titles, modelID):
    wordCount = {}
    for product in titles:
        tokens = nltk.word_tokenize(product)
        for token in tokens:
            if token not in wordCount.keys():
                wordCount[token] = 1
            else:
                wordCount[token] += 1

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


def jaccard(a, b):
    union_of_vectors = np.logical_or(a, b)
    intersection_of_vectors = np.logical_and(a, b)
    measure = intersection_of_vectors.sum() / float(union_of_vectors.sum())
    return measure


def isPrime(n):
    for i in range(2, n):
        if (n % i) == 0:
            return False
    return True


def nearestPrime(k):
    potential_prime = k
    while isPrime(potential_prime) is False:
        potential_prime = potential_prime + 1
    return potential_prime


def h(a, b, x, m):
    return (a + b * x) % m


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


def printClusters(labels, modelID):
    print("Clusters with more then 1 products: ")
    for label in range(labels.max() + 1):
        cluster_products = list(np.where(labels == label))
        cluster_len = np.count_nonzero(labels == label)
        print("cluster ", label, " has ", cluster_len, " products: ", list(modelID[i] for i in cluster_products[0]))


def totalDuplicates(modelID):
    duplicates_total = 0
    n = len(modelID)
    for i in range(n):
        for j in range(i + 1, n):
            if modelID[i] == modelID[j]:
                duplicates_total = duplicates_total + 1
    return duplicates_total


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


def Bootstrap(titles, shop, modelID):
    n = len(titles)
    full_indices = list(range(n))

    bootstrap = np.random.choice(full_indices, size=n, replace=True)
    train_indices = np.unique(bootstrap)
    test_indices = [item for item in full_indices if item not in train_indices]

    titles_train = list(titles[i] for i in train_indices)
    titles_test = list(titles[i] for i in test_indices)

    modelID_train = list(modelID[i] for i in train_indices)
    modelID_test = list(modelID[i] for i in test_indices)

    shop_train = list(shop[i] for i in train_indices)
    shop_test = list(shop[i] for i in test_indices)

    return titles_train, titles_test, modelID_train, modelID_test, shop_train, shop_test


def Replicate(number_of_bootstraps, titles, shop, modelID, bands):
    sample_size = []
    F1_train = []
    PQ = []
    PC = []
    fraction_of_Comparison = []
    b = bands

    for i in range(number_of_bootstraps):
        titles_train, titles_test, modelID_train, modelID_test, shop_train, shop_test = Bootstrap(titles, shop, modelID)
        train_size = len(titles_train)
        sample_size.append(train_size)
        data_matrix_train, df_train, number_of_modelwords = binaryMatrix(titles_train, modelID_train)

        prime = nearestPrime(number_of_modelwords)
        signature_matrix_train = minHashing(data_matrix_train, prime, 123, 720)
        df2_train = pd.DataFrame(signature_matrix_train, columns=modelID_train)

        sig_matrix_train = df2_train.to_numpy()
        lsh_list_train, number_of_comparisons = LSH(sig_matrix_train, b)

        duplicates = []
        for i in lsh_list_train:
            if modelID_train[i[0]] == modelID_train[i[1]]:
                duplicates.append(modelID_train[i[0]])

        count_duplicates = len(np.unique(duplicates))
        total_possible_comparison = comb(train_size, 2)
        duplicates_total = totalDuplicates(modelID_train)

        fraction_of_Comparison.append(number_of_comparisons / total_possible_comparison)
        PQ.append(count_duplicates / number_of_comparisons)
        PC.append(count_duplicates / duplicates_total)

        dissimilarity_matrix = distanceMatrix(lsh_list_train, shop_train, data_matrix_train)
        df3_train = pd.DataFrame(dissimilarity_matrix, columns=modelID_train, index=modelID_train)
        # df3.to_excel("distancematrix.xlsx")
        dist_matrix_train = df3_train.to_numpy()

        hierachical_clustering_train = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                                               linkage="complete",
                                                               distance_threshold=0.4)
        clusters_train = hierachical_clustering_train.fit(dist_matrix_train)
        labels_train = clusters_train.labels_
        clusters_duplicates, clusters_comparison, precision, recall, f1 = validationMetrics(labels_train, modelID_train,
                                                                                            duplicates_total)
        F1_train.append(f1)

    return sample_size, fraction_of_Comparison, PC, PQ, F1_train


# Section 3-------------------------------------------------------------------------------------------------------------
"""
In this part of the code,
one bootstrap sample will be performed.
And some validation metrics including all the clusters will be printed in order to give the user insight into 
how the clusters are formed for 1 bootstrap.
Remove the triple quotation marks in line 295 and 412 to run this part of the code.
"""

"""
# Using bootstrap to get training en test sample
titles_train, titles_test, modelID_train, modelID_test, shop_train, shop_test = Bootstrap(titles, shop, modelID)
train_size = len(titles_train)
test_size = len(titles_test)
print("Train sample size is: ", train_size)
print("Test sample size is: ", test_size)
print()

# Optimize on train sample
data_matrix_train, df_train, number_of_modelwords = binaryMatrix(titles_train, modelID_train)
print("The below metrics are for the training sample: ")
print()
print("number of modelwords: ", number_of_modelwords)
print()

prime = nearestPrime(number_of_modelwords)
signature_matrix_train = minHashing(data_matrix_train, prime, 123, 720)
df2_train = pd.DataFrame(signature_matrix_train, columns=modelID_train)

sig_matrix_train = df2_train.to_numpy()
b = 720
lsh_list_train, number_of_comparisons = LSH(sig_matrix_train, b)

duplicates = []
for i in lsh_list_train:
    if modelID_train[i[0]] == modelID_train[i[1]]:
        duplicates.append(modelID_train[i[0]])

count_duplicates = len(np.unique(duplicates))
total_possible_comparison = comb(train_size, 2)
duplicates_total = totalDuplicates(modelID_train)

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

dissimilarity_matrix = distanceMatrix(lsh_list_train, shop_train, data_matrix_train)
df3_train = pd.DataFrame(dissimilarity_matrix, columns=modelID_train, index=modelID_train)
# df3.to_excel("distancematrix.xlsx")
dist_matrix_train = df3_train.to_numpy()

hierachical_clustering_train = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="complete",
                                                       distance_threshold=0.4)
clusters_train = hierachical_clustering_train.fit(dist_matrix_train)
labels_train = clusters_train.labels_
printClusters(labels_train, modelID_train)
clusters_duplicates, clusters_comparison, precision, recall, f1 = validationMetrics(labels_train, modelID_train,
                                                                                    duplicates_total)
print()
print("Number of duplicates found: ", clusters_duplicates)
print("Number of comparisons with clustering: ", clusters_comparison)
print("precision is: ", precision)
print("recall is: ", recall)
print("F1 measure is: ", f1)
print("---------------------------------------------------------------------------------------------------------------")

# performance on the test sample
data_matrix_test, df_test, number_of_modelwords_test = binaryMatrix(titles_test, modelID_test)
print("The below metrics are for the test sample: ")
print("number of modelwords: ", number_of_modelwords_test)
print()

prime = nearestPrime(number_of_modelwords_test)
signature_matrix_test = minHashing(data_matrix_test, prime, 123, 360)
df2_test = pd.DataFrame(signature_matrix_test, columns=modelID_test)

sig_matrix_test = df2_test.to_numpy()
b = 270
lsh_list_test, number_of_comparisons = LSH(sig_matrix_test, b)

duplicates = []
for i in lsh_list_test:
    if modelID_test[i[0]] == modelID_test[i[1]]:
        duplicates.append(modelID_test[i[0]])

count_duplicates = len(np.unique(duplicates))
total_possible_comparison = comb(test_size, 2)
duplicates_total = totalDuplicates(modelID_test)

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

dissimilarity_matrix = distanceMatrix(lsh_list_test, shop_test, data_matrix_test)
df3_test = pd.DataFrame(dissimilarity_matrix, columns=modelID_test, index=modelID_test)
# df3.to_excel("distancematrix.xlsx")
dist_matrix_test = df3_test.to_numpy()

hierachical_clustering_test = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="complete",
                                                      distance_threshold=0.4)
clusters_test = hierachical_clustering_test.fit(dist_matrix_test)
labels_test = clusters_test.labels_
printClusters(labels_test, modelID_test)
clusters_duplicates, clusters_comparison, precision, recall, f1 = validationMetrics(labels_test, modelID_test,
                                                                                    duplicates_total)
print()
print("Number of duplicates found: ", clusters_duplicates)
print("Number of comparisons with clustering: ", clusters_comparison)
print("precision is: ", precision)
print("recall is: ", recall)
print("F1 measure is: ", f1)
print("---------------------------------------------------------------------------------------------------------------")
"""

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

bootstrap = 5
number_of_bands = 360
average_fraction_of_comparison = []
average_PC = []
average_PQ = []
average_F1 = []

for i in range(0, number_of_bands+1, 60):
    band = i
    if band == 0: # number of bands should always be greater then 0.
        band = 1

    print(band)
    train_size, fraction_of_Comparison, PC, PQ, F1_bootstrap = Replicate(number_of_bootstraps=bootstrap, titles=titles,
                                                                         shop=shop,
                                                                         modelID=modelID, bands=band)
    average_fraction_of_comparison.append(mean(fraction_of_Comparison))
    average_PC.append(mean(PC))
    average_PQ.append(mean(PQ))
    average_F1.append(mean(F1_bootstrap))

#average_fraction_of_comparison.sort()
print(average_fraction_of_comparison)
print(average_PC)
print(average_PQ)
print(average_F1)

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