# CS-for-Business-Analytics

# Scalable Product Duplicate Detection 
> The number of Web shops has grown rapidly over the past decades and a great variety of products has been available online. However, different Web shops use different representations for the same product. Hence, it can be very time consuming for customers to find their most preferred product for the best combination of price and specifications. It is greatly beneficial for consumers if data  from  different 
Web  shops  could  be  aggregated  to  form  a  more  complete  set  of  product  information. To  automatically  aggregate  data  from  various  Web 
sites,  it is necessary to perform duplicate detection. Due to the ever increasing size of the data,  the  scalability issue for duplicate product detection must be adressed. In this project, a scalability solution for the involved computations by reducing the number of comparisons using Locality Sensitive  Hashing (LSH) in combination with Hierarchical clustering is proposed for finding duplicates. The python codes for this project can be found in the ComputerSciencePaper.py file. In this readme, a simple explantion to the structure of the code will be given. And a brief guide on how to effectively use the code with some screenshots from the results if the code has ran properly.

## Table of Contents
* [Code structure](#technologies-used)
* [Code usage guide](#features)
* [Screenshots from results](#screenshots)
<!-- * [License](#license) -->


## Code structure
The code is divided into 4 sections and each section is separated with a section number and dash line. An example is: 
" # Section 2-------------------------------------------------------------------------------------------------------------"

- In Section 1 of the code, the data is imported and cleaned to increase the effectiveness of the algorithm. The cleaning procedure is explained under methods in the paper.
- In Section 2 of the code, all the neccessary methods and function are created. The core methods to look for are `binaryMatrix`, `minHashing`, `LSH` and `Replicate`. Under `Replicate` the LSH and clustering are performed.
- In Section 3 of the code, the algorithms are performed and results are calculated over 1 bootstrap sample, and the structure of the clusters are also printed. This section is interesting if the user wants to have a very detailed insight into how the clusters are formed and the different metrics of the evaluation measure. It can also be used for the user to check the quality of the clusters.
- In section 4, the average performance measures over different bootstraps are computed for a variety of fraction of comparisons. Run this section if you are only interested in the final results.

## Code usage guides
In order to use the code, first make sure the packages `collections`, `itertools`, `random`, `re`, `sys`, `nltk`, `heapq`, `numpy`, `pandas`, `json`, `sklearn.cluster`, `math`, `statistics` and `matplotlib.pyplot` are installed correctly on the python IDE. And make sure the data is imported from the correct filepath. If these steps are correctly done, then the code can be executed without any further modifications to give the replication of the results shown in the paper. Do notice that there are some randomnes involved due to random bootstrapping hence the results can not be 100% replicated. In case the user is interested in the results for different inputs, read the steps below:

- Section 1 and 2 of the code do not need modification as they are predefined. 
- For section 3, if one wants to see different outputs for one bootstrap sample. He can modify the input for the part `b = 122` in line 309 of the code in order to run for different fraction of comparisons. To change the clustering structure, change the parameter distance_threshold of `hierachical_clustering_bootstrap = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="complete", distance_threshold=0.4)` in line 353.
- If the user wants to use a different number of bootstraps in section 4, just modifty `bootstrap = 5` in line 382. 

## Screenshots from results
If Section 3 of the code is performed correctly, it should give the user output like this:
![Screenshot_3](https://user-images.githubusercontent.com/113337636/206700119-d844b19c-68e5-4e7e-acdc-83901c0f030d.png)
![Screenshot_1](https://user-images.githubusercontent.com/113337636/206700154-966eb502-1320-43b1-9c90-7c1d11fef838.png)
If Section 4 of the code is performed correctly, an example output should be given like this:
![image](https://user-images.githubusercontent.com/113337636/206701576-18eada10-874a-4a38-a87b-fb79e5a76b05.png)




