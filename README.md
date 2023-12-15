The computerscience.py file contains the main program of the research. 
This file references the other files when using their functions, such as getting the character or signature matrix. 

**Computerscience.py**
At first, some initial values are specified, such as the jaccard similarity and the amount of permutations.
Then, the data is obtained from the .json file, after which the key-value pairs are split up into more convenient lists.
The bootstraps (**see get_bootstrap.py**) are started in a for loop, saving the results for each round in a list. 

The titles for each of the products in the dataset are used for the classification procedure. 
Therefore, at first steps are taken to make their way of representing information as similar as possible.
Also, information is extracted from the titles, such as the specifications of the product (inch, hertz, resolution) and the estimated model id and estimated brand.
Then, the 'cleaned' titles are used to obtain all the words that occur in them, to make the title words list. 

Using the title words, the following steps are taken in this order:
1. Construct the character matrix (**see character_matrix.py**)
2. Construct the signature matrix (**see signature_matrix.py**)
3. The candidate pairs matrix is made (**see lsh.py**)

Then the logic-based additional classification is done. 
1. Comparison of estimated model ids
2. Comparison of estimated brands
3. Comparison of website
4. Jaccard similarity calculation, to determine if this is above or below a pre-specified threshold (**see jaccard_similarity**)
5. Comparison of the specifications of the televisions

Then, the correct matrix representing what tvs are actually duplicates is constructed.

Lastly, based on the correct matrix and the candidate pairs matrix the TP, TN, FP, and FN are calculated (**see results.py**), which are used in the precision, recall, and F1*-score calculation.
These results are saved for each bootstrap round. 

When all bootstrap rounds have concluded, the mean for each of the metrics is calculated, which is used as a final result. 

