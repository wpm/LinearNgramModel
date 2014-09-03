Linear N-gram Model Serialization
=================================

This package illustrates how to use an n-gram model created with a non-Java toolkit in a Java program.

The *train_linear_model.py* uses the [scikit-learn](http://scikit-learn.org/stable/) toolkit to train a multinomial naive Bayes n-gram classifier with tf-idf features.
It serializes the model as a zipped JSON file containing model parameters, a vocabulary, and inverse document frequency scores.
The training data is in a tab-delimited file with integer labels followed by text. Each line of text is treated as a single document.

    0	The horse and the cow lived on the farm
    1	Boil two eggs for five minutes
    0	The hayloft of the barn was full
    1	Drain the pasta
   
The Java *ApplyModel* program deserializes the model, tokenizes the test data, generates tf-idf vectors based on the serialized idf table, and calculates class log likelihoods.
Its output looks like this.

    0	-47.8674 -47.1280	The harvest was finished early this year 	
    0	-47.0950 -42.8352	We fed the horses and the pigs
    1	-45.3605 -46.8341	Place the garlic in a pan

This depends on the [tfidf](https://github.com/wpm/tfidf) Java package which you must install from github.
