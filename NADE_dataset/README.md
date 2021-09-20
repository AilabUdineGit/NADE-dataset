## NADE
This folder contains all the NADE dataset.

Files and folder description:
* *smm4h19_augmented.csv*: this file contains all the tweets of the test and train sets. Due to Twitter policy, we can't share the tweets of SMM4H. In *smm4h19_augmented.csv* we keep only the artificially negated tweets, but in order to run the pipeline and use the NADE dataset you must download the entire [SMM4H19](https://healthlanguageprocessing.org/smm4h19/) and [SMM4H20](https://healthlanguageprocessing.org/smm4h/) datasets (both for NER and classification task);
* *artificially_negated_tweets.txt*: the tweet ids of the artificially negated tweets.
* **test_set**: contains all the tweet ids divided by tweet type: *ade*, *noade* and *negade*. 