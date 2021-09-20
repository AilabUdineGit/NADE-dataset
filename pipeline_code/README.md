
## Pipeline guidelines
* Insert a pickle file with the evaluated sentences (with columns ```[tweet_id, tweet]```) in the ```datasets/input/to_evaluate``` directory;
* in the ```datasets/input/model_predictions``` put the prediction file with columns ```[tweet_id, start, end, token]``` (each evaluated sentence might have more than one prediction);
* in ```src/config.py``` in the enum element ```TO_EVALUATE``` insert the name of the file that has been put in ```datasets/input/to_evaluate``` directory (without ```.pickle``` and inside a list, even if is a single element);
* at the same way, put the names of the files that have been put in ```datasets/input/model_predictions``` in the enum field ```MODELS_TO_EVALUATE```.
* the results will be available in the ```datasets/output/n``` directory, where ```n``` is the first available number.