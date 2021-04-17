# Word Count and Pair Count Analysis
 **Predicting Taxi Trip Duration using Regression Trees (and Enhanced Regression Trees)**

 The goal of this assignment was to perform
 regression on the NYC taxi dataset.

 *I would suggest going through the Report to get the links to download the data, understand the code, the assignment and overall function of each file* 

## Dataset Description (from Kaggle)
The [competition dataset](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) is based on the 2016 NYC Yellow Cab trip record data made available in Big Query on Google Cloud Platform. The data was originally published by the NYC Taxi and Limousine Commission (TLC). The data was sampled and cleaned for the purposes of this playground competition. Based on individual trip attributes, participants should predict the duration of each trip in the test set.

## Directory Structure:
**/Cleaner.py**: Cleans the data. Steps are detailed in the Report 

**/command line.txt**: Results obtained by running spark on the command line

**/regressor.py**: Regression program in python (spark)

**/Report.pdf**: Report of the project done
 
## Modelling and Algorithms
1.	### Regression Tree Model:
    A decision tree (regressor) was trained, validated (10 folds) and tested using 80% for training and 20% for testing. While you won’t see the specific code for validation, and feature selection (since there are only a few parameters with which it can be tuned), I manually performed the validation, which resulted in the best model with depth set to 3. With larger values of depth, the RMSE (Root mean squared error)/MAE (mean absolute error) either overfits or produces worse results. Selecting depth of 3 also resulted in only 8 leaves, which allowed for easier testing/debugging.


2.	### Enhanced Regression Tree Model
    An enhanced regression tree was constructed to predict the trip_duration. Similar to the regression tree model, the enhanced regression tree was also trained, validated and tested. Manual testing was not possible since there were a few parameters to tune. The parameters that were tuned were, epsilon, maxIter, regParam and elasticNetParam. Therefore, we can see that regularization was used in this tree model. 80% of the data was used as the training set with cross validation (10 folds) and the remaining was used as the test set.


## Conclusion
The Regression tree performed quite decently with the data given. Without any data exploration and feature extraction, the decision tree performed with an RMSE of about ~6000, which reduced to ~3000 after the case (not shown in the code. Was computed earlier). The Enhanced Decision Tree performed better than the original tree in some cases; this could be due to the randomness added due to the random split of the data. In some cases, even the original tree performed better. 
