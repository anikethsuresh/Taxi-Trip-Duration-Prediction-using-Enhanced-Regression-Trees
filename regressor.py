from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor, LinearRegression
from pyspark.ml.evaluation import *
from pyspark.ml.tuning import *

spark = SparkSession.builder \
    .master("local") \
    .appName("Decision Tree") \
    .getOrCreate()

schema = StructType([StructField("id", StringType(),False),
                    StructField("vendor_id",IntegerType(),False),
                    StructField("passenger_count",IntegerType(),False),
                    StructField("trip_duration",DoubleType(),False),
                    StructField("month",IntegerType(),False),
                    StructField("day",IntegerType(),False),
                    StructField("hour",IntegerType(),False),
                    StructField("weekday",IntegerType(),False),
                    StructField("distance",DoubleType(),False)])
print("Reading data")
data = spark.read.csv("hdfs:///user/asuresh2/assignment2/cleaned.csv",schema=schema, header=True)
print("Data read")
# Implement my own evaluator:
# Since the Cross validator maximises the loss. I will minimize it lol
class myRmseEvaluator():
    def __init__(self, oldEval):
        self.oldEval = oldEval
    
    def evaluate(self, dataset):
        return -self.oldEval.evaluate(dataset)
        
    # It would make sense to implement only the following function, rather than 
    # returning the result from the negative evaluate function
    def isLargerBetter(self):
        return True


"""
REGRESSION
"""

model_data = data.drop("id")
vectorizer = VectorAssembler().setInputCols(["vendor_id","passenger_count","month","day","hour","weekday","distance"]).setOutputCol("features")
model_data = vectorizer.transform(model_data).select("features","trip_duration")

# Split the data 70-30
train_test_data = model_data.randomSplit([0.8,0.2],16430212)
train_data = train_test_data[0]
test_data = train_test_data[1]

print("Train DT")
rmseEvaluator = myRmseEvaluator(RegressionEvaluator(predictionCol="prediction",labelCol="trip_duration", metricName="rmse"))
maeEvaluator = RegressionEvaluator(predictionCol="prediction",labelCol="trip_duration", metricName="mae")
dtr = DecisionTreeRegressor(maxDepth=3).setFeaturesCol("features").setLabelCol("trip_duration")
trained_model = dtr.fit(train_data)
predictions = trained_model.transform(test_data)

# final_result = predictions.select("prediction", "trip_duration").rdd
print(trained_model)
print("RMSE for Regression Tree:", rmseEvaluator.evaluate(predictions))
print("MAE for Regression Tree:", maeEvaluator.evaluate(predictions))

"""
DecisionTreeRegressionModel: uid=DecisionTreeRegressor_826b1c042824, depth=3, numNodes=15, numFeatures=7
  If (feature 6 <= 2.7002733639393384)
   If (feature 6 <= 1.3071311166631614)
    If (feature 6 <= 0.825910208978972)
     Predict: 481.0347939172201
    Else (feature 6 > 0.825910208978972)
     Predict: 704.5021037177617
   Else (feature 6 > 1.3071311166631614)
    If (feature 6 <= 1.8821559421975278)
     Predict: 894.6080785101466
    Else (feature 6 > 1.8821559421975278)
     Predict: 1121.6659710628826
  Else (feature 6 > 2.7002733639393384)
   If (feature 6 <= 9.536082191310886)
    If (feature 6 <= 5.077256060808084)
     Predict: 1422.6793786966068
    Else (feature 6 > 5.077256060808084)
     Predict: 1959.3655415697908
   Else (feature 6 > 9.536082191310886)
    If (feature 4 <= 19.5)
     Predict: 3028.8669666598485
    Else (feature 4 > 19.5)
     Predict: 2309.267728834739
"""
# Get the count for distinct output classes
distinct_classes = predictions.select("prediction").distinct()
distinct_classes_count = distinct_classes.count()
print("Number of Distinct classes:" ,distinct_classes_count)

all_data_through_model = trained_model.transform(model_data)
(train_data, test_data) = all_data_through_model.randomSplit([0.8,0.2])


dict_lin_reg = {}
best_lin_reg = {}
output = {}
for i in distinct_classes.collect():
    print("Currently running for:" , i[0])
    required_dataframe = train_data.filter(train_data.prediction == i[0]).drop("prediction")
    temp_lin_reg = LinearRegression().setFeaturesCol("features").setLabelCol("trip_duration")
    grid_builder = ParamGridBuilder() \
        .addGrid(temp_lin_reg.regParam,[0.5,1,100,1000]) \
        .addGrid(temp_lin_reg.elasticNetParam,[0.2,0.5,0.8,1]) \
        .addGrid(temp_lin_reg.epsilon,[2,3,5,9,50]) \
        .addGrid(temp_lin_reg.maxIter,[10, 20, 50, 75]) \
        .build()
    cross_validator = CrossValidator(estimator=temp_lin_reg,estimatorParamMaps=grid_builder,evaluator=rmseEvaluator,numFolds=10)
    cv_model = cross_validator.fit(required_dataframe)
    dict_lin_reg[i[0]] = cv_model
    best_lin_reg[i[0]] = cv_model.bestModel
    output[i[0]] = best_lin_reg[i[0]].transform(test_data.filter(test_data.prediction == i[0]).drop("prediction"))
    print("RMSE for",i[0]," = ",rmseEvaluator.evaluate(output[i[0]]))

results = list(output.values())
final = results.pop(0)
for elements in results:
    final = final.union(elements).distinct()

print("RMSE for Enhanced Regression Tree: ",rmseEvaluator.evaluate(final))
print("MAE for Enhanced Regression Tree: ",maeEvaluator.evaluate(final))