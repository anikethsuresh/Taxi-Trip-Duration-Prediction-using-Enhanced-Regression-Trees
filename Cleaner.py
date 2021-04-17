from pyspark.sql.types import *
from pyspark.sql.functions import *
from geopy.distance import *

spark = SparkSession.builder \
    .master("local") \
    .appName("Decision Tree") \
    .getOrCreate()
schema = StructType([StructField("id", StringType(),False),
                    StructField("vendor_id",IntegerType(),False),
                    StructField("pickup_datetime",TimestampType(), False), #TimestampType
                    StructField("dropoff_datetime",TimestampType(),False),
                    StructField("passenger_count",IntegerType(),False),
                    StructField("pickup_longitude",DoubleType(),False),
                    StructField("pickup_latitude",DoubleType(),False),
                    StructField("dropoff_longitude",DoubleType(),False),
                    StructField("dropoff_latitude",DoubleType(),False),
                    StructField("store_and_fwd_flag",StringType(),False),
                    StructField("trip_duration",DoubleType(),False)])
csv_data = spark.read.csv("hdfs:///user/asuresh2/assignment2/train.csv",schema=schema, header=True)
# TODO The nullable field is not working. Take a look at this.

"""
Dataset contains 1458644 records.
Validate dataset
"""
assert csv_data.count() == 1458644
# ID
# Are there duplicate IDS?
assert csv_data.select("id").distinct().count() == 1458644

# VENDOR ID
# What is the distribution of vendors?
print("Count: Vendor 1 -->",csv_data.filter(csv_data.vendor_id == 1).count())
print("Count: Vendor 2 -->",csv_data.filter(csv_data.vendor_id == 2).count())

# PASSENGER_COUNT
csv_data.groupBy("passenger_count").count().show()
# Interestingly there were 60 trips with no passengers
# Retrieving the trip details for these 60 trips
csv_data.filter("passenger_count==0").show()
# Deleting the trips with passenger_count=0
csv_data = csv_data.filter("passenger_count>0")
# Check again that the values are deleted
csv_data.groupBy("passenger_count").count().show()

# STORE_AND_FWD_FLAG
# We shall remove this column as it poses no use to us
csv_data = csv_data.drop("store_and_fwd_flag")

# PICKUP AND DROPOFF DATETIME
# Since the trip_duration can be directly calculated from the difference between
# pickup_datetime and dropoff_datetime, We check if there are values where this
# condition is not true
seconds = csv_data.select("id","pickup_datetime","dropoff_datetime").rdd

def convert_datetime_difference_seconds(x):
    y = x[2]-x[1]
    return (x[0],y.seconds)

seconds = seconds.map(convert_datetime_difference_seconds)
seconds = spark.createDataFrame(seconds).withColumnRenamed("_1","id").withColumnRenamed("_2","time_difference")

test_time_condition = csv_data.join(seconds,"id","inner").rdd
# test_time_condition = test_time_condition.select("id","time_difference","trip_duration").rdd
# test_time_condition = test_time_condition.filter(lambda x: x[1] !=x[2])
# test_time_condition.count()
# csv_data = csv_data.drop(csv_data.id == test_time_condition.id)

# There are 5 indices where this is not true. We remove these values since their
# trip duration spans a few days in some cases
# TODO Check that this is true.
csv_data = test_time_condition.filter(lambda x: x[-1] == x[-2])
csv_data = csv_data.toDF().drop("time_difference")
csv_data.printSchema()

# Drop the dropoff_datetime as it provides too much information with pickup already available
csv_data = csv_data.drop("dropoff_datetime")

# convert the pickup_datetime to individual attributes: month, day, hour, weekday

def extract_datetime_values(x):
    month = x[1].month
    day = x[1].day
    hour = x[1].hour
    weekday = x[1].weekday()
    return x[0], month, day, hour, weekday

date_time = csv_data.select(csv_data.id, csv_data.pickup_datetime).rdd
date_time = date_time.map(extract_datetime_values)
date_time = spark.createDataFrame(date_time).withColumnRenamed("_1","id").withColumnRenamed("_2", "month")\
    .withColumnRenamed("_3","day").withColumnRenamed("_4","hour").withColumnRenamed("_5","weekday")
csv_data = csv_data.join(date_time,"id","inner")
csv_data = csv_data.drop("pickup_datetime")

# LATITUDES AND LONGITUDES
# The data contains pickup and dropoff latitudes and longitudes
# We shall clean this and use other metrics as distance
# For now we shall use the geodesic distance
# Other distance metrics can be tried out and showed as comparative results

def get_distance(x):
    point_a = (x[1],x[2])
    point_b = (x[3],x[4])
    return x[0], geodesic(point_a,point_b).miles

distance = csv_data.select(csv_data.id, csv_data.pickup_latitude, csv_data.pickup_longitude, csv_data.dropoff_latitude, csv_data.dropoff_longitude).rdd
distance = distance.map(get_distance)
distance = spark.createDataFrame(distance).withColumnRenamed("_1","id").withColumnRenamed("_2", "distance")
csv_data = csv_data.drop("pickup_latitude").drop("pickup_longitude").drop("dropoff_latitude").drop("dropoff_longitude")
csv_data = csv_data.join(distance,"id","inner")

# Cast the columns as required
csv_data = csv_data.withColumn("vendor_id",csv_data["vendor_id"].cast(IntegerType()))
csv_data = csv_data.withColumn("month",csv_data["month"].cast(IntegerType()))
csv_data = csv_data.withColumn("day",csv_data["day"].cast(IntegerType()))
csv_data = csv_data.withColumn("hour",csv_data["hour"].cast(IntegerType()))
csv_data = csv_data.withColumn("weekday",csv_data["weekday"].cast(IntegerType()))
csv_data = csv_data.withColumn("distance",csv_data["distance"].cast(DoubleType()))
csv_data = csv_data.withColumn("passenger_count",csv_data["passenger_count"].cast(IntegerType()))

# Save to CSV file
csv_data.toPandas().to_csv("cleaned.csv",index=False)
"""
The data has been cleaned
""