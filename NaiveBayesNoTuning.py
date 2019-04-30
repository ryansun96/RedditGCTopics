from pyspark.ml.feature import StringIndexer, CountVectorizer, IDF, StringIndexer
from pyspark.sql.functions import udf
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.types import FloatType

df = spark.read.json("wasbs://containerName@storageAcctName.blob.core.windows.net/path")

## Naive Bayes tuning pipelines can be easily built by following official Spark documentation
cv = CountVectorizer(inputCol="tokens", outputCol="bow_features", vocabSize=10_000_000, minDF=2, minTF=1) # We use a large number for vocabSize to include all vocabulary
model = cv.fit(df)
result = model.transform(df)
idf = IDF(inputCol="bow_features", outputCol="features")

rescaledData = idf.fit(result).transform(result) # It's worth discussing whether to do TFIDF on whole dataset or just training set
data_labeled = rescaledData.filter("link_flair_text is not null") # Train and test data
data_unlabeled = rescaledData.filter("link_flair_text is null") # Unlabeled data

# Encode flair into indices
labelIndexer = StringIndexer(inputCol ="link_flair_text", outputCol ="label")

# Currently, there is no way in PySpark to apply a non-aggregate function to every group (suppose we want to group by subreddit) - to do this, you need Java or Scala.
# China
data_labeled_china = data_labeled.filter("subreddit = 'China'")
data_unlabeled_china = data_unlabeled.filter("subreddit = 'China'")
data_labeled_china_encoded = labelIndexer.fit(data_labeled_china).transform(data_labeled_china)
(trainingData, testData) = data_labeled_china_encoded.randomSplit([0.8, 0.2], seed = 100)

# Follow similar steps to get sub-dataframes for Shanghai and Taiwan
# For simplicity they are omitted here.

nb = NaiveBayes(smoothing=1,labelCol='label')
model = nb.fit(trainingData)
# predictions = model.transform(testData)
predictions = model.transform(data_unlabeled_china)

# Retrieve label and its encoded index
label_index = data_labeled_china_encoded.select("link_flair_text","label").distinct().toDF('category', 'encoded_index')
pred_label = predictions.join(label_index, predictions.prediction==label_index.encoded_index, "inner").withColumn('link_flair_text', 'category')

# Define a user-defined function in PySpark to extract the largest probability (the probability associated with predicted label)
def extractProb(probArray, index):
    index = int(index) # Just a pre-caution, do casting here
    return float(probArray.values[index]) # Perform type cast, otherwise you might get pickle error

extractProbability = udf(extractProb, FloatType())

pred_label_prob = pred_label.withColumn('prob', extractProbability('probability', 'prediction'))

output = pred_label_prob.filter("prob > 0.6") # We discard low confidence predictions, without losing too much data

output.write.json("wasbs://containerName@storageAcctName.blob.core.windows.net/path") # Has to be an empty directory!
