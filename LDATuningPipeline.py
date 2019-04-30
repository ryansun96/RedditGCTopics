## WORKS WITH PYSPARK 2.3.1, MAY NOT WORK WITH 2.4.0!
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.clustering import LDA
from pyspark.ml import Pipeline
from pyspark.ml.tuning import TrainValidationSplit, TrainValidationSplitModel, ParamGridBuilder

from multiprocessing.pool import ThreadPool
import itertools
import numpy as np

# Loads data
dataset = spark.read.format("json").load("wasbs://containerName@storageAcctName.blob.core.windows.net/path")

bowEstimator = CountVectorizer(inputCol='tokens', outputCol='features', vocabSize=10000000)
ldaEstimator = LDA(maxIter=100).setFeaturesCol('features')

tuningPipeline = Pipeline(stages=[bowEstimator, ldaEstimator])

paramGrid = ParamGridBuilder().addGrid(bowEstimator.minDF, [5, 10]).addGrid(bowEstimator.minTF, [2, 5]) \
    .addGrid(ldaEstimator.k, [5, 10]).addGrid(ldaEstimator.topicConcentration, [0.1, 0.5]) \
    .build()


# from pyspark.ml.evaluation import Evaluator

# class LDAEvaluator(Evaluator): ## For simplicity we did not formally use an evaluator.


def LDAparallelFitTasks(est, train, epm):

    modelIter = est.fitMultiple(train, epm)

    def singleTask():
        index, model = next(modelIter)
        metric = model.stages[1].logLikelihood(
            model.stages[0].transform(train))  ## This is inefficient, as we transformed / CountVectorized twice
        return index, metric

    return [singleTask] * len(epm)


class LDATunner(TrainValidationSplit):

    def __init__(self, estimator=None, estimatorParamMaps=None, parallelism=3):

        super(LDATunner, self).__init__(estimator=estimator, estimatorParamMaps=estimatorParamMaps, evaluator=None,
                                        trainRatio=0.75, parallelism=parallelism, seed=None)
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        # seed = self.getOrDefault(self.seed)
        train = dataset.cache()

        # subModels = None

        tasks = LDAparallelFitTasks(est, train, epm)
        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        metrics = [None] * numModels
        for j, metric in pool.imap_unordered(lambda f: f(), tasks):
            metrics[j] = metric

        train.unpersist()
        bestIndex = np.argmax(metrics)
        #         if eva.isLargerBetter():
        #             bestIndex = np.argmax(metrics)
        #         else:
        #             bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex]) ## We are assuming to use log likelihood measure, which is the larger the better.
        return self._copyValues(TrainValidationSplitModel(bestModel, metrics))


ldatuner = LDATunner(estimator=tuningPipeline, estimatorParamMaps=paramGrid)
ldamodel = ldatuner.fit(dataset)
