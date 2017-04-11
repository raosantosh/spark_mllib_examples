package com.yahoo.sparknlp.model

import scala.collection.mutable.HashMap

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.linalg.Vectors

import com.yahoo.sparknlp.globals.Globals.spark
import com.yahoo.sparknlp.globals.Globals.sqlContext.implicits.newProductEncoder

object MultiDomainModelBuilder {

  case class ModelFeatures(classifierFeatures: Vector, previousTask: String, strLabel: String)

  val modelClassifier = ModelClassifier

  def loadModel(modelDirPath: String): PipelineModel = {
    PipelineModel.load(modelDirPath)
  }

  def createModel(domain: String, filePath: String, outputDir: String, modelName: String) {
    createModel(filePath).write.overwrite().save(s"$outputDir/$modelName")
  }

  private def createModel(filePath: String): PipelineModel = {

    import com.yahoo.sparknlp.globals.Globals.{ spark, sqlContext }

    val multimodelRDD = spark.sparkContext.textFile(filePath)
    import sqlContext.implicits._

    val transformedData = getModelFeaturesBuilding(multimodelRDD.collect.toList)

    val dataFrame = spark.createDataset(transformedData).toDF

    val previousTaskIndexer = new StringIndexer().setInputCol("previousTask").setOutputCol("indexedPreviousTask").fit(dataFrame)
    val encoder = new OneHotEncoder().setInputCol(previousTaskIndexer.getOutputCol).setOutputCol("taskEncoded")
    val assembler = new VectorAssembler().setInputCols(Array("classifierFeatures", encoder.getOutputCol)).setOutputCol("features")
    val labelIndexer = new StringIndexer().setInputCol("strLabel").setOutputCol("label").fit(dataFrame)
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3)
    val rfc = new RandomForestClassifier().setLabelCol(labelIndexer.getOutputCol).setFeaturesCol(assembler.getOutputCol).setNumTrees(10)
    val ovr = new OneVsRest().setClassifier(rfc)
    val labelConverter = new IndexToString().setInputCol(ovr.getPredictionCol).setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(previousTaskIndexer, labelIndexer, encoder, assembler, ovr, labelConverter))
    val model = pipeline.fit(dataFrame)

    model
  }

  def getModelFeaturesBuilding(input: List[String]): List[ModelFeatures] = {
    var nbaCount = 0
    var weatherCount = 0

    input.map { inputRDD =>
      inputRDD.split("\t")
    }.filter { inputRdd => inputRdd.size >= 3 }
      .map { inputSplitRdd =>
        var featureMap = new HashMap[Integer, Double]()
        val inputText = inputSplitRdd(0)
        val classifierOutput = modelClassifier.classifyAll("nba", inputText) ++ modelClassifier.classifyAll("weather", inputText).sortBy(result => result._1)
        // Currently only nba and weather
        val totalCount = nbaCount + weatherCount + 1
        val domainBias = inputSplitRdd.size > 3 match {
          case true if inputSplitRdd(3) == "nba" =>
            nbaCount += 1
            Array((nbaCount * 1.0) / totalCount, (weatherCount * 1.0) / totalCount)
          case true if inputSplitRdd(3) == "weather" =>
            weatherCount += 1
            Array((nbaCount * 1.0) / totalCount, (weatherCount * 1.0) / totalCount)
          case false => Array(0.5, 0.5)
        }

        var featureArray = classifierOutput.zipWithIndex.map {
          myResult => myResult._1._2
        }.toArray ++ domainBias

        val previousTask = inputSplitRdd(1)
        val label = inputSplitRdd(2)
        ModelFeatures(Vectors.dense(featureArray), previousTask, label)
      }.toList
  }

  def getModelFeatures(input: List[String]): List[ModelFeatures] = {
    input.map { inputRDD =>
      inputRDD.split("\t")
    }.filter { inputRdd => inputRdd.size >= 3 }
      .map { inputSplitRdd =>
        var featureMap = new HashMap[Integer, Double]()
        val inputText = inputSplitRdd(0)
        val classifierOutput = modelClassifier.classifyAll("nba", inputText) ++ modelClassifier.classifyAll("weather", inputText).sortBy(result => result._1)
        // Currently only nba and weather
        val domainBias = inputSplitRdd.size > 3 match {
          case true =>
            inputSplitRdd(3).split(",").sortBy(_.toLowerCase).map(_.split(":")(1).toDouble)
          case false => Array(0.5, 0.5)
        }

        var featureArray = classifierOutput.zipWithIndex.map {
          myResult => myResult._1._2
        }.toArray ++ domainBias

        val previousTask = inputSplitRdd(1)
        val label = inputSplitRdd(2)
        ModelFeatures(Vectors.dense(featureArray), previousTask, label)
      }.toList
  }
}