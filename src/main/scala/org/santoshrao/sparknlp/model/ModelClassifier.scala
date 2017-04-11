package com.yahoo.sparknlp.model

import java.io.File

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.Row

import com.yahoo.sparknlp.globals.Globals.modelPath
import com.yahoo.sparknlp.globals.Globals.schema
import com.yahoo.sparknlp.globals.Globals.spark

object ModelClassifier {

  var cachedModels: Map[String, PipelineModel] = Map()

  val multiDomainModel = MultiDomainModelBuilder

  private def loadModel(basePath: String, domain: String, classifier: String): PipelineModel = {
    val key = domain + "_" + classifier
    cachedModels.get(key) match {
      case Some(model) => model
      case None =>
        val model = PipelineModel.load(s"$modelPath/$domain/$classifier")
        cachedModels += (key -> model)
        model
    }
  }

  def classifyAll(domain: String, text: String): List[(String, Double)] = {
    val result = (new File(s"$modelPath/$domain")).listFiles.filterNot(_.getName.endsWith("domain")).map(_.getName).map { classifierName =>
      classify(domain, classifierName, text)
    }
    result.toList
  }

  def classify(domain: String, classifier: String, text: String): (String, Double) = {

    import com.yahoo.sparknlp.globals.Globals._

    val model = loadModel(modelPath, domain, classifier)
    val transformedText = SlotHandler.replaceSlots(domain, text)

    val inputRDD = spark.sparkContext.parallelize(Seq(transformedText._1), 1).map { value => Row(value, classifier) }
    val inputDF = spark.createDataFrame(inputRDD, schema)

    val modelOutput = model.transform(inputDF)
    val predictions = modelOutput.select("prediction", "predictedLabel", "label", "features", "rawPrediction", "probability")

    val score = predictions.take(1)(0)(1) match {
      case "Other" => 1.0 - predictions.take(1)(0)(5).asInstanceOf[DenseVector](0)
      case `classifier` => predictions.take(1)(0)(5).asInstanceOf[DenseVector](0)
      case _ => 0.0
    }

    println(s"[SPARK_NLP] Classification result for $domain, $classifier and ($text , ${transformedText._1}) : $score ")
    (classifier, score)
  }

  def classifyLabel(input: String, previousTask: String, domainBias: String, dialogAct:String): String = {
    import com.yahoo.sparknlp.globals.Globals.{ spark, sqlContext }
    import sqlContext.implicits._

    val taskOnly = previousTask.split("\\.").toList.last
    val transformedText = SlotHandler.replaceSlots("nba", input)._1
    val transformedData = multiDomainModel.getModelFeatures(List(s"$transformedText\t$taskOnly\tcomparePlayers\t$domainBias"))
    val model = loadModel(modelPath, "longRunning", "longRunning")
    val modelOutput = model.transform(spark.createDataset(transformedData).toDF)
    val predictions = modelOutput.select("prediction", "predictedLabel", "label")

    val result = predictions.take(1)(0)(1).asInstanceOf[String]
    println(s"[SPARK_NLP] Classification result for ($input, ${transformedText}) and $previousTask : $result ")
    result
  }
}