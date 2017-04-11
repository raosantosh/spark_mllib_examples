package com.yahoo.sparknlp.model

import java.io.File

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row

import com.yahoo.sparknlp.globals.Globals.schema
import com.yahoo.sparknlp.globals.Globals.spark
import com.yahoo.sparknlp.model.ModelEvaluator.printMetrics

object MultiModelBuilder {

  def loadMultiModel(modelDirPath: String): List[(String, PipelineModel)] = {
    new File(modelDirPath).listFiles.filter { _.isFile }.map { _.getAbsolutePath }.map { file => (file.split("/").last, PipelineModel.load(file)) }.toList
  }

  def createModel(domain: String, filePath: String, outputDir: String) {
    createMultiModel(domain, filePath).foreach(modelPair => modelPair._2.write.overwrite().save(s"$outputDir/${modelPair._1}"))
  }

  def updatedInput(domain: String, text: String): (String, Integer) = {
    SlotHandler.replaceSlots(domain, text)
  }

  private def createMultiModel(domain: String, filePath: String): List[(String, PipelineModel)] = {
    val dataRDD = spark.sparkContext.textFile(filePath)

    val splitRDD = dataRDD.map(_.split("\t")).filter { _.size > 1 }
    splitRDD.cache

    val allKeys = splitRDD.map { splitData => splitData(1) }.distinct.collect

    allKeys.map { key =>
      {
        val keyData = splitRDD.filter(splitData => (splitData(1).trim.equals(key) || splitData(1).trim.equals("Other"))).map { splitData =>
          val transformedInput = updatedInput(domain, splitData(0).trim)
          Row(transformedInput._1, splitData(1).trim)
        }

        val nbaDF = spark.createDataFrame(keyData, schema)

        val Array(trianingData, testData) = nbaDF.randomSplit(Array(0.8, 0.2), seed = 11L)

        val labelIndexer = new StringIndexer().setInputCol("strLabel").setOutputCol("label").fit(trianingData)

        val labelConverter = new IndexToString()
          .setInputCol("prediction")
          .setOutputCol("predictedLabel")
          .setLabels(labelIndexer.labels)

        // Tokenize the words
        val tokenizer = new Tokenizer().setInputCol("utterance").setOutputCol("words")

        // Create NGram Transformer
        val ngram = new NGram().setInputCol("words").setOutputCol("ngrams")

        // Convert words to features
        val hashingTF = new HashingTF().setInputCol(ngram.getOutputCol).setOutputCol("features").setNumFeatures(10000)

        // Create classifier and predict
        val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3)
        val rf = new RandomForestClassifier()
          .setLabelCol("label")
          .setFeaturesCol("features")
          .setNumTrees(500).setMaxDepth(5)
        val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, ngram, hashingTF, lr, labelConverter))
        println(s"[SPARK_NLP] Generating model for $key for input $filePath")
        val res = (key, pipeline.fit(nbaDF))

        import com.yahoo.sparknlp.model.ModelEvaluator.printMetrics
        val testResults = res._2.transform(testData).select("prediction", "label").collect().map { row => (row(0).asInstanceOf[Double], row(1).asInstanceOf[Double]) }
        printMetrics(new MulticlassMetrics(spark.sparkContext.parallelize(testResults)))

        res
      }
    }.toList
  }
}