package com.yahoo.sparknlp.model

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.Row
import com.yahoo.sparknlp.globals.Globals
import com.yahoo.sparknlp.globals.Globals.schema
import com.yahoo.sparknlp.globals.Globals.spark
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object ModelBuilder {

  def loadModel(modelDirPath: String): PipelineModel = {
    PipelineModel.load(modelDirPath)
  }

  def createModel(domain:String, filePath: String, outputDir: String, modelName: String) {
    createModel(filePath).write.overwrite().save(s"$outputDir/$modelName")
  }

  private def createModel(filePath: String): PipelineModel = {
    import com.yahoo.sparknlp.globals._

    // Read Data and create Data Frame
    val nbaRDD = Globals.spark.sparkContext.textFile(filePath)
    val nbaRowRDD = nbaRDD.map(_.split("\t")).map(attributes => Row(attributes(0), attributes(1).trim))
    val nbaDF = Globals.spark.createDataFrame(nbaRowRDD, schema)

    val Array(trianingData, testData) = nbaDF.randomSplit(Array(0.8, 0.2), seed = 11L)

    //Convert label string to numeric
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
    val hashingTF = new HashingTF().setInputCol(ngram.getOutputCol).setOutputCol("features").setNumFeatures(5000)

    // Create classifier and predict
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, ngram, hashingTF, lr, labelConverter))
    val res = pipeline.fit(nbaDF)

    import com.yahoo.sparknlp.model.ModelEvaluator.printMetrics
    val testResults = res.transform(testData).select("prediction", "label").collect().map { row => (row(0).asInstanceOf[Double], row(1).asInstanceOf[Double]) }
    printMetrics(new MulticlassMetrics(spark.sparkContext.parallelize(testResults)))

    res
  }

}