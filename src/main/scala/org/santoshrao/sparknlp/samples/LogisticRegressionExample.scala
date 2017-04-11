package com.yahoo.sparknlp.samples

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType

object LogisticRegressionExample {
  def main(args: Array[String]) {

    // Create Session
    val spark = SparkSession.builder().appName("Spark SQL Example").config("spark.some.config.option", "some-value").getOrCreate()

    import spark.implicits._

    // Define Schema
    val schemaString = "utterance strLabel"
    val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = true))
    val schema = StructType(fields)

    // Read Data and create Data Frame
    val nbaRDD = spark.sparkContext.textFile("/Users/santrao/work/spike/spark/spark_nlp/data/nba_binary.txt")
    val nbaRowRDD = nbaRDD.map(_.split("\t")).map(attributes => Row(attributes(0), attributes(1).trim))
    val nbaDF = spark.createDataFrame(nbaRowRDD, schema)

    //Convert label string to numeric
    val labelIndexer = new StringIndexer().setInputCol("strLabel").setOutputCol("label").fit(nbaDF)

    // Tokenize the words
    val tokenizer = new Tokenizer().setInputCol("utterance").setOutputCol("words")

    // Create NGram Transformer
    val ngram = new NGram().setInputCol("words").setOutputCol("ngrams")

    // Convert words to features
    val hashingTF = new HashingTF().setInputCol(ngram.getOutputCol).setOutputCol("features").setNumFeatures(5000)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Create classifier and predict
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, ngram, hashingTF, lr, labelConverter))
    val model = pipeline.fit(nbaDF)


    val inputRDD = spark.sparkContext.parallelize(Seq("Curry last game stats"), 1).map { value => Row(value, "comparePlayers") }
    val inputDF = spark.createDataFrame(inputRDD, schema)
    
    val predictions1 = model.transform(inputDF)
    
    predictions1.show()
    
    val predictions = predictions1.select("prediction", "predictedLabel", "label", "features", "rawPrediction", "probability")

    println("The predictions are ... ")

    val a = predictions.take(1)(0)(1).asInstanceOf[String]

    predictions.take(10).foreach(println)

    println(s"MY prediction is $a")

    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error : ${1 - accuracy}")
  }
}
