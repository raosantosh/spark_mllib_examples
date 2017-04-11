package com.yahoo.sparknlp.intent

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.OneVsRest
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType

import com.yahoo.sparknlp.globals.Globals

object Classifier {

  case class Intent(intentName: String, score: Double = 1.0)

  val schema: StructType = {
    val schemaString = "utterance strLabel"
    val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = true))
    StructType(fields)
  }

  def classify(texts: Seq[String]): List[Intent] = {
    import com.yahoo.sparknlp.globals._
    val myModel = PipelineModel.load("/tmp/spark-nba-model")

    val inputRowRDD = Globals.spark.sparkContext.parallelize(texts, 1).map { value => Row(value, "comparePlayers") }
    val inputDF = Globals.spark.createDataFrame(inputRowRDD, schema)

    val predictions = myModel.transform(inputDF).select("prediction", "predictedLabel", "label", "features")

    println(s"The predictions are ${predictions}")

    predictions.collect().map { result => Intent(result(1).asInstanceOf[String]) }.toList
  }

  def buildModel = {
    model.write.overwrite().save("/tmp/spark-nba-model")
  }

  def model: PipelineModel = {
    import com.yahoo.sparknlp.globals._

    // Read Data and create Data Frame
    val nbaRDD = Globals.spark.sparkContext.textFile("/Users/santrao/work/spike/spark/sample/data/nba.txt")
    val nbaRowRDD = nbaRDD.map(_.split("\t")).map(attributes => Row(attributes(0), attributes(1).trim))
    val nbaDF = Globals.spark.createDataFrame(nbaRowRDD, schema)

    //Convert label string to numeric
    val labelIndexer = new StringIndexer().setInputCol("strLabel").setOutputCol("label").fit(nbaDF)

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

    val ovr = new OneVsRest().setClassifier(lr)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, tokenizer, ngram, hashingTF, ovr, labelConverter))
    pipeline.fit(nbaDF)
  }

  def multiClassify(texts: Seq[String]): List[Intent] = {
    multiModel.flatMap(pipeline => {
      val inputRowRDD = Globals.spark.sparkContext.parallelize(texts, 1).map { value => Row(value, pipeline._1) }
      val inputDF = Globals.spark.createDataFrame(inputRowRDD, schema)
      pipeline._2.transform(inputDF).select("prediction", "predictedLabel", "label", "features", "probability").collect().map { result => Intent(result(1).asInstanceOf[String], result(4).asInstanceOf[Double]) }
    })
  }

  lazy val multiModel: List[(String, PipelineModel)] = {
    import com.yahoo.sparknlp.globals._

    // Read Data and create Data Frame
    val nbaRDD = Globals.spark.sparkContext.textFile("/Users/santrao/work/spike/spark/sample/data/nba.txt")

    //    val d = nbaRDD.map(_.split("\t")).groupBy { splitResult => splitResult(1) }.ke
    val splitRDD = nbaRDD.map(_.split("\t"))
    splitRDD.cache

    val allKeys = splitRDD.map { splitData => splitData(1) }.distinct.collect

    allKeys.map { key =>
      {
        val keyData = splitRDD.map(splitData =>
          if (splitData(0) == key)
            Row(splitData(0), splitData(1))
          else
            Row(splitData(0), "Other"))

        val nbaDF = Globals.spark.createDataFrame(keyData, schema)

        val labelIndexer = new StringIndexer().setInputCol("strLabel").setOutputCol("label").fit(nbaDF)

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
        (key, pipeline.fit(nbaDF))
      }
    }.toList
  }
}
