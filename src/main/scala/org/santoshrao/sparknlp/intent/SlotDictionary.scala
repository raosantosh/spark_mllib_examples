package com.yahoo.sparknlp.intent

import org.apache.spark.SparkContext._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.feature.{ Tokenizer, HashingTF, NGram, StringIndexer }
import org.apache.spark.ml.classification.{ LogisticRegression, OneVsRest }
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline

object SlotDictionary {
  lazy val dictionaries = {
    import com.yahoo.sparknlp.globals._
    val dictionaryFile = Globals.spark.sparkContext.textFile("/Users/santrao/work/spike/spark/spark_nlp/data/nba_team.tsv")
    val teamDictionary = dictionaryFile.flatMap { teamRDD =>
      val tokens = teamRDD.split(",")
      tokens.slice(1, tokens.size).map { token => (token, tokens(0)) }
    }.collect.toMap

    val playerdictionaryFile = Globals.spark.sparkContext.textFile("/Users/santrao/work/spike/spark/spark_nlp/data/nba_player.tsv")
    val playerDictionary = playerdictionaryFile.flatMap { playerRDD =>
      val tokens = playerRDD.split(",")
      tokens.slice(1, tokens.size).map { token => (token, tokens(0)) }
    }.collect.toMap

    Map("team" -> teamDictionary, "player" -> playerDictionary)
  }

  def main(args: Array[String]) {
    println(s"The dictionary entries are ${SlotDictionary.dictionaries}")
  }
}