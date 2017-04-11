package com.yahoo.sparknlp.globals

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StructType

import com.typesafe.config.Config
import com.yahoo.sparknlp.main.ClassifierMain

import akka.actor.ActorSystem

object Globals {

  var config: Config = ClassifierMain.config
  val modelPath = config.getString("model.modelPath")
  val dataPath = config.getString("model.dataPath")
  val spark = SparkSession.builder().appName("Spark Session").config("spark.some.config.option", "some-value").getOrCreate()
  val sqlContext = new org.apache.spark.sql.SQLContext(spark.sparkContext)
  implicit val system: ActorSystem = ActorSystem("agent_as", config)

  val schema: StructType = {
    val schemaString = "utterance strLabel"
    val fields = schemaString.split(" ").map(fieldName => StructField(fieldName, StringType, nullable = true))
    StructType(fields)
  }
}
