package com.yahoo.sparknlp.main

import com.yahoo.sparknlp.globals.Globals.dataPath
import com.yahoo.sparknlp.globals.Globals.modelPath
import com.yahoo.sparknlp.model.ModelBuilder
import com.yahoo.sparknlp.model.ModelClassifier
import com.yahoo.sparknlp.model.MultiDomainModelBuilder
import com.yahoo.sparknlp.model.MultiModelBuilder

object TrainerMain {
  val nbaModelPath = s"${dataPath}/nba/nba.txt"
  val nbaModelOutuptDir = s"${modelPath}/nba"

  val weatherModelPath = s"${dataPath}/weather/weather.txt"
  val weatherModelOutuptDir = s"${modelPath}/weather"

  val nbaDomainModelPath = s"${dataPath}/nba/domain.txt"
  val nbaDomainModelOutuptDir = s"${modelPath}/nba"

  val weatherDomainModelPath = s"${dataPath}/weather/domain.txt"
  val weatherDomainModelOutuptDir = s"${modelPath}/weather"

  val longRunningTaskPath = s"${dataPath}/multidomain/utterances.txt"
  val longRunningTaskOutputDir = s"${modelPath}/longRunning"

  val multiModelDomainPath = s"${dataPath}/multidomain/multi_utterances_shuffled.txt"
  val multiModelDomainOutputDir = s"${modelPath}/multiDomain"

  val intentModelBuilder = MultiModelBuilder
  val domainModelBuilder = ModelBuilder
  val multiDomainModelBuilder = MultiDomainModelBuilder

  val classifier = ModelClassifier

  def main(args: Array[String]) {
    trainAll()
    checkModel()
  }

  private def trainAll() {
    println("[SPARK_NLP] Starting NBA Model Building ... ")
    intentModelBuilder.createModel("nba", nbaModelPath, nbaModelOutuptDir)
    println("[SPARK_NLP] Starting Weather Model Building ... ")
    intentModelBuilder.createModel("weather", weatherModelPath, weatherModelOutuptDir)
    println("[SPARK_NLP] Starting NBA Domain Model Building ... ")
    domainModelBuilder.createModel("nba", nbaDomainModelPath, nbaDomainModelOutuptDir, "nbaDomain")
    println("[SPARK_NLP] Starting Weather Domain Model Building ... ")
    domainModelBuilder.createModel("weather", weatherDomainModelPath, weatherDomainModelOutuptDir, "weatherDomain")
    println("[SPARK_NLP] Starting Long Running Model Bulding")
    multiDomainModelBuilder.createModel("multi", longRunningTaskPath, longRunningTaskOutputDir, "longRunning")
    println("[SPARK_NLP] Starting Multi Model Bulding")
    multiDomainModelBuilder.createModel("multi", multiModelDomainPath, multiModelDomainOutputDir, "multiDomain")
    println("[SPARK_NLP] Done Model Building ... ")

  }

  private def checkModel() {

    def classifyAndTest(domain: String, label: String, text: String) {
      val classifierResult = classifier.classify(domain, label, text)
      println(s"[SPARK_NLP] The result of classification is : ${classifierResult._1} with score : ${classifierResult._2}")
    }
    classifyAndTest("nba", "comparePlayers", "compare Kobe and Curry")
    classifyAndTest("weather", "Weather.isWeatherCondition", "weather in sfo")
    classifyAndTest("nba", "nbaDomain", "compare Kobe and Curry")
    classifyAndTest("weather", "weatherDomain", "How is weather in sfo")

    def classifyLabelAndTest(text: String, previousTask: String) {
      val classifierResult = classifier.classifyLabel(text, previousTask, "nba:0.5,weather:0.5", "inform")
      println(s"[SPARK_NLP] The result of classification for multiDomain: ${classifierResult} for input $text and previous $previousTask")
    }

    classifyLabelAndTest("curry", "comparePlayers")
    classifyLabelAndTest("curry", "na")
  }
}