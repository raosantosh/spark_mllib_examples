package com.yahoo.sparknlp.model

import scala.collection.JavaConverters.asScalaSetConverter

import org.apache.spark.rdd.RDD.rddToPairRDDFunctions

import com.aliasi.dict.DictionaryEntry
import com.aliasi.dict.ExactDictionaryChunker
import com.aliasi.dict.MapDictionary
import com.aliasi.tokenizer.IndoEuropeanTokenizerFactory
import com.yahoo.sparknlp.globals.Globals.dataPath
import com.yahoo.sparknlp.globals.Globals.spark

object SlotHandler {

  def replaceSlots(domain: String, inputText: String): (String, Integer) = {
    var result = ""
    val domainDictionaries = slotDictionary(domain)

    var processingText = inputText.toLowerCase
    var count = 0

    domainDictionaries.foreach { dictionaryItem =>
      val chunkResult = dictionaryItem._2.chunk(processingText).chunkSet().asScala
      result = ""
      var prevEnd = 0

      chunkResult.foreach { chunk =>
        val start = chunk.start
        val end = chunk.end
        result += (processingText.slice(prevEnd, start) + dictionaryItem._1)
        prevEnd = end
        count += 1
      }

      result += processingText.slice(prevEnd, processingText.size)
      processingText = result
    }

    (result, count)
  }

  val slotDictionary: Map[String, Map[String, ExactDictionaryChunker]] = {
    import com.yahoo.sparknlp.globals.Globals.dataPath

    def createChunkDict(filePath: String): ExactDictionaryChunker = {
      val playerDictionary = new MapDictionary[String]
      loadSlotData(filePath).foreach(mapEntry => playerDictionary.addEntry(new DictionaryEntry(mapEntry._1, mapEntry._2)))
      new ExactDictionaryChunker(playerDictionary, IndoEuropeanTokenizerFactory.INSTANCE, true, true)
    }

    Map("nba" -> Map("nba_player" -> createChunkDict(s"${dataPath}/dict/nba_player.tsv"), "nba_team" -> createChunkDict(s"${dataPath}/dict/nba_team.tsv")), "weather" -> Map())
  }

  def loadSlotData(filePath: String): Map[String, String] = {
    import com.yahoo.sparknlp.globals.Globals._
    val dictionaryRDD = spark.sparkContext.textFile(filePath)
    val splitRDD = dictionaryRDD.map(_.split(",")).filter { _.size > 1 }

    val resultMap = splitRDD.flatMap { dictEntry =>
      dictEntry.slice(1, dictEntry.size - 1).map {
        dictKey => (dictKey.toLowerCase, dictEntry(0).toLowerCase)
      }
    }.collectAsMap.toMap

    resultMap
  }
}