package com.yahoo.sparknlp.nlp

import java.util.Properties

import scala.collection.JavaConversions.asScalaBuffer

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation
import edu.stanford.nlp.pipeline.Annotation
import edu.stanford.nlp.pipeline.StanfordCoreNLP

object NaturalLanguageProcessor {

  val pipeline = createNLPPipeline()

  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplits,pos,lemma")
    new StanfordCoreNLP(props)
  }

  def textToLemma(inputText: String): Seq[String] = {
    val document = new Annotation(inputText)
    pipeline.annotate(document)

    val sentences = document.get(classOf[SentencesAnnotation]).toList
    sentences.flatMap { sentence =>
      sentence.get(classOf[TokensAnnotation]).toList.map {
        token => token.get(classOf[LemmaAnnotation])
      }
    }
  }
}