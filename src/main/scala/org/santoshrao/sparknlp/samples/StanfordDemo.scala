package com.yahoo.sparknlp.samples

import java.util
import java.util.Properties
import java.util.Properties
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.util.CoreMap
import edu.stanford.nlp.pipeline.Annotation
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation
import edu.stanford.nlp.util.CoreMap
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation
import edu.stanford.nlp.ling.CoreAnnotations

object StanfordDemo {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)

    val props: Properties = new Properties
    props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref")
    props.put("dcoref.maxdist", "10")
    props.put("coref.mode", "statistical")
    props.put("coref.md.type", "rule")
    props.put("coref.doClustering", "true")
    val pipeline: StanfordCoreNLP = new StanfordCoreNLP(props)

    val document: Annotation = new Annotation("Why is the sky so high in Japan ?")
    pipeline.annotate(document)

    val sentences: util.List[CoreMap] = document.get(classOf[CoreAnnotations.SentencesAnnotation])

    val tree = sentences.get(0).get(classOf[TreeAnnotation])
    println(s"parse tree:$tree")

    val dependencies = sentences.get(0).get(classOf[CollapsedCCProcessedDependenciesAnnotation])
    println(s"dependency graph $dependencies")
  }
}