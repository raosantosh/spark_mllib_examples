package com.yahoo.sparknlp.main

import scala.language.postfixOps

import com.yahoo.sparknlp.globals.Globals.system
import com.yahoo.sparknlp.model.ModelClassifier

import akka.actor.Actor
import akka.actor.ActorLogging
import akka.actor.PoisonPill
import akka.actor.actorRef2Scala
import akka.io.IO
import spray.can.Http
import spray.http.HttpEntity.apply
import spray.http.HttpMethods.GET
import spray.http.HttpRequest
import spray.http.HttpResponse
import spray.http.Uri
import spray.json.DefaultJsonProtocol
import spray.json.pimpAny

class RestInterface extends Actor with ActorLogging {

  val classifier = ModelClassifier

  def receive = {
    case (command: String, port: Int) => IO(Http) ! Http.Bind(self, interface = "0.0.0.0", port = port)
    case _: Http.Connected => sender ! Http.Register(self)
    case hreq @ HttpRequest(GET, Uri.Path("/classify"), _, _, _) => handleClassify(hreq)
    case hreq @ HttpRequest(GET, Uri.Path("/taskScore"), _, _, _) => handleTaskScore(hreq)
    case someString: String =>
  }

  def handleClassify(req: HttpRequest) {
    val query = req.uri.query.get("text").getOrElse("Dummy")
    val resultType = req.uri.query.get("type").getOrElse("text")

    val result = req.uri.query.get("domain") match {
      case Some(domain) => convert(classifier.classifyAll(domain, query), resultType)
      case _ => {
        convert(classifier.classifyAll("nba", query) ++ classifier.classifyAll("weather", query), resultType)
      }
    }

    sender() ! HttpResponse(entity = result)
  }

  def handleTaskScore(req: HttpRequest) {
    val input = req.uri.query.get("text").getOrElse("Dummy")
    val previousTask = req.uri.query.get("prevTask").getOrElse("na")
    val domainBias = req.uri.query.get("domainBias").getOrElse("")
    val dialogAct = req.uri.query.get("dialogAct").getOrElse("inform")
    sender() ! HttpResponse(entity = convertTaskScore(classifier.classifyLabel(input, previousTask, domainBias, dialogAct), 1.0))
  }

  private def killYourself = self ! PoisonPill

  def convertTaskScore(taskName: String, result: Double): String = {
    case class ScorePair(label: String, score: Double)
    object MyJsonProtocol extends DefaultJsonProtocol {
      implicit val socrePairFormat = jsonFormat2(ScorePair)
    }

    import MyJsonProtocol._
    ScorePair(taskName, result).toJson.prettyPrint
  }

  def convert(results: List[(String, Double)], resultType: String): String = {

    if (resultType == "text") {
      results.toString
    } else {
      case class Scores(domainScores: List[ScorePair], intentScores: List[ScorePair])
      case class ScorePair(label: String, score: Double)
      object MyJsonProtocol extends DefaultJsonProtocol {
        implicit val socrePairFormat = jsonFormat2(ScorePair)
        implicit val scoreFormat = jsonFormat2(Scores)
      }

      import MyJsonProtocol._
      val scorePairs = results.map { result => ScorePair(result._1, result._2) }
      Scores(List(), scorePairs).toJson.prettyPrint
    }
  }
}