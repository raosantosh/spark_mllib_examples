package com.yahoo.sparknlp.main

import spray.can.Http
import com.typesafe.config.ConfigFactory
import akka.actor._
import akka.io.IO
import akka.util.Timeout
import spray.can.Http
import scala.concurrent.duration._
import akka.actor.Props
import akka.pattern.ask

object ClassifierMain {

  val config = ConfigFactory.load()

  def main(args: Array[String]) {

    val host = config.getString("server.http-host")
    val port = config.getInt("server.http-port")
    implicit val system = ActorSystem("classifier-service")
    val api = system.actorOf(Props(new RestInterface()), "httpInterface")
    implicit val executionContext = system.dispatcher
    implicit val timeout = Timeout(10 seconds)

    api ! ("Start", port)
  }
}