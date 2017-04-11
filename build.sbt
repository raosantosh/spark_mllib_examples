crossBuildingSettings

name := "spark_ml_nlp"

version := "1.0"

scalaVersion := "2.11.2"

sparkVersion := "2.0.0"

classpathTypes += "maven-plugin"

retrieveManaged := true

resolvers += "Akka Repository" at "http://repo.akka.io/releases/"

assemblyMergeStrategy in assembly := {
  case "application.conf"            => MergeStrategy.concat
  case "reference.conf"              => MergeStrategy.concat
  case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
  case _ => MergeStrategy.first
    //val baseStrategy = (assemblyMergeStrategy in assembly).value
    //baseStrategy(x)
}

sparkComponents ++= Seq("streaming", "sql", "mllib")

libraryDependencies ++= Seq(
  "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0",
  "com.google.protobuf" % "protobuf-java" % "2.6.1",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0" classifier "models",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",
  "de.julielab" % "aliasi-lingpipe" % "4.1.0"
)

libraryDependencies ++= {
    val akkaVersion = "2.3.9"
    val sprayVersion = "1.3.2"
    Seq(
        "com.typesafe.akka" %% "akka-actor" % akkaVersion,
        "io.spray" %% "spray-can" % sprayVersion,
        "io.spray" %% "spray-routing" % sprayVersion,
        "io.spray" %% "spray-json" % "1.3.1",
        "com.typesafe.akka" %% "akka-slf4j" % akkaVersion,
        "ch.qos.logback" % "logback-classic" % "1.1.2",
        "com.typesafe.akka" %% "akka-testkit" % akkaVersion % "test",
        "io.spray" %% "spray-testkit" % sprayVersion % "test",
        "org.specs2" %% "specs2" % "2.3.13" % "test"
    )
}

val imageIoVersion = "3.1.1"
val dl4jVersion = "0.4-rc3.4"
val nd4jVersion = "0.4-rc3.5"

// DL4J does not work with scala 2.11
//libraryDependencies ++= Seq(
//  "org.deeplearning4j" % "dl4j-spark-ml" % dl4jVersion,
//  "org.nd4j" % "nd4j-x86" % nd4jVersion exclude("com.github.fommil.netlib", "all"),
//  "org.nd4j" % "nd4j-api" % nd4jVersion exclude("com.github.fommil.netlib", "all"),
//  "org.nd4j" % "nd4j-bytebuddy" % nd4jVersion exclude("com.github.fommil.netlib", "all")
//)

spIncludeMaven := false

retrieveManaged := true
    
EclipseKeys.relativizeLibs := true
    
// Avoid generating eclipse source entries for the java directories
(unmanagedSourceDirectories in Compile) <<= (scalaSource in Compile)(Seq(_))
   
(unmanagedSourceDirectories in Test) <<= (scalaSource in Test)(Seq(_))
