---
layout: post
title: My entry to the Kaggle SF crime classification competition using Apache Spark
caegories:
  - machine learning
  - tutorial
  - Spark
  - Kaggle
---

This post will detail how I built my entry to [the Kaggle San Francisco crime
classification competition](https://www.kaggle.com/c/sf-crime) using
[Apache Spark](http://spark.apache.org/) and the new ML library.

We'll be exploring the San Francisco crime dataset which contains crimes which
took place between 2003 and 2015 as detailed on [the
Kaggle competition page](https://www.kaggle.com/c/sf-crime).

You can find the code for this post on [Github](https://github.com/BenFradet/
spark-kaggle/tree/master/sfCrime).
<br><br>

### The dataset

The dataset given by Kaggle is split into two archives `train.csv.zip` and
`test.csv.zip`.
The training and test datasets contain a few features:

- `Dates`: timestamp of the crime incident in the PST timezone
- `Descript`: detailed description of the crime incident (unfortunately this is
only available in the training dataset so it'll be pretty much useless)
- `DayOfWeek`: the day of the week of the crime incident
- `PdDistrict`: name of the police department district which handled the crime
incident
- `Resolution`: how the crime incident was resolved (it's also only in the
training dataset)
- `Address`: the approximate street address of the incident
- `X`: longitude of the incident
- `Y`: latitude of the incident

The goal of the competition is to predict the `Category` variable which
corresponds to the type of offense committed, for example, here are the ten most
common offenses:

| Category | count |
|----+----|
| LARCENY/THEFT  | 174900 |
| OTHER OFFENSES | 126182 |
| NON-CRIMINAL   | 92304 |
| ASSAULT        | 76876 |
| DRUG/NARCOTIC  | 53971 |
| VEHICLE THEFT  | 53781 |
| VANDALISM      | 44725 |
| WARRANTS       | 42214 |
| BURGLARY       | 36755 |
| SUSPICIOUS OCC | 31414 |

<br>

### Loading the data

Since the data is in csv format, we'll use [spark-csv](https://github.com/
databricks/spark-csv) which will parse our csv and give us back `DataFrame`s:

{% highlight scala %}
val csvFormat = "com.databricks.spark.csv"
def loadData(
    trainFile: String,
    testFile: String,
    sqlContext: SQLContext
): (DataFrame, DataFrame) = {
  val schemaArray = Array(
    StructField("Id", LongType),
    StructField("Dates", TimestampType),
    StructField("Category", StringType), // target variable
    StructField("Descript", StringType),
    StructField("DayOfWeek", StringType),
    StructField("PdDistrict", StringType),
    StructField("Resolution", StringType),
    StructField("Address", StringType),
    StructField("X", DoubleType),
    StructField("Y", DoubleType)
  )

  val trainSchema = StructType(schemaArray.filterNot(_.name == "Id"))
  val testSchema = StructType(schemaArray.filterNot { p =>
    Seq("Category", "Descript", "Resolution") contains p.name
  })

  val trainDF = sqlContext.read
    .format(csvFormat)
    .option("header", "true")
    .schema(trainSchema)
    .load(trainFile)

  val testDF = sqlContext.read
    .format(csvFormat)
    .option("header", "true")
    .schema(testSchema)
    .load(testFile)

  (trainDF, testDF)
}
{% endhighlight %}

This function takes the paths to the uncompressed training and test files and
a `sqlContext` which will  have been initialized beforehand like so:

{% highlight scala %}
val sc = new SparkContext(new SparkConf().setAppName("SFCrime"))
val sqlContext = new SQLContext(sc)
{% endhighlight %}

I should point out that you can let Spark infer the schema. However, I tend to
always specify the type of every field to make sure every value of every
variable is of the expected type.

We can now load our data:

{% highlight scala %}
val (rawTrainDF, rawTestDF) = loadData(trainFile, testFile, sqlContext)
{% endhighlight %}

As a side note, parsing csv will be part of Apache Spark 2.0 so we won't have
to rely on another jar anymore.
<br><br>

### Feature engineering

Basically, we're left with six features:

- `Dates`
- `DayOfWeek`
- `Address` which is unusable as is (23 228 distinct addresses)
- `PdDistrict`
- `X` and `Y` which is basically one single feature: the coordinates of the
incident

Let's see if there are other features which could help our classification.
<br>

#### Time-related features

Along the lines of `DayOfWeek` I wanted to see if other time-related features
had any impact on the type of offense being committed.

Intuitively, given the top 10 categories, we would think that some types of
crime take place during night hours most of the time such as vehicle thefts or
vandalism.

Another trend I wanted to investigate was whether there were more crimes being
committed during certain years (and months). For example, here is the
distribution of `LARCENY/THEFT` crimes according to the year the incident
occurred:

| Year | count |
|----+----|
|2014|18901|
|2013|18152|
|2012|15639|
|2006|13798|
|2011|13084|

<br>
We clearly see an upward trend in the last few years.

This can be obtained with:

{% highlight scala %}
enrichedTrainDF
  .filter(enrichedTrainDF("Category") === "LARCENY/THEFT")
  .groupBy("Year")
  .count()
  .sort(desc("count"))
  .show(5)
{% endhighlight scala %}

Conversely, the `ASSAULT`s occurred the most during the first few years of the
time span covered by this dataset:

| Year | count |
|----+----|
|2003| 6555|
|2004| 6467|
|2006| 6364|
|2008| 6327|
|2013| 6280|

<br>

To sum things up, we need three new features: the hour of day, the month and
the year. Fortunately, Spark comes bundled with a lot of sql-like utility
functions which were presented in [a Databricks blogpost](https://
databricks.com/blog/2015/09/16/apache-spark-1-5-dataframe-api-highlights.html) a
few months back. This will greatly simplify our work:

{% highlight scala %}
df
  .withColumn("HourOfDay", hour(col("Dates")))
  .withColumn("Month", month(col("Dates")))
  .withColumn("Year", year(col("Dates")))
{% endhighlight scala %}

#### Weekend feature

Next up, I wrote a UDF to check if the incident took place during the weekend:

{% highlight scala %}
val enrichWeekend = (df: DataFrame) => {
  def weekendUDF = udf { (dayOfWeek: String) =>
    dayOfWeek match {
      case _ @ ("Friday" | "Saturday" | "Sunday") => 1
      case _ => 0
    }
  }
  df.withColumn("Weekend", weekendUDF(col("DayOfWeek")))
}
{% endhighlight %}

#### Address features

Then, I wanted to make the `Address` variable usable. If you have a look at a
few addresses in the dataset, you'll notice that they come in two forms:

- {street 1} / {street 2} to denote an intersection
- {number} Block of {street}

Consequently, I introduced two features from this column:

- `AddressType` which indicates whether the incident took place at an
intersection or on a particular street
- `Street` where I attempted to parse the `Address` variable to a single street
name, this reduced the cardinality of the original feature by 10x

Unfortunately, the `Street` variable will only contain the first address
(alphabetically) if `Address` is an intersection. So, is is possible that two
addresses containing the same street represented by intersections won't result
in the same street.

For example, given two `Address`: `A STREET / B STREET` and
`B STREET / C STREET` the resulting `Street` will be `A STREET` and `B STREET`.

{% highlight scala %}
val enrichAddress = (df: DataFrame) => {
  def addressTypeUDF = udf { (address: String) =>
    if (address contains "/") "Intersection"
    else "Street"
  }

  val streetRegex = """\d{1,4} Block of (.+)""".r
  val intersectionRegex = """(.+) / (.+)""".r
  def addressUDF = udf { (address: String) =>
    streetRegex findFirstIn address match {
      case Some(streetRegex(s)) => s
      case None => intersectionRegex findFirstIn address match {
        case Some(intersectionRegex(s1, s2)) => if (s1 < s2) s1 else s2
        case None => address
      }
    }
  }
  df
    .withColumn("AddressType", addressTypeUDF(col("Address")))
    .withColumn("Street", addressUDF(col("Address")))
}
{% endhighlight %}

#### Day or night feature

Along the same lines of the `HourOfDay` feature, I reasoned that it would be
interesting to see if an incident occurred during the day or the night.

To solve this issue, I used [the sunrise-sunset.org API](http://
sunrise-sunset.org/api) with [this script](https://github.com/BenFradet/
spark-kaggle/blob/master/sfCrime/src/main/scala/io/github/benfradet/
BuildSetRiseDataset.scala).
It basically makes a request to the API for each day present in the dataset to
retrieve the time of sunrise and sunset, parses the json-formatted result thanks
to [circe](https://github.com/travisbrown/circe) (which is a great json library
for Scala by the way) and writes it to a file.
There is an important thing to note about this script: a request is made every
five seconds in order not to overload the API.

The resulting dataset can be found [in the repo](https://github.com/BenFradet/
spark-kaggle/blob/master/sfCrime/src/main/resources/sunsetrise.json).

Once we have all our sunset and sunrise times, we can load them into a
dataframe:

{% highlight scala %}
val sunsetDF = {
  val rdd = sc.wholeTextFiles(sunsetFile).map(_._2)
  sqlContext.read.json(rdd)
}
{% endhighlight %}

We still have to write a UDF to determine if the incident took place during the
night or the day. To do that we just compare our timestamp to the time of
sunrise and sunset:

{% highlight scala %}
def enrichDayOrNight(sunsetDF: DataFrame)(df: DataFrame): DataFrame = {
  def dayOrNigthUDF = udf { (timestampUTC: String, sunrise: String, sunset: String) =>
    val timestampFormatter = DateTimeFormatter.ofPattern("YYYY-MM-dd HH:mm:ss")
    val timeFormatter = DateTimeFormatter.ofPattern("h:mm:ss a")
    val time = LocalTime.parse(timestampUTC, timestampFormatter)
    val sunriseTime = LocalTime.parse(sunrise, timeFormatter)
    val sunsetTime = LocalTime.parse(sunset, timeFormatter)
    if (sunriseTime.compareTo(sunsetTime) > 0) {
      if (time.compareTo(sunsetTime) > 0 && time.compareTo(sunriseTime) < 0) {
        "Night"
      } else {
        "Day"
      }
    } else {
      if (time.compareTo(sunriseTime) > 0 && time.compareTo(sunsetTime) < 0) {
        "Day"
      } else {
        "Night"
      }
    }
  }

  df
    .join(sunsetDF, df("Date") === sunsetDF("date"))
    .withColumn("DayOrNight", dayOrNigthUDF(col("TimestampUTC"), col("sunrise"), col("sunset")))
}
{% endhighlight %}

We join with the `sunsetDF` dataframe in order to benefit from the sunrise and
sunset columns.
You'll notice that we don't directly use the provided `Dates` variable. This is
because sunrise/sunset times are given by the API in the UTC timezone. As a
result we converted `Dates` to the UTC timezone (`TimestampUTC`) and extracted
its date (`Date`).
<br>

#### Weather features

We could imagine that incidents for a few categories would occur mostly outdoor
like `VEHICLE THEFT` for which the weather could have an influence as opposed to
other types of incidents which would occur indoor and for which the weather
wouldn't have any impact.

To check this assumption, I assembled a dataset containing the most occurring
weather condition and average temperature of every day in the dataset using
[wunderground.com](https://www.wunderground.com/). The script can be found [here
](https://github.com/BenFradet/spark-kaggle/blob/master/sfCrime/src/main/scala/
io/github/benfradet/BuildWeatherDataset.scala). It works similarly to the
sunrise/sunset scraping we just saw except that wunderground gives us csv
instead of json.

The resulting dataset can be found [here](https://github.com/BenFradet/
spark-kaggle/blob/master/sfCrime/src/main/resources/weather.json).

As we've done before, we need to turn this dataset into a dataframe:

{% highlight scala %}
val weatherDF = {
  val rdd = sc.wholeTextFiles(weatherFile).map(_._2)
  sqlContext.read.json(rdd)
}
{% endhighlight %}

Next, we join our training and test dataframes with the new `weatherDF`:

{% highlight scala %}
def enrichWeather(weatherDF: DataFrame)(df: DataFrame): DataFrame =
  df.join(weatherDF, df("Date") === weatherDF("date"))
{% endhighlight %}

As a result, we get two new features: `weather` and `temperatureC`.
<br>

#### Neighborhood feature

Finally, I wanted to see what I could do with the latitude and longitude
variables. I came up with the idea of trying to find the neighborhood where the
incident occurrend thinking that particular types of crimes are more inclined
to happen in particular neighborhoods.

To find San Francisco neighborhoods as latitude/longitude polygons I used
[the Zillow API](http://www.zillow.com/howto/api/neighborhood-boundaries.htm)
which fitted the bill perfectly providing neighborhoods for California as a
[shapefile](https://en.wikipedia.org/wiki/Shapefile). I used a bit of R to turn
the shapefile into a json containing my polygons as [WKT](https://en.wikipedia.
org/wiki/Well-known_text) which can be found [here](https://github.com/BenFradet
/spark-kaggle/blob/master/sfCrime/src/main/resources/neighborhoods.json).

Now that we have our SF neighborhoods as polygons we still have to determine
in which one of these neighborhoods every incident occurred. For this task, I
used [the ESRI geometry api](https://github.com/Esri/geometry-api-java/) which
lets you do all kinds of spatial data processing and, for me, check if a point
(corresponding to an incident) is inside a polygon (corresponding to a
neighborhood).

First up, we'll need functions which turns WKTs into ESRI geometries
(i.e. `Point` and `Polygon`):

{% highlight scala %}
def createGeometryFromWKT[T <: Geometry](wkt: String): T = {
  val wktImportFlags = WktImportFlags.wktImportDefaults
  val geometryType = Geometry.Type.Unknown
  val g = OperatorImportFromWkt.local().execute(wktImportFlags, geometryType, wkt, null)
  g.asInstanceOf[T]
}
{% endhighlight %}

Next, we need a function which tells us if a geometry contains another:

{% highlight scala %}
def contains(container: Geometry, contained: Geometry): Boolean =
  OperatorContains.local().execute(container, contained, SpatialReference.create(3426), null)
{% endhighlight %}

As you may have noticed, the API is a bit cumbersome but [it's really
well-documented](https://github.com/Esri/geometry-api-java/wiki).

Now that we have our neighborhoods and our utility functions, we can write a UDF
which will tell us in which neighborhood an incident occurred:

{% highlight scala %}
def enrichNeighborhoods(nbhds: Seq[Neighborhood])(df: DataFrame): DataFrame = {
  def nbhdUDF = udf { (lat: Double, lng: Double) =>
    val point = createGeometryFromWKT[Point](s"POINT($lat $lng)")
    nbhds
      .filter(nbhd => contains(nbhd.polygon, point))
      .map(_.name)
      .headOption match {
        case Some(nbhd) => nbhd
        case None => "SF"
      }
  }
  df.withColumn("Neighborhood", nbhdUDF(col("X"), col("Y")))
}
{% endhighlight %}

#### Putting it all together

We can now add those features to both our datasets:

{% highlight scala %}
val enrichFunctions = List(enrichTime, enrichWeekend, enrichAddress,
  enrichDayOrNight(sunsetDF)(_), enrichWeather(weatherDF)(_), enrichNeighborhoods(nbhds)(_))
val Array(enrichedTrainDF, enrichedTestDF) =
  Array(rawTrainDF, rawTestDF) map (enrichFunctions reduce (_ andThen _))
{% endhighlight %}
<br>

### Machine learning pipeline

Now that we have all our features, we can build our machine learning pipeline.
<br>

#### Building the pipeline

In order to be processed by our classifier, our categorical features need to
be indexed, this is done through `StringIndexer`. However, we still have one
difficulty to sort through: unfortunately, some addresses are only present
in the test dataset. Hence, we'll have to index our categorical variables with
all the data:

{% highlight scala %}
val allData = enrichedTrainDF
  .select((numericFeatColNames ++ categoricalFeatColNames).map(col): _*)
  .unionAll(enrichedTestDF
    .select((numericFeatColNames ++ categoricalFeatColNames).map(col): _*))
allData.cache()

val stringIndexers = categoricalFeatColNames.map { colName =>
  new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(colName + "Indexed")
    .fit(allData)
}
{% endhighlight scala %}

Since our label variable is also categorical, we'll have to index it as well:

{% highlight scala %}
val labelIndexer = new StringIndexer()
  .setInputCol(labelColName)
  .setOutputCol(labelColName + "Indexed")
  .fit(enrichedTrainDF)
{% endhighlight %}

Then, we can assemble all our features into a single vector column as required
by Spark's ML algorithms:

{% highlight scala %}
val assembler = new VectorAssembler()
  .setInputCols((categoricalFeatColNames.map(_ + "Indexed") ++ numericFeatColNames).toArray)
  .setOutputCol(featuresColName)
{% endhighlight %}

Finally, we can define our classifier:

{% highlight scala %}
val randomForest = new RandomForestClassifier()
  .setLabelCol(labelColName + "Indexed")
  .setFeaturesCol(featuresColName)
  .setMaxDepth(10)
  .setMaxBins(2089)
{% endhighlight %}

I chose a random forest classifier because it is one of the only multiclass
classifier available in Spark (`OneVsRest` coupled with `LogisticRegression`
wasn't really an option on my computer).

A couple notes regarding the parameters:

- The number of bins needs to be at least as great as the maximum cardinality
of the different features (i.e. 2089 which is the number of different streets
in the `Street` column).
- Given the RAM on my computer I couldn't go higher than 10 for the depth of a
decision tree but usually a higher depth gives better results.

Because we indexed our label variable `Category` to `CategoryIndexed`, we'll
need to get back our original labels, this is done with `IndexToString`:

{% highlight scala %}
val indexToString = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol(predictedLabelColName)
  .setLabels(labelIndexer.labels)
{% endhighlight %}

We can finally construct our pipeline:

{% highlight scala %}
val pipeline = new Pipeline()
  .setStages(Array.concat(
    stringIndexers.toArray,
    Array(labelIndexer, assembler, randomForest, indexToString)
  ))
{% endhighlight %}

#### Cross validation

We can now find the best model through cross validation. In this example,
cross validation is a bit artificial because I was limited by my computer in
terms of processing power.

To run cross validation, you'll have to setup three different things:

- an evaluator:

{% highlight scala %}
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol(labelColName + "Indexed")
{% endhighlight %}

- a param grid containing the values you want to validate against:

{% highlight scala %}
val paramGrid = new ParamGridBuilder()
  .addGrid(randomForest.impurity, Array("entropy", "gini"))
  .build()
{% endhighlight %}

- a cross validator:

{% highlight scala %}
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(3)
{% endhighlight %}

We can finally train our model:

{% highlight scala %}
val cvModel = cv.fit(enrichedTrainDF)
{% endhighlight %}

and make predictions on the test set:

{% highlight scala %}
val predictions = cvModel
  .transform(enrichedTestDF)
  .select("Id", predictedLabelColName)
{% endhighlight %}

#### Best params

A shortcoming we have to face when using cross validation is that to find the
best params found you have to look at the `INFO`-level logs of your Spark
application which tend to be really noisy.
To circumvent this, you can disable `INFO` level logs and retrieve them
directly:

{% highlight scala %}
val bestEstimatorParamMap = cvModel.getEstimatorParamMaps
  .zip(cvModel.avgMetrics)
  .maxBy(_._2)
  ._1
println(bestEstimatorParamMap)
{% endhighlight %}

#### Feature importances

Another nifty trick I learned is how to retrieve the importance of every feature
in your model:

{% highlight scala %}
val featureImportances = cvModel
  .bestModel.asInstanceOf[PipelineModel]
  .stages(categoricalFeatColNames.size + 2)
  .asInstanceOf[RandomForestClassificationModel].featureImportances
assembler.getInputCols
  .zip(featureImportances.toArray)
  .foreach { case (feat, imp) => println(s"feature: $feat, importance: $imp") }
{% endhighlight %}

### Formatting the results

Last step, we have to format our results according to the Kaggle instructions
which are to have one column per `Category` each filled with 0 except for
one filled with a 1 for the predicted `Category`.

First, we need a list of all labels sorted alphabetically:

{% highlight scala %}
val labels = enrichedTrainDF.select(labelColName).distinct().collect()
  .map { case Row(label: String) => label }
  .sorted
{% endhighlight %}

Next, we need a function which will turn our predicted `Category` into a
sequence of zeros and a one at the right spot:

{% highlight scala %}
val labelToVec = (predictedLabel: String) => {
  val array = new Array[Int](labels.length)
  array(labels.indexOf(predictedLabel)) = 1
  array.toSeq
}
{% endhighlight %}

Then, we can create our result dataframe:

{% highlight scala %}
val schema = StructType(predictions.schema.fields ++ labels.map(StructField(_, IntegerType)))
val resultDF = sqlContext.createDataFrame(
  predictions.rdd.map { r => Row.fromSeq(
    r.toSeq ++
      labelToVec(r.getAs[String](predictedLabelColName))
  )},
  schema
)
{% endhighlight %}

Basically, we create our wanted schema and recreate a dataframe from the
existing `predictions` which fits the schema.

Finally, we can write it to a file:

{% highlight scala %}
resultDF
  .drop("predictedLabel")
  .coalesce(1)
  .write
  .format(csvFormat)
  .option("header", "true")
  .save(outputFile)
{% endhighlight %}
<br>

### spark-submit script

I wrote a little script which takes care of the file handling and submission to
Spark which you can find [here](https://github.com/BenFradet/spark-kaggle/blob/
master/sfCrime/bin/submit.sh). It just needs `spark-submit` to be on your path.
<br><br>

### Conclusion

I hope this could demonstrate the power of Apache Spark and in particular how
easy it is to manipulate features and create new ones.

However, there are a few things which could have been done differently/better:

- I think this dataset would have been particularly suitable for multiclass
logistic regression which is unfortunately not available in Spark right now
(it's currently being worked on and might end up in Spark 2.1, [see SPARK-7159](
https://issues.apache.org./jira/browse/SPARK-7159).
- The weather features could have been more accurate by taking the closest (in
time) weather condition and temperature instead of taking the most occurring
weather condition and average temperature for each day.
- We could also have worked on whether or not the police district handling the
incident was in the same neighborhood as the incident itself. My reasoning is
that some types of incident like fraud would be handled by particular police
districts.
- Another lead for better classification would be to get more accurate (and
including the areas around SF) neighborhood data as for two thirds of the data
the neighborhood was not found in the dataset.
- Also, I would have liked to explore decision trees with a depth greater than
10 (amongst other things) but unfortunately my computer couldn't handle it. I'll
maybe try this job on AWS using [flintrock](https://github.com/nchammas/
flintrock) which is a blazingly fast command-line tool for launching Spark
cluster on AWS.
