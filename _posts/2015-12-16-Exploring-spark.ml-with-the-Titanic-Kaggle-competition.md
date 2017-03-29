---
layout: post
title: Exploring spark.ml with the Titanic Kaggle competition
categories:
    - machine learning
    - tutorial
    - Spark
    - Kaggle
---

It's been a while since my last post, and in this post I'm going to talk about
a technology I've been using for almost a year now:
[Apache Spark](http://spark.apache.org/).

Basically, Spark lets you do data processing in a distributed manner. The
project is subdivided in modules:

- Spark streaming which lets you interact with streams of data
- Spark SQL which aims to let you write SQL queries and execute them on your
    data
- MLlib, a machine learning library to train classification/regression models
    on your data (among other things)
- GraphX which provides an API to interact with graphs and graphs computation

Today, I'll talk about MLlib which is, as previously mentioned, the Spark
submodule dedicated to machine learning. This submodule is split in two:
[spark.mllib](http://spark.apache.org/docs/latest/mllib-guide.html#mllib-types-algorithms-and-utilities)
which is built on top of the old RDDs and
[spark.ml](http://spark.apache.org/docs/latest/mllib-guide.html#sparkml-high-level-apis-for-ml-pipelines)
which is built on top of the DataFrame API. In this post, I'll talk
exclusively about spark.ml which aims to ease the process of creating machine
learning pipelines.

If you want to follow this tutorial you will have to download spark which can
be done [here](http://spark.apache.org/downloads.html). Additionnally, you will
need a few dependencies in order to build your project:

| groupId | artifactId | version | scope |
|----+----|
| org.apache.spark | spark-core_2.10 | 1.5.2 | provided |
| org.apache.spark | spark-sql_2.10 | 1.5.2 | compile |
| org.apache.spark | spark-mllib_2.10 | 1.5.2 | compile |
| com.databricks | spark-csv_2.10 | 1.2.0 | compile |

<br>

We'll be using the Titanic dataset taken from a
[Kaggle competition](https://www.kaggle.com/c/titanic). The goal is to predict
if a passenger survived from a set of features such as the class the passenger
was in, hers/his age or the fare the passenger paid to get on board.

You can find the code for this post on [Github](https://github.com/BenFradet/spark-kaggle/tree/master/titanic).
<br><br>

### Data exploration and data transformation
<br>

#### Getting to know the Titanic dataset

You can find a description of the features on [Kaggle](https://www.kaggle.com/c/titanic/data).

The dataset is split in two: `train.csv` and `test.csv`. As you've probably
already guessed, `train.csv` will contain labeled data (the `Survived` column
will be filled) and `test.csv` will be unlabeled data. The goal is to predict
for each example/passenger in `test.csv` whether or not she/he survived.
<br><br>

#### Loading the Titanic dataset

Since the data is in csv format, we'll use [spark-csv](https://github.com/databricks/spark-csv)
which will parse our csv data and give us back `DataFrames`.

To load the `train.csv` and `test.csv` file, I wrote the following function:

{% highlight scala %}
def loadData(
  trainFile: String,
  testFile: String,
  sqlContext: SQLContext
): (DataFrame, DataFrame) = {
  val nullable = true
  val schemaArray = Array(
    StructField("PassengerId", IntegerType, nullable),
    StructField("Survived", IntegerType, nullable),
    StructField("Pclass", IntegerType, nullable),
    StructField("Name", StringType, nullable),
    StructField("Sex", StringType, nullable),
    StructField("Age", FloatType, nullable),
    StructField("SibSp", IntegerType, nullable),
    StructField("Parch", IntegerType, nullable),
    StructField("Ticket", StringType, nullable),
    StructField("Fare", FloatType, nullable),
    StructField("Cabin", StringType, nullable),
    StructField("Embarked", StringType, nullable)
  )

  val trainSchema = StructType(schemaArray)
  val testSchema = StructType(schemaArray.filter(p => p.name != "Survived"))

  val csvFormat = "com.databricks.spark.csv"

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

This function takes the paths to the `train.csv` and `test.csv` files as the two
first arguments and a `sqlContext` which will have been initialized beforehand
like so:

{% highlight scala %}
val sc = new SparkContext(new SparkConf().setAppName("Titanic"))
val sqlContext = new SQLContext(sc)
{% endhighlight %}

Although not mandatory, we define the schema for the data as it is the same
for both files except for the `Survived` column.

Then, we use [spark-csv](https://github.com/databricks/spark-csv) to load our
data.
<br><br>

#### Feature engineering

Next, we'll do a bit of feature engineering on this dataset.

If you have a closer look at the `Name` column, you probably see that there is
some kind of title included in the name such as "Sir", "Mr", "Mrs", etc.
I think it is a valuable piece of information and I think it can influence
whether someone survived or not, that's why I extracted it in its own column.

My first intuition was to extract this title with a regex with the help of a
UDF (for user-defined function):

{% highlight scala %}
val Pattern = ".*, (.*?)\\..*".r
val title: (String => String) = {
  case Pattern(t) => t
  case _ => ""
}
val titleUDF = udf(title)

val dfWithTitle = df.withColumn("Title", titleUDF(col("Name")))
{% endhighlight %}

Unfortunately, every passenger's name doesn't comply with this regex and this
resulted in some noise in the `Title` column. As a result, I just looked for the
distinct titles produced by my UDF
{% highlight scala %}dfWithTitle.select("Title").distinct(){% endhighlight %}
and adapted it a bit:

{% highlight scala %}
val Pattern = ".*, (.*?)\\..*".r
val titles = Map(
  "Mrs"    -> "Mrs",
  "Lady"   -> "Mrs",
  "Mme"    -> "Mrs",
  "Ms"     -> "Ms",
  "Miss"   -> "Miss",
  "Mlle"   -> "Miss",
  "Master" -> "Master",
  "Rev"    -> "Rev",
  "Don"    -> "Mr",
  "Sir"    -> "Sir",
  "Dr"     -> "Dr",
  "Col"    -> "Col",
  "Capt"   -> "Col",
  "Major"  -> "Col"
)
val title: ((String, String) => String) = {
  case (Pattern(t), sex) => titles.get(t) match {
    case Some(tt) => tt
    case None     =>
        if (sex == "male") "Mr"
        else "Mrs"
  }
  case _ => "Mr"
}
val titleUDF = udf(title)

val dfWithTitle = df.withColumn("Title", titleUDF(col("Name"), col("Sex")))
{% endhighlight %}

This UDF tries to match on the previously defined pattern `Pattern`. If the
regex matches we'll try to find the title in our `titles` map. Finally, if we
don't find it, we'll define the title based on the `Sex` column: "Mr" if "male",
"Mrs" otherwise.

I, then, wanted to represent the family size of each passenger with the help of
the `Parch` column (which represents the number of parents/children aboard the
Titanic) and the `SibSp` column (which represents the number of siblings/spouses
aboard):

{% highlight scala %}
val familySize: ((Int, Int) => Int) = (sibSp: Int, parCh: Int) => sibSp + parCh + 1
val familySizeUDF = udf(familySize)
val dfWithFamilySize = df
  .withColumn("FamilySize", familySizeUDF(col("SibSp"), col("Parch")))
{% endhighlight %}

The family size UDF just does the sum of the `SibSp` and the `Parch` columns
plus one.
<br><br>

#### Handling NA values

You have two options when dealing with NA:

  - either drop them through:
    {% highlight scala %}df.na.drop(){% endhighlight %}
  - or fill them with default values with:
    {% highlight scala %}df.na.fill(){% endhighlight %}

After noticing that NA values were present in the `Age`, `Fare` and `Embarked`
columns, I chose to replace them:

  - with the average age for the `Age` column
  - with the average fare for the `Fare` column
  - with "S" for the `Embarked` column which represents the city of Southampton

In order to do this, I calculated the average of the `Age` column like so:

{% highlight scala %}
val avgAge = trainDF.select("Age").unionAll(testDF.select("Age"))
  .agg(avg("Age"))
  .collect() match {
  case Array(Row(avg: Double)) => avg
  case _ => 0
}
{% endhighlight %}

Same thing for the `Fare` column. I, then, filled my dataset with those
averages:

{% highlight scala %}
val dfFilled = df.na.fill(Map("Fare" -> avgFare, "Age" -> avgAge)
{% endhighlight %}

Another option, which I won't cover here, is to train a regression model on the
`Age` column and use this model to predict the age for the examples where the
`Age` is NA. Same thing goes for handling NA for the `Fare` column.

However, spark-csv treats NA strings as empty strings instead of NAs (this is a
known bug described [here](https://github.com/databricks/spark-csv/issues/86)).
This is why I coded a UDF which transforms empty strings in the `Embarked`
column to "S" for Southampton:

{% highlight scala %}
val embarked: (String => String) = {
  case "" => "S"
  case a  => a
}
val embarkedUDF = udf(embarked)
val dfFilled = df.withColumn("Embarked", embarkedUDF(col("Embarked")))
{% endhighlight %}
<br><br>

### Building the ML pipeline

What's very interesting about spark.ml compared to spark.mllib, aside from
dealing with DataFrames instead of RDDs, is the fact that you can build and tune
your own machine learning pipeline as we'll see in a bit.

There are two main concepts in spark.ml (extracted from the
[guide](http://spark.apache.org/docs/latest/ml-guide.html#main-concepts)):

  - `Transformers`, which are algorithms which transfrom a DataFrame into
another. For example, a machine learning model is a `Transformer` which
transforms DataFrames with features into DataFrames with predictions.
  - `Estimators`, which are algorithms which can be fit on a DataFrame to
produce a `Transformer`. For example, a learning algorithm is an `Estimator`
which trains on a DataFrame to produce a machine learning model (which is a
`Transformer`).

A pipeline is an ordered combination of `Transformers` and `Estimators`.
<br><br>

#### Description of our pipeline

In this post, we'll be training a random forest and since spark.ml can't
handle categorical features or labels unless they are indexed, our first job
will be to do just that.

Then, we'll assemble all our feature columns into one vector column because
every spark.ml machine learning algorithm expects that.

Once this is done, we can train our random forest as our data is in the expected
format.

Finally, we'll have to *unindex* our labels so they can be interpretable by us
and the Kaggle tester.
<br><br>

#### Indexing categorical features and labels

Fortunately, there are already built-in transformers to index categorical
features, we just have to choose between two options:

  1. Assemble all the features into one vector (through
[VectorAssembler](http://spark.apache.org/docs/latest/ml-features.html#vectorassembler))
and then use a
[VectorIndexer](http://spark.apache.org/docs/latest/ml-features.html#vectorindexer).
The problem with `VectorIndexer` is that it will index every feature which has
less than `maxCategories` (which you can set with `setMaxCategories`) no
matter whether it is indeed categorical or not. In our case, there are
categorical features with quite a few categories (`Title` for example) and
quantitative features without too many different values (such as `SibSp` or
`Parch`). That's why I don't think this is the way to go.
2. Index every feature, which you know is categorical, one by one with
[StringIndexer](http://spark.apache.org/docs/latest/ml-features.html#stringindexer).
At the time of this writing, there is, unfortunately, no way to create a
single `StringIndexer` which will index all your categorical features in one
step (there is a PR going on to do just that
[here](https://github.com/apache/spark/pull/9183) though).

We wil proceed with option 2 in order to have a bit more control over which
features is getting indexed:

{% highlight scala %}
val categoricalFeatColNames = Seq("Pclass", "Sex", "Embarked", "Title")
val stringIndexers = categoricalFeatColNames.map { colName =>
  new StringIndexer()
    .setInputCol(colName)
    .setOutputCol(colName + "Indexed")
    .fit(allData)
}
{% endhighlight %}

We also index our label which corresponds to the `Survived` column:

{% highlight scala %}
val labelIndexer = new StringIndexer()
.setInputCol("Survived")
.setOutputCol("SurvivedIndexed")
.fit(allData)
{% endhighlight %}
<br>

#### Assembling our features into one column

Now that our indexing is done, we just need to assemble all our feature columns
into one single column containing a vector regrouping all our features.
To do that, we'll use the built-in
[VectorAssembler](http://spark.apache.org/docs/latest/ml-features.html#vectorassembler)
transformer:

{% highlight scala %}
val numericFeatColNames = Seq("Age", "SibSp", "Parch", "Fare", "FamilySize")
val idxdCategoricalFeatColName = categoricalFeatColNames.map(_ + "Indexed")
val allIdxdFeatColNames = numericFeatColNames ++ idxdCategoricalFeatColName
val assembler = new VectorAssembler()
  .setInputCols(Array(allIdxdFeatColNames: _*))
  .setOutputCol("Features")
{% endhighlight %}

We'll now have two columns:

- `SurvivedIndexed` containing our indexed label
- `Features` containing a vector of our different features (quantitative and
indexed categorical)
<br><br>

#### Using a classifier

Now that our data is in the proper format expected by spark.ml we can use a
classifier. Here I'll use a
[RandomForestClassifier](http://spark.apache.org/docs/latest/ml-ensembles.html#random-forests)
but since our data is properly formatted we can replace it by any spark.ml
classifier we want:

{% highlight scala %}
val randomForest = new RandomForestClassifier()
  .setLabelCol("SurvivedIndexed")
  .setFeaturesCol("Features")
{% endhighlight %}
<br>

#### Retrieving our original labels

`IndexToString` is the reverse operation of `StringIndexer` and will convert
back our indexes to the original labels so they can be interpretable. Indeed,
as indicated in the [documentation for random forests](http://spark.apache.org/docs/latest/ml-ensembles.html#random-forests),
the call on the `transform` method of the model produced by the
`RandomForestClassifier` will produce a `prediction` column which will contain
indexed labels which we need *unindexed*.

{% highlight scala %}
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)
{% endhighlight %}
<br>

#### Bringing it all together

Since all our different steps have been implemented, we can create our pipeline:

{% highlight scala %}
val pipeline = new Pipeline().setStages(Array.concat(
  stringIndexers.toArray,
  Array(labelIndexer, assembler, randomForest, labelConverter)
))
{% endhighlight %}

We first apply each `StringIndexer` for every one of our categorical features
and our label, we then assemble every feature into one column. Then, we train
our random forest and we finally convert back the indexed labels predicted to
the original ones.
<br><br>

### Selecting the best model

In order to select the best model, you'll often find yourself performing a grid
search over a set of parameters, for each combination of parameters do cross
validation and keep the best model according to some performance indicator.

This is a bit tedious and spark.ml aims to simplify that with an easy-to-use
API.

A quick reminder if you don't know what
[cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
is: you chose a number `k` of folds, for example 3, your dataset will be split
into three parts, from those 3 parts, 3 different pairs of training and test
data will be generated (2/3 of the data for the training and 1/3 for the test).
Then the model is evaluated on the average of the chosen performance indicator
over the three pairs.

First, we're going to want to create a grid of parameters:

{% highlight scala %}
val paramGrid = new ParamGridBuilder()
  .addGrid(randomForest.maxBins, Array(25, 28, 31))
  .addGrid(randomForest.maxDepth, Array(4, 6, 8))
  .addGrid(randomForest.impurity, Array("entropy", "gini"))
  .build()
{% endhighlight %}

The different parameters for spark.ml's random forests can be found in the
[scaladoc](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.RandomForestClassifier).

Next, we need to define an `Evaluator` which, as its name implies, will evaluate
our model according to some metric. There are three built-in evaluator: one for
regression, one for binary classification and another one multiclass
classification. In our case, we're only interested in the
[BinaryClassificationEvaluator](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.evaluation.BinaryClassificationEvaluator).
The default metric used for binary classification is the area under
[the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic).
A `BinaryClassificationEvaluator` can be created in the following way:

{% highlight scala %}
val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("SurvivedIndexed")
{% endhighlight %}

However, another metric is available for binary classification: the area under
[the precision-recall curve](https://en.wikipedia.org/wiki/Precision_and_recall)
which can be used with:

{% highlight scala %}
val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("SurvivedIndexed")
  .setMetricName("areaUnderPR")
{% endhighlight %}

We also need an `Estimator` to be trained, in our case, it will be our whole
pipeline.

Finally, after chosing `k=10`, the number of folds the data will be split into
during cross validation, we can create a `CrossValidator` object like so:

{% highlight scala %}
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(10)
{% endhighlight %}

Since our `CrossValidator` is an `Estimator`, we can obtain the best model for
our data by calling the `fit` method on it:

{% highlight scala %}
val crossValidatorModel = cv.fit(data)
{% endhighlight %}

We can now make predictions on the `test.csv` file given by Kaggle:

{% highlight scala %}
val predictions = crossValidatorModel.transform(dfToPredict)
{% endhighlight %}

WARNING: You have to be careful when running cross validation, especially on
bigger datasets, as it will train `k x p` models where `k` represents the number
of folds used for cross validation and `p` is the product of the number of
values for each param in your grid.

If we go back to our previous example with `k=10` and the following parameter
grid:

{% highlight scala %}
val paramGrid = new ParamGridBuilder()
  .addGrid(randomForest.maxBins, Array(25, 28, 31)) // 3 different values
  .addGrid(randomForest.maxDepth, Array(4, 6, 8)) // 3 different values
  .addGrid(randomForest.impurity, Array("entropy", "gini")) // 2 different values
  .build()
{% endhighlight %}

We get `p = 3 x 3 x 2 = 18`, so our cross validation will train
`k x p = 10 x 18 = 180` different models.
<br><br>

### Submitting the results to Kaggle

Now that we have our predictions, we just need to transform our data in order to
fit the expected format by Kaggle and save it to a csv file:

{% highlight scala %}
predictions
  .withColumn("Survived", col("predictedLabel"))
  .select("PassengerId", "Survived")
  .coalesce(1)
  .write
  .format(csvFormat)
  .option("header", "true")
  .save(outputFilePath)
{% endhighlight %}

There is still one more step to be performed: the output file will unfortunately
be in a directory in the `part-[0-9]{5}` hadoop format. As a result, I wrote a
little script to launch the Spark job and rename the output file so it is ready
to be submitted to Kaggle:

{% highlight bash %}
#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT="classified.csv"
TMP_FILE="${OUTPUT}2"

rm -rf ${OUTPUT}
rm -rf ${TMP_FILE}

cd ${DIR}
mvn clean package
spark-submit \
  --class com.github.benfradet.Titanic \
  --master local[2] \
  target/titanic-1.0-SNAPSHOT.jar \
  src/main/resources/train.csv src/main/resources/test.csv ${OUTPUT}

mv ${OUTPUT}/part-00000 ${TMP_FILE}
rm -rf ${OUTPUT}
mv ${TMP_FILE} ${OUTPUT}
{% endhighlight %}
<br><br>

### Conclusion

I hope this was an interesting introduction to spark.ml and that I could convey
the simplicity and expressiveness of the API.

For information, I managed to score 0.80383 on the contest.
