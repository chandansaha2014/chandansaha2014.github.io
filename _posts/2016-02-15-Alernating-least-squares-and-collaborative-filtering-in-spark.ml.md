---
layout: post
title: Alternating least squares and collaborative filtering in spark.ml
categories:
  - machine learning
  - tutorial
  - Spark
---

In this post, I'll show you how to use alternating least squares (ALS for short)
in [spark.ml](http://spark.apache.org/docs/latest/ml-guide.html).

*Disclaimer: This post is mostly a copy/paste from
[a pull request](https://github.com/apache/spark/pull/10411) I wrote for Spark
documenting ALS and collaborative filtering in general in spark.ml.
Since the PR will likely be incorporated in the 2.0 release which is still a few
months away, I thought I'd share it. This is also in response to this
[stackoverflow question](http://stackoverflow.com/questions/35219854/where-to-find-spark-ml-dataframe-implements-about-collaborative-filtering) asking about documentation
regarding collaborative filtering in spark.ml.*

<br>

### Collaborative filtering 

[Collaborative filtering](http://en.wikipedia.org/wiki/Recommender_system#Collaborative_filtering)
is commonly used for recommender systems. These techniques aim to fill in the
missing entries of a user-item association matrix. `spark.ml` currently supports
model-based collaborative filtering, in which users and products are described
by a small set of latent factors that can be used to predict missing entries.
`spark.ml` uses the [alternating least squares
(ALS)](http://dl.acm.org/citation.cfm?id=1608614)
algorithm to learn these latent factors.

The implementation in `spark.ml` has the following parameters:

* *numBlocks* is the number of blocks the users and items will be partitioned
into in order to parallelize computation (defaults to 10).
* *rank* is the number of latent factors in the model (defaults to 10).
* *maxIter* is the maximum number of iterations to run (defaults to 10).
* *regParam* specifies the regularization parameter in ALS (defaults to 1.0).
* *implicitPrefs* specifies whether to use the *explicit feedback* ALS variant
or one adapted for *implicit feedback* data, see more below (defaults to `false`
which means using *explicit feedback*).
* *alpha* is a parameter applicable to the implicit feedback variant of ALS that
governs the *baseline* confidence in preference observations (defaults to 1.0).
* *nonnegative* specifies whether or not to use nonnegative constraints for
least squares (defaults to `false`).
<br><br>

#### Explicit vs. implicit feedback

The standard approach to matrix factorization based collaborative filtering
treats the entries in the user-item matrix as *explicit* preferences given by
the user to the item, for example, users giving ratings to movies.

It is common in many real-world use cases to only have access to *implicit
feedback* (e.g. views, clicks, purchases, likes, shares etc.). The approach used
in `spark.ml` to deal with such data is taken from
[Collaborative Filtering for Implicit Feedback Datasets](http://dx.doi.org/10.1109/ICDM.2008.22).
Essentially, instead of trying to model the matrix of ratings directly, this
approach treats the data as numbers representing the *strength* in observations
of user actions (such as the number of clicks, or the cumulative duration
someone spent viewing a movie). Those numbers are then related to the level of
confidence in observed user preferences, rather than explicit ratings given to
items. The model then tries to find latent factors that can be used to predict
the expected preference of a user for an item.
<br><br>

#### Scaling of the regularization parameter

We scale the regularization parameter `regParam` in solving each least squares
problem by the number of ratings the user generated in updating user factors,
or the number of ratings the product received in updating product factors.
This approach is named "ALS-WR" and discussed in the paper
"[Large-Scale Parallel Collaborative Filtering for the Netflix Prize](http://dx.doi.org/10.1007/978-3-540-68880-8_32)".
It makes `regParam` less dependent on the scale of the dataset, so we can apply
the best parameter learned from a sampled subset to the full dataset and expect
similar performance.
<br><br>

### Examples

In the following examples, we load rating data from the
[MovieLens dataset](http://grouplens.org/datasets/movielens/), each row
consisting of a user, a movie, a rating and a timestamp.
We then train an ALS model which assumes, by default, that the ratings are
explicit (`implicitPrefs` is `false`).
We evaluate the recommendation model by measuring the root-mean-square error of
rating prediction.
<br>

#### Scala example

{% highlight scala %}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
object Rating {
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }
}

val ratings = sc.textFile("data/sample_movielens_ratings.txt")
  .map(Rating.parseRating)
  .toDF()
val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

// Build the recommendation model using ALS on the training data
val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
val model = als.fit(training)

// Evaluate the model by computing the RMSE on the test data
val predictions = model.transform(test)
  .withColumn("rating", col("rating").cast(DoubleType))
  .withColumn("prediction", col("prediction").cast(DoubleType))

val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")
{% endhighlight %}

You can have a look at the
[ALS Scala docs](http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.recommendation.ALS)
for more details on the API.

If the rating matrix is derived from another source of information (i.e. it is
inferred from other signals), you can set `implicitPrefs` to `true` to get
better results:

{% highlight scala %}
val als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setImplicitPrefs(true)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
{% endhighlight %}
<br>

#### Java example

{% highlight java %}
import java.io.Serializable;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.DataTypes;

public static class Rating implements Serializable {
  private int userId;
  private int movieId;
  private float rating;
  private long timestamp;

  public Rating() {}

  public Rating(int userId, int movieId, float rating, long timestamp) {
    this.userId = userId;
    this.movieId = movieId;
    this.rating = rating;
    this.timestamp = timestamp;
  }

  public int getUserId() {
    return userId;
  }

  public int getMovieId() {
    return movieId;
  }

  public float getRating() {
    return rating;
  }

  public long getTimestamp() {
    return timestamp;
  }

  public static Rating parseRating(String str) {
    String[] fields = str.split("::");
    if (fields.length != 4) {
      throw new IllegalArgumentException("Each line must contain 4 fields");
    }
    int userId = Integer.parseInt(fields[0]);
    int movieId = Integer.parseInt(fields[1]);
    float rating = Float.parseFloat(fields[2]);
    long timestamp = Long.parseLong(fields[3]);
    return new Rating(userId, movieId, rating, timestamp);
  }
}

JavaRDD<Rating> ratingsRDD = jsc.textFile("data/sample_movielens_ratings.txt")
  .map(Rating::parseRating);
DataFrame ratings = sqlContext.createDataFrame(ratingsRDD, Rating.class);
DataFrame[] splits = ratings.randomSplit(new double[]{0.8, 0.2});
DataFrame training = splits[0];
DataFrame test = splits[1];

// Build the recommendation model using ALS on the training data
ALS als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating");
ALSModel model = als.fit(training);

// Evaluate the model by computing the RMSE on the test data
DataFrame rawPredictions = model.transform(test);
DataFrame predictions = rawPredictions
  .withColumn("rating", rawPredictions.col("rating").cast(DataTypes.DoubleType))
  .withColumn("prediction", rawPredictions.col("prediction").cast(DataTypes.DoubleType));

RegressionEvaluator evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction");
Double rmse = evaluator.evaluate(predictions);
System.out.println("Root-mean-square error = " + rmse);
{% endhighlight %}

You can have a look at the
[ALS Java docs](http://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/recommendation/ALS.html)
for more details on the API.

In Java as well, if the rating matrix is derived from another source of
information (i.e. it is inferred from other signals), you can set
`implicitPrefs` to `true` to get better results:

{% highlight java %}
ALS als = new ALS()
  .setMaxIter(5)
  .setRegParam(0.01)
  .setImplicitPrefs(true)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating");
{% endhighlight %}
<br>

#### Python example

{% highlight python %}
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

lines = sc.textFile("data/sample_movielens_ratings.txt")
parts = lines.map(lambda l: l.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=long(p[3])))
ratings = sqlContext.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
rawPredictions = model.transform(test)
predictions = rawPredictions\
    .withColumn("rating", rawPredictions.rating.cast("double"))\
    .withColumn("prediction", rawPredictions.prediction.cast("double"))
evaluator =\
    RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
{% endhighlight %}

You can have a look at the
[ALS Python docs](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.recommendation.ALS)
for more details on the API.

Same in Python, if the rating matrix is derived from another source of
information (i.e. it is inferred from other signals), you can set
`implicitPrefs` to `True` to get better results:

{% highlight python %}
als = ALS(maxIter=5, regParam=0.01, implicitPrefs=True,
          userCol="userId", itemCol="movieId", ratingCol="rating")
{% endhighlight %}

<br>

### Conclusion

You can find the full examples and the scripts to run them on my repo
[sparkml-als](https://github.com/BenFradet/sparkml-als).

Hoping this was informative and made you want to try out ALS in spark.ml.
