---
layout: post
title: Using k-nearest neighbors to diagnose cancer
categories:
    - machine learning
    - tutorial
    - R
---

In this post, I will walk you through an application of
[the k-nearest neighbors algorithm](http://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
to diagnose breast cancer.

In order to follow along, you will need [R](http://www.r-project.org/) installed
along with a couple of packages:

  - [class](http://cran.r-project.org/web/packages/class/index.html) which
contains various functions related to classification and notably the `knn`
function which will be used in this post
  - [gmodels](http://cran.r-project.org/web/packages/gmodels/index.html) which
will be useful to us towards the end of this post when we will want to evaluate
the performance of our algorithm with the `CrossTable` function

As a reminder, you can install a package in R with the following command:

{% highlight R %}
install.packages("name_of_the_package")
{% endhighlight %}

We will also need a dataset to apply the k-nearest neighbors algorithm. I chose
[the popular breast cancer data set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
which is publicy available and comes from [the repository of machine learning
datasets of UCI](http://archive.ics.uci.edu/ml).

You can find the code for this post on
[GitHub](https://github.com/BenFradet/kNNPost).

<br>

### Data exploration and data transformation
<br>

#### Getting to know the dataset without R

Without doing any R programming, you can learn a lot from the dataset by reading
[the document presenting it](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names).

For example, you can learn the names of the different features and what they
represent, the total number of examples and how they are split between benign
and malignant, etc.

We learn that there are actually 10 numeric features which are measured for each
cell nucleus and for each of these features the mean, the standard error and
the largest (or "worst") values are stored in the dataset.

Another useful nugget of information in this document is that there are no
missing attribute values in the dataset so we won't have to do much data
transformation which is great.

<br>

#### Adding the names of the features in the dataset

One thing you will notice when downloading [the dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
is that there are no header lines in the CSV file.

Personally, I don't like naming features in R and since the number of features
(32) is manageable, I added the names of the features directly in the CSV file.

Consequently, the first line becomes:

{% highlight text %}
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,
compactness_mean,concavity_mean,concave_point_mean,symmetry_mean,
fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,
compactness_se,concavity_se,concave_point_se,symmetry_se,fractal_dimension_se,
radius_largest,texture_largest,perimeter_largest,area_largest,smootness_largest,
compactness_largest,concavity_largest,concave_point_largest,symmetry_largest,
fractal_dimension_largest
{% endhighlight %}

The modified dataset can be found on [GitHub](https://github.com/BenFradet/kNNPost/blob/master/wdbc.data).

<br>

#### Reading data into R

We are now ready to load the data into R:

{% highlight R %}
wdbc <- read.csv('wdbc.data', stringsAsFactors = FALSE)
{% endhighlight %}

As you may have noticed from the dataset or the names of the features, there is
an `id` column in the dataset and since it doesn't contain any information it
should be removed from the dataset.

{% highlight R %}
wdbc <- wdbc[-1]
{% endhighlight %}

We will also need to transform our `diagnosis` feature which contains "B" or "M"
depending on whether the cancer was benign or malignant into a factor to be able
to use the `knn` function later.

{% highlight R %}
wdbc$diagnosis <- factor(wdbc$diagnosis, levels = c('B', 'M'),
    labels = c('Benign', 'Malignant'))
{% endhighlight %}

<br>

#### Random permutation

Another thing we will need to is to randomly permute our examples in order to
avoid any kind of ordering which might be already present in the dataset. This
is important because if the dataset were ordered by diagnosis for example and if
we were to split our dataset between a training set and a test set, as we will
do later, the test set would be filled by either only benign or malignant tumors
to which the algorithm wouldn't have been confronted against during the training
phase.

{% highlight R %}
wdbc <- wdbc[sample(nrow(wdbc)), ]
{% endhighlight %}

<br>

#### Feature scaling

If you have a look at the range of the different mean features with:

{% highlight R %}
> lapply(wdbc[2:11], function(x) { max(x) - min(x) })
$radius_mean
[1] 21.129

$texture_mean
[1] 29.57

$perimeter_mean
[1] 144.71

$area_mean
[1] 2357.5

$smoothness_mean
[1] 0.11077

$compactness_mean
[1] 0.32602

$concavity_mean
[1] 0.4268

$concave_point_mean
[1] 0.2012

$symmetry_mean
[1] 0.198

$fractal_dimension_mean
[1] 0.04748
{% endhighlight %}

You can see that there are ranges of features which are 1000 times bigger than
others, this is a problem because since the kNN algorithm relies on distance
measurement it will give more importance to features with larger values.

In order to prevent our algorithm from giving too much importance to a few
features, we will need to normalize them. One way to do so is to use the `scale`
function:

{% highlight R %}
wdbcNormalized <- as.data.frame(scale(wdbc[-1]))
{% endhighlight %}

You can check that the mean of every feature is null now:

{% highlight R %}
summary(wdbcNormalized[c('radius_mean', 'area_worst', 'symmetry_se')])
{% endhighlight %}

<br>

#### Splitting the dataset in two: training and test sets

We will have to split our dataset in order to train our algorithm (training set)
and then evaluate its performance (test set).

A good rule of thumb is to take 75% of the dataset for the training and leave
the rest as a test set.
We learned from [the dataset description](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names)
that there were 569 examples in our dataset, so we'll take the first 427
examples as our training set and the last 142 as our test set.

{% highlight R %}
wdbcTraining <- wdbcNormalized[1:427, ]
wdbcTest <- wdbcNormalized[428:569, ]
{% endhighlight %}

We will also need the diagnosis feature as a separate factor in order to use the
`knn` function which we'll use later in this post.

{% highlight R %}
wdbcTrainingLabels <- wbdc[1:427, 1]
wdbcTestLabels <- wdbc[428:569, 1]
{% endhighlight %}

<br>

### Training our model

<br>

#### Running kNN

We are now ready to use the `knn` function contained in the `class` package.
The function takes four arguments:

  - `train` which is our training set
  - `test` the training set
  - `class` which should be a factor containing the classes of the training set
  - `k` the number of nearest neighbors to consider in order to predict the
class of a new example

This function will give us back a factor containing the predicted class for
each example in the test set. Since we have already stored the actual classes of
our test set in the `wdbcTestLabels` variable, we will be able to compare both
and evaluate the performace of the algorithm.

<br>

#### Choosing the parameter k

A good place to start when choosing k is to pick the square root of the number
of examples in our training set. Here we have 427 examples in our training
set, and so the square root of 427 is approximately 21. Keep in mind that you
are not stuck with this value, and it is often a good idea to fiddle around with
the value of k to see if we can get better results as we will see later in this
post.

We are now ready to use the `knn` function:

{% highlight R %}
k <- 21
wdbcPredictedLabels <- knn(train = wdbcTraining,
                           test = wdbcTest,
                           cl = wbdcTrainingLabels,
                           k)
{% endhighlight %}

<br>

### Evaluating the performance of our model

<br>

You can have a pretty good idea of how your model is performing by computing
the percentage of right predictions the algorithm made:

{% highlight R %}
actualVsPredicted <- cbind(wdbcTestLabels, wdbcPredictedLabels)
colnames(actualVsPredicted) <- c('actual', 'predicted')
percentage <- sum(apply(actualVsPredicted, 1,
                        function(row) { ifelse(row[1] == row[2], 1, 0) }
        )) / dim(actualVsPredicted)[1]
{% endhighlight %}

You will notice that we lose the factor class when we perform the `cbind` and we
are left with 1 for Benign and 2 for Malignant. Here, it doesn't really matter
for the computation of our percentage of right predictions.

I personally get around 96% of right predictions. This result depends on how
"lucky" you were with your random permutation. If there are atypical examples
contained only in the test set, your model didn't learn from them and,
consequently, they might have been misclassified.

Another way to evaluate the performance of your algorithm is to create a
`CrossTable` to check on how we misclassified our examples:

{% highlight R %}
CrossTable(x = wdbcTestLabels, y = wdbcPredictedLabels,
           prop.chisq = F, dnn = c('actual', 'predicted'))
{% endhighlight %}

This is the table I get with my particular permutation, as previously
mentioned, your results may differ:
<br>

|                  | predicted |                 |                 |
| ---------------- | --------- | --------------- | --------------- |
|      **actual**  |**Benign** |   **Malignant** |   **Row Total** | 
|  **Benign**      |        94 |               0 |              94 | 
|                  |     1.000 |           0.000 |           0.662 | 
|                  |     0.940 |           0.000 |                 | 
|                  |     0.662 |           0.000 |                 | 
|                  |           |                 |                 |
| **Malignant**    |         6 |              42 |              48 | 
|                  |     0.125 |           0.875 |           0.338 | 
|                  |     0.060 |           1.000 |                 | 
|                  |     0.042 |           0.296 |                 | 
|                  |           |                 |                 |
|**Column Total**  |       100 |              42 |             142 | 
|                  |     0.704 |           0.296 |                 | 

<br>
We can see that we predicted 6 examples as benign although they were malignant
(false negatives).

Additionally, you can compute other indicators on how your algorithm is
performing:

For example, you can measure the precision which represents of all
examples where we predicted the tumor was malignant, what fraction of examples
is actually malignant?

![precision](http://bit.ly/1u2x4H6)

Here we see that we have perfect precision since we didn't have any false
positives.

We can also compute the recall:

![recall](http://bit.ly/1u2wMQD)

The recall represents of all examples where the tumor was malignant, what
fraction did we correctly detect as being malignant? That's where our algorithm
is not performing as well, we only detected 87.5% of malignant tumors as being
malignant, which leaves us which 12.5% of malignant tumors which we classified
as benign.

From the values of both the precision and recall we can compute
[the F1 score](http://en.wikipedia.org/wiki/F1_score) which measures our test's
accuracy, it can be thought of as a weighted average between precision and
recall:

![f1Score](http://bit.ly/1CoESpt)

This score prevents us from classifying every test example as malignant in order
to avoid false negatives (predict benign tumors although they are malignant)
because in this case our precision would greatly decrease (due to the fact that
we would have 94 false positives).

There are several ways you can improve the algorithm's performance one of which
is to toy around with the value of the parameter k. For example, one could
imagine trying a few odd numbers around 21 and computing the percentage of
misclassified examples as well as the number of false positives and false
negatives. It's important to choose an odd k to lessen the risk of randomly
choosing between malignant or benign in case there is the same number of votes
for each class. For example, if we were to choose 20 as k and 10 of the
nearest neighbors of an example were malignant and the 10 others were benign,
the algorithm would choose randomly between both.

I hope this was an interesting introduction to the k-nearest neighbors
algorithm. Do not hesitate to email me if you're having difficulties running or
understanding the code.
