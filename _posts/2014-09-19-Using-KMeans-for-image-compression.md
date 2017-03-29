---
layout: post
title: Using K-means for image compression
categories:
    - machine learning
    - tutorial
    - Octave
---

I recently used [the K-means algorithm](http://en.wikipedia.org/wiki/K-means_clustering)
to cluster the colors I had in a picture to create a, what I think is,
pretty cool profile picture. In this post, I will show you how I managed to do
that.

What you will need is a not too large picture (otherwise the algorithm will take
much too long) and [Octave](http://www.gnu.org/software/octave/) installed which
is available on pretty much any Linux distributions.

The code is available on [GitHub](https://github.com/BenFradet/KMeansPost) with
comments and line by line explanations.

### Running K-means

I will split the process of running K-means in two different functions:

  - `initCentroids` will initialize random centroids from pixels in the image
  - `runKMeans` will actually run the K-means algorithm

If you're familiar with the K-means algorithm you know that the algorithm is
based on two steps and so `runKMeans` will be calling two different
functions:

  - `assignCentroids` which will assign the closest centroid to each
example
  - `computeCentroids` which will compute new centroids from the assignments
done in `assignCentroids`

### The initialization function

As previously mentioned,
[initCentroids](https://github.com/BenFradet/KMeansPost/blob/master/initCentroids.m)
will pick K (here representing the resulting number of colors we want) random
examples from our dataset and choose them as our initial centroids.

It is defined as follows:

{% highlight octave %}
function centroids = initCentroids(X, K)

centroids = zeros(K, size(X, 2));
randIndex = randperm(size(X, 1));
centroids = X(randIndex(1:K), :);

end
{% endhighlight %}

### The centroid assignment function

[assignCentroids](https://github.com/BenFradet/KMeansPost/blob/master/assignCentroids.m)
loops over every example in the dataset and associates the closest centroid to
each and everyone of them.

{% highlight octave %}
function idx = assignCentroids(X, centroids)

K = size(centroids, 1);
idx = zeros(size(X, 1), 1);

for i = 1:size(X, 1)
    distances = zeros(1, K);

    for j = 1:K
        distances[1, j] = sumsq(X(i, :) - centroids(j, :));
    end

    [one, idx(i)] = min(distances);
end

end
{% endhighlight %}

### The compute centroids function

[computeCentroids](https://github.com/BenFradet/KMeansPost/blob/master/computeCentroids.m)
recomputes the centroids as the mean of the points which were assigned to each
centroid previously thanks to `assignCentroids`.

{% highlight octave %}
function centroids = computeCentroids(X, idx, K)

n = size(X, 2);
centroids = zeros(K, n);

for i = 1:K
    centroids(i, :) = mean(X(find(idx == i), :));
end

end
{% endhighlight %}

### The function running K-means

[runKMeans](https://github.com/BenFradet/KMeansPost/blob/master/runKMeans.m)
will call `assignCentroids` and `computeCentroids` successively a fixed number
of times (thanks to the `iter` parameter) and will use the initial centroids
computed by `initCentroids` as a starting point.

{% highlight octave %}
function [centroids, idx] = runKMeans(X, initialCentroids, iter)

m = size(X, 1);
K = size(initialCentroids, 1);
centroids = initialCentroids;
idx = zeros(m, 1);

for i = 1:iter
    idx = assignCentroids(X, centroids);
    centroids = computeCentroids(X, idx, K);
end

end
{% endhighlight %}

### Putting things together

In order to tie everything together I wrote a simple script which runs the
algorithm and saves the compressed image. It is defined as follows:

{% highlight octave %}
#!/usr/bin/octave -q

originalImgPath = argv(){1};
newImgPath = argv(){2};

K = str2num(argv(){3};
iter = str2num(argv(){4};

img = double(imread(originalImgPath));
img = img / 255;
imgSize = size(img);

X = reshape(img, imgSize(1) * imgSize(2), 3);

initialCentroids = initCentroids(X, K);

[centroids, idx] = runKMeans(X, initialCentroids, iter);

idx = assignCentroids(X, centroids);
compressedImg = centroids(idx, :);
compressedImg = reshape(compressedImg, imgSize(1), imgSize(2), 3);

imwrite(compressedImg, newImgPath);
{% endhighlight %}

You can also find this script [on GitHub](https://github.com/BenFradet/KMeansPost/blob/master/main.m).

As you may have noticed, this script takes the following command line arguments:

  - the path to the original image
  - the path to the compressed image to write to
  - the number of classes/colors wanted
  - the number of iterations to perform

As an example, this is what I originally had as a picture:

![](/images/originalPicture.jpg)

I used the script like so:

{% highlight bash %}
./main.m ~/Documents/originalPicture.jpg ~/Documents/newPicture.jpg 8 10
{% endhighlight %}

To get back the following image:

![](/images/newPicture.png)

Be careful though because, as you might have caught on, the complexity of the
whole algorithm is `O(i * np * K)` where:

  - i is the number of iterations to perform
  - np is the number of pixels (so the length * width of your picture)
  - K the number of classes/colors wanted

So if you have a very large image, or, alternatively if you want a large number
of colors or want to run the algorithm for many iterations, the script may
run for quite a while.

To give you an idea, I timed how much time my example took (I had a 535 * 535
image and I wanted 8 colors and ran the algorithm for 10 iterations):

{% highlight bash %}
time ./main.m ~/Documents/originalPicture.jpg ~/Documents/newPicture.jpg 8 10
{% endhighlight %}

gives me back approximately 9 minutes and 45 seconds.

I hope this post was a pretty good and fun introduction to the K-means
algorithm. Do not hesitate to reach me if you encounter difficulties running or
understanding the code.
