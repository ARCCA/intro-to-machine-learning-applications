---
title: "Scikit Learn - The Iris Dataset"
teaching: 60
exercises: 0
questions:
- "Using Scikit Learn to analize a machine learning problem?"
objectives:
- "FIX"
keypoints:
- "FIX"
---

In this lesson we will use a popular machine learning example, the Iris dataset, to understand some of the most basic concepts around machine learning applications. For this, we will employ Scikit-learn one of the most popular and prominent Python library for machine learning. Scikit-learn comes with a few standard datasets, for instance the [iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) and [digits](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) datasets for classification and the diabetes dataset for [regression](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html), you can find more information about these and other datasets in the context of Scikit-learn usage [here](https://scikit-learn.org/stable/datasets/index.html#datasets).

## The Iris Dataset

The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters. You can find out more about this dataset [here](https://en.wikipedia.org/wiki/Iris_flower_data_set) and [here](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).


> ## Features
> In machine learning datasets, each entity or row here is known as a sample (or data point), while the columns—the properties that describe these entities—are called features.
>
{: .callout}

To start our work we can open a new Python session and import our dataset:

~~~
from sklearn.datasets import load_iris
iris_dataset = load_iris()
~~~
{: .language-python}

> ## Datasets
> In general, in machine learning applications, a dataset is a dictionary-like object that holds all the data and some metadata about the data. In particular, when we use Scikit-learn data loaders we obtain an object of type *Bunch*:
> ~~~
> type(iris_dataset)
> ~~~
> {: .language-python}
> 
> ~~~
> sklearn.utils.Bunch
> ~~~
> {: .output}
>
> This type of object is a container that exposes its keys as attributes. You can find out more about Scikit-learn *Bunch* objects [here](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html#sklearn.utils.Bunch).
{: .callout}

Take a look at the keys in our newly loaded dataset:
~~~
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
~~~
{: .language-python}

~~~
Keys of iris_dataset: 
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
~~~
{: .output}

The dataset consists of the following sections:
 - **data**: contains the numeric measurements of sepal length, sepal width, petal length, and petal width in a NumPy array. The array contains 4 measurements (*features*) for 150 different flowers (*samples*).
 - **target**: contains the species of each of the flowers that were measured, also as a NumPy array. Each entry consists of a integer number in the set 0, 1 or 2.
 - **target_names**: contains the meanings of the numbers given in the **target_names** array: 0 means setosa, 1 means versicolor, and 2 means virginica.
 - **DESCR**: a short description of the dataset.
 - **feature_names**: a list of strings, giving the description of each feature in **data**.
 - **filename**: location where the dataset is stored.

And we can explore the contents of these sections:
~~~
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
print("\n")
print("Targets:\n{}".format(iris_dataset['target'][:]))
print("\n")
print("Target names:\n{}".format(iris_dataset['target_names']))
print("\n")
print("Feature names:\n{}".format(iris_dataset['feature_names']))
print("\n")
print("Dataset location:\n{}".format(iris_dataset['filename']))
~~~
{: .language-python}

~~~
First five columns of data:
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]


Targets:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]


Target names:
['setosa' 'versicolor' 'virginica']


Feature names:
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']


Dataset location:
/home/vagrant/anaconda3/envs/deep-learning/lib/python3.7/site-packages/sklearn/datasets/data/iris.csv
~~~
{: .output}

In particular, try printing the content of the **DESCR** key:
~~~
print(iris_dataset['DESCR'])
~~~
{: .language-python}

~~~
.. _iris_dataset:

Iris plants dataset
--------------------

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
                
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988

The famous Iris database, first used by Sir R.A. Fisher. The dataset is taken
from Fisher's paper. Note that it's the same as in R, but not as in the UCI
Machine Learning Repository, which has two wrong data points.

This is perhaps the best known database to be found in the
pattern recognition literature.  Fisher's paper is a classic in the field and
is referenced frequently to this day.  (See Duda & Hart, for example.)  The
data set contains 3 classes of 50 instances each, where each class refers to a
type of iris plant.  One class is linearly separable from the other 2; the
latter are NOT linearly separable from each other.

.. topic:: References

   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.
     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
     Structure and Classification Rule for Recognition in Partially Exposed
     Environments".  IEEE Transactions on Pattern Analysis and Machine
     Intelligence, Vol. PAMI-2, No. 1, 67-71.
   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
     on Information Theory, May 1972, 431-433.
   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
     conceptual clustering system finds 3 classes in the data.
   - Many, many more ...
~~~
{: .output}

## Training

Scikit-learn's *train_test_split* function allow us to shuffle and split the dataset in a single line. The function takes a sequence of arrays (the arrys must be of the same length) and options to specify how to split the arrays. By default, the function extracts 75% of the rows in the arrays as the training set while the remaining 25% of rows is declared as the test set. Deciding how much data you want to put into the training and the test set respectively is some‐ what arbitrary, but using a test set containing 25% of the data is a good rule of thumb. The function also allow us to control the shuffling applied to the data before applying the split with the option *random_state*, this ensures reproducible results.

~~~
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=0)
print("X_train shape: {}".format(X_train.shape)) 
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape)) 
print("y_test shape: {}".format(y_test.shape))
~~~
{: .language-python}

~~~
X_train shape: (112, 4)
y_train shape: (112,)
X_test shape: (38, 4)
y_test shape: (38,)
~~~
{: .output}


We can use Pandas' plotting function *scatter_matrix* that draws a matrix of scatter plots to visualize our data. This function takes a dataframe and plots each variable contained within, against each other. The first step is import Pandas and transfor our Numpy array into a Pandas dataframe:
~~~
import pandas as pd
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8)
~~~
{: .language-python}

<img src="{{ page.root }}/fig/scatter_matrix_iris_dataset.png" alt="Scatter Matrix of Iris Dataset" width="35%" height="35%" />

The diagonal in the image shows the distribution of the three numeric variables of our example data. In the other cells of the plot matrix, we have the scatter plots of each variable combination of our dataframe. Take a look at Pandas' *scatter_matrix* [documentation](https://pandas.pydata.org/pandas-docs/version/1.1.3/reference/api/pandas.plotting.scatter_matrix.html#pandas.plotting.scatter_matrix) to find out more about the options we have used. Keep in mind as well that Pandas' *scatter_matrix* makes use of Matplotlib's *scatter* function under the hood to perform the actual plotting and that we have passed a couple of arguments to this function in our previous command, specifically point color and size options. Find out more at Matplotlib's *scatter* function [documentation](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html).

Back to the plots, we can see that the three classes seem to be relatively well separated using the sepal and petal measurements. This means that a machine learning model will likely be able to learn to separate them.

## Building the model
There are [many](https://scikit-learn.org/stable/supervised_learning.html) classification algorithms in scikit-learn that we could use. Here we will use a [k-nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification) classifier, which is easy to understand. Neighbors-based classification algorithms are a type of instance-based learning that does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

Building this model only consists of storing the training set. To make a prediction for a new data point, the algorithm finds the point in the training set that is closest to the new point. Then it assigns the label of this training point to the new data point.

## Plot the decision boundaries


Plot the decision boundary. For that, we will assign a color to each
point in the mesh [x_min, x_max]x[y_min, y_max].

