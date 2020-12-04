---
title: "Mathematics of neural networks"
teaching: 30
exercises: 0
questions:
- "What are the main mathematical building blocks of Deep Learning?"
objectives:
- "FIX"
keypoints:
- "FIX"
---

## A first Hello World example.

We will use the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, a classic machine learning algorithm. The dataset consists of 60,000 training images and 10,000 testing images. Each element in the dataset correspond to a grayscale image of hand written digits (28x28 pixels). You can think of solving MNIST, this is, classifying each number into one of 10 categories or *class*, as the "Hello World" of deep learning and can be used to verify that algorithms work correctly.

We can use Keras to work with MNIST since it comes preloaded as a set of four Numpy arrays:
~~~
from keras.datasets import mnist
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
~~~
{: .language-python}

~~~
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
11493376/11490434 [==============================] - 7s 1us/step
~~~
{: .output}


The model will learn from the training set composed by *train_images* and *train_labels* and then will be tested on the test set *test_images* and *test_labels* . The images are encoded as Numpy arrays, and the labels are an array of digits, ranging from 0 to 9. We can check which digit is placed in position 8 in the training set (remember that numpy arrays are zero-based indexed) with:

> ## Training vs Test
> Unfortunately, we cannot use the data we used to build the model to evaluate it. This is because our model can always simply remember the whole training set, and will therefore always predict the correct label for any point in the training set. This “remembering” does not indicate to us whether our model will generalize well (in other words, whether it will also perform well on new data).
>
> To assess the model’s performance, we show it new data (data that it hasn’t seen before) for which we have labels. This is usually done by splitting the labeled data we have collected (here, our 150 flower measurements) into two parts. One part of the data is used to build our machine learning model, and is called the training data or training set. The rest of the data will be used to assess how well the model works; this is called the test data, test set, or hold-out set.
> Unfortunately, we cannot use the data we used to build the model to evaluate it. This is because our model can always simply remember the whole training set, and will therefore always predict the correct label for any point in the training set. This “remembering” does not indicate to us whether our model will generalize well (in other words, whether it will also perform well on new data).
{: .callout}

~~~
train_labels[7]
~~~
{: .language-python}
~~~
3
~~~
{: .output}

If we wanted to see the corresponding image we need an additional Python library to be able to plot images:
~~~
import matplotlib.pyplot as plt
digit = train_images[7]
plt.imshow(digit,cmap=plt.cm.binary)
~~~
{: .language-python}

<img src="{{ page.root }}/fig/mnist_number_three.png" alt="MNIST - Number three" width="25%" height="25%" />

We can further confirm the size of our sets:
~~~
print("Number of train images: ", train_images.shape)
print("Number of train labels: ", train_labels.shape)
print("Number of test images: ", test_images.shape)
print("Number of test labels: ", test_labels.shape)
~~~
{: .language-python} 

~~~
Number of train images:  (60000, 28, 28)
Number of train labels:  (60000,)
Number of test images:  (10000, 28, 28)
Number of test labels:  (10000,)
~~~
{: .output}

## Tensors
At this point we have been working with some multidimensional Numpy arrays, also known as *tensors*, fundamental to Machine Leaning methods. A matrix is a type of two-dimensional tensor with which you might already be familiar, but scalars and vectors are also categorized as tensors (0D and 1D respectively) as well as higher dimensional arrays (3D, 4D and upwards).
Tensors are defined by three key attributes:
 - *Rank* - This is the number of axes in the tensor. 
 - *Shape* - This is how many dimensions the tensor has in every axis.
 - *Data type* - The elements of a NumPy array are usually numbers, but can also be booleans, strings, or other objects. When containing numbers, these must be of the same type, (integers or floating point numbers) and size.
 
For example our *train_images* array is a rank 3 tensor with one axis corresponding to the number of images and the other two representing the image's pixels. It has 60000 images (size of first axis) and 28 in each of the other two.  We can confirm this with:
~~~
print("Number of axis in train_images: ",train_images.ndim)
print("Number of dimensions in each axis of train_images:", train_images.shape)
print("Type of number contained in train_images:", train_images.dtype)
~~~
{: .language-python}

~~~
Number of axis in train_images:  3
Number of dimensions in each axis of train_images: (60000, 28, 28)
Type of number contained in train_images: uint8
~~~
{: .output}


## Batches
In general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis (sometimes called the samples dimension ). In the MNIST example, samples are images of digits. In addition, deep-learning models don’t process an entire dataset at once; rather, they break the data into small batches. Concretely, here’s one batch of our MNIST digits, with batch size of 128:

When considering such a batch tensor, the first axis (axis 0) is called the batch axis or batch dimension . This is a term you’ll frequently encounter when using Keras and other deep-learning libraries.


