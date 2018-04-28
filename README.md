## TensorFlow Introduction

TensorFlow is an open source Library for doing Complex Numerical Computation to build machine learning models, which is developed by Google Brain team. TensorFlow is written in C++ but is quite user-friendly to python interface. TensorFlow has three main components TensorFlow(API), TensorFlow Serving and TensorBoard. TensorFlow(API) contains the API's to develop models and to train the models with data. TensorFlow Serving is designed to deploy the models. And TensorBoard is designed to visualize, analyze, and debug machine learning models.
From my perspective, TensorFlow is currently most popular and powerful framework for building deep learning model.

## Install
I am now going to introduce how to install TensorFlow on macOS 10.12.6 (Sierra) or higher.
Right now, the most popular programming language that use TensorFlow is python. Below is a screenshot from github by searching "tensorflow"
![](https://github.com/zhenyangy/5914_techteam/blob/master/1.png)

Hence I am going to provide install steps for python with anaconda environment
1. install anaconda following the steps here: [anaconda install](https://docs.anaconda.com/anaconda/install/mac-os#macos-graphical-install)
2. Create a conda environment named tensorflow by invoking the following command:
```
$ conda create -n tensorflow pip python=2.7 # or python=3.3, etc.
```
3. Activate the conda environment by issuing the following command:
```
$ source activate tensorflow
```
4. Issue a command of the following format to install TensorFlow inside your conda environment:
```
(targetDirectory)$ pip install --ignore-installed --upgrade TF_PYTHON_URL
```
where TF_PYTHON_URL is https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.7.0-py2-none-any.whl for python 2.7 or https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.7.0-py3-none-any.whl for python 3.4, 3.5, or 3.6.

## what to start

In TensorFlow, the dataset is represented as tensors. so, what is tensor?
* Tensors are the standard way of representing data in TensorFlow
* Tensors are basically multidimensional arrays, an extension of two-dimensional tables(matrices) to data with higher dimensions

Tensors Rank

| rank |            Math Entity           |          Python Example          |
|------|:--------------------------------:|:--------------------------------:|
| 0    |      Scalar (magnitude only)     |             s = 5914             |
| 1    | Vector (magnitude and direction) |         v = [5, 9, 1, 4]         |
| 2    |     Matrix (table of numbers)    | m = [[5, 9, 1, 4], [5, 9, 1, 4]] |
| 3    |    3-Tensor (tube of numbers)    |   t = [[[5], [9]], [[1], [4]]]   |
| n    |             n-Tensor             |                ...               |

In addition to the dimensionality, Tensors also have various data type (float32, int32, or string, for example) and in python it is simply 
```
tf.float32 # or tf.bool, etc.
```

Now that we have our data as tensors. We need to build something called computational graph for our model.
The computational graph essentially defines the structure of the model such as neural network. (To get some sense about structure of neural network, you can play around with [NN playground](https://playground.tensorflow.org/) Here is a simple example:
```
import tensorflow as tf

W = tf.variable([.3], tf.float32)   #weight
b = tf.variable([-.5], tf.float32)  #bias

x = tf.placeholder(tf.float32)      #input data
y = tf.placeholder(tf.float32)      #inputdata

model = W * x + b                   #linear model

sqaure_delta = tf.square(model - y) #square error
loss = tf.reduce_sum(squared_delta) #computes sum
```
Now, to run this computational graph, we need to run it within a **session**. As the session encapsulates the control and state of the TensorFlow runtime:
```
sess = tf.session()

sess.run(tf.global_variables_initializer())

print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))
```
The printed information is the loss of the model when the session ends, which is 0 (why?)

## TensorBoard

To visualize the computational graph, you would need to use tensorBoard.
```
File_Writer = tf.summary.FileWriter(path, sess.graph)
```
In our previous code example, it should be:
```
import tensorflow as tf
import os

W = tf.Variable([.3], tf.float32)   #weight
b = tf.Variable([-.5], tf.float32)  #bias

x = tf.placeholder(tf.float32)      #input data
y = tf.placeholder(tf.float32)      #inputdata

model = W * x + b                   #linear model

sqaure_delta = tf.square(model - y) #square error
loss = tf.reduce_sum(sqaure_delta) #computes sum

sess = tf.Session()
File_Writer = tf.summary.FileWriter(os.getcwd(), sess.graph)
sess.run(tf.global_variables_initializer())

print(sess.run(loss, {x:[1, 2, 3, 4], y:[0, -1, -2, -3]}))
```
Then, in terminal, go to the parent directory of the directory that contains the graph file, input following command:
```
tensorboard --logdir='parent dir'
```
Then you will get responses:
```
TensorBoard 1.7.0 at http://Zhenyangs-MacBook-Pro.local:6006 (Press CTRL+C to quit)
```
Now, open a browser and got to http://localhost:6006/. You can then view the graph!
![Alt Text](https://github.com/zhenyangy/5914_techteam/blob/master/tensorboard.gif)

## Reference
https://www.tensorflow.org/api_docs/python/tf/Tensor
