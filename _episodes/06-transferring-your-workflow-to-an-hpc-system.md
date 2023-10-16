---
title: "Transferring your workflow to an HPC system"
teaching: 30
exercises: 0
questions:
- "Main challenges and reasons to transition to an HPC system."
objectives:
- "Understand main steps related with moving to an HPC system"
keypoints:
- "Moving to an HPC system can be challenging but there are sources of help 
   that can make the transition easier."
- "As you progress in the development of machine and deep learning applications
   new challenges are likely to appear. Requesting help in those case can save
   you a lot of time."
---

It is very often the case when working on machine and deep learning problems to 
begin prototyping in your laptop or desktop computer, but as your make progress
you may soon find that the problem grows in complexity and your local hardware
might not be suitable anymore. Purchasing new hardware can be prohibitive
expensive and the technical details hard to understand. 

One solution for these situations is to approach services like [Google Colab](
https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
or [Amazon SageMaker](https://aws.amazon.com/sagemaker/) which offer access to
free GPU devices and can be quite attractive due to reducing the amount of
hardware configuration knowledge required to start producing results.

The downside is that, as before, as your model and datasets grow, you might
need the pay versions of these services to continue making progress at a 
suitable rate (especially when trying to meet a deadline). This option might 
work for expert data and machine learning scientists and practitioners who can
accurately budget number of core hours and amount of storage required, but 
might not work as well for somebody in the model developing stages and who has
started to test more challenging scenarios.

What to do in this situation? There is no simple answer, and most of the time
it might be to try to adapt to what you have, but it might also be that you are
a student/researcher in a university that provides you with a High Performance 
Computing system and that you can use it free of charge as is the case with 
Cardiff University's [Hawk Supercomputer](https://portal.supercomputing.wales).

The challenge with this solution is to adapt to a new working environment which 
includes learning *Linux* commands (in a text-based console!), concepts such as
job schedulers and scripting in new languages such as *Bash*. Fortunately,
there is also a dedicated team ([Advance Research at Cardiff](
https://www.cardiff.ac.uk/advanced-research-computing)) that works continuously
to try to make this transition as smooth as possible by offering regular
training sessions on these topics for new users, solving queries when a cryptic
software error occurs or any other technical challenge in the process of
implementing your machine or deep learning application.

## Making the transition
So, what is actually involved in this transition? For this training course we
will assume that you have or are willing to [request](
https://portal.supercomputing.wales/index.php/getting-access/) an account on
the Hawk supercomputer.

As mentioned above, you will need to learn new concepts on [Linux](
https://arcca.github.io/An-Introduction-to-Linux-with-Command-Line/),
programming on [*Bash*](
https://arcca.github.io/An-Introduction-to-Linux-Shell-Scripting/) and 
[general HPC terms](https://arcca.github.io/hpc-intro/) such as the use of job
schedulers. And to put all of them together to be able to write a *job script*
that will execute the application that you developed in your local computer or
cloud service of choice. We won't delve into the specifics of these topics
since you can follow the provided links to find dedicate training courses but
instead summarize the essential steps to transfer our deep learning model
develop in the previous lesson to Hawk.

## Exporting your Jupyter Notebook to a Python script
This is the first step in moving towards a non-interactive system and we have
some advice on this respect in our [Code migration to HPC systems][1]
training material. Following the advice in that link, you might end with a 
Python script similar to this one:

{::options parse_block_html="true" /}

<details><summary markdown="span"> **Click to expand the code**  
</summary>
```python
#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

print("Number of train images: ", train_images.shape)
print("Number of train labels: ", train_labels.shape)
print("Number of test images: ", test_images.shape)
print("Number of test labels: ", test_labels.shape)

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28 * 28,)))
network.add(layers.Dense(10,activation='softmax'))

network.summary()
network.get_weights()

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000,28 * 28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000,28 * 28))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images,train_labels,epochs=20,batch_size=128,verbose=2)
test_loss, test_acc = network.evaluate(test_images,test_labels,verbose=2)
print('test_acc:',test_acc)
```
</details>
<br/>

{::options parse_block_html="false" /}

## Transferring data
First, we need to be able to transfer our scripts and any data required. There
are a [few options](
https://arcca.github.io/hpc-intro/16-transferring-files/index.html) to reach 
this goal, but we will focus on using the terminal and *scp*

~~~
$ scp my-script.py username@hawklogin.cf.ac.uk
~~~
{: .language-bash}

The above command will transfer your script directly into your home directory
in the remove server (the supercomputer Hawk). You can repeat the command to 
transfer any other required files. After this, we can [login](
https://portal.supercomputing.wales/index.php/index/accessing-the-system/)
into the remote server to prepare things to run our application.

~~~
$ ssh username@hawklogin.cf.ac.uk
~~~
{: .language-bash}

## Installing software
HPC systems like Hawk usually provide a rich software environment with many
useful tools. However, it might be that we have a specific working environment
that we would like to replicate as is often the case when working with Anaconda
and Python libraries. In this case it is useful to 
[create a virtual environment](
https://arcca.github.io/slurm_advanced_topics/08-packages/index.html) 
with all the dependencies that your application requires. On Hawk you can do 
this with (using the environment yaml file provided for this course, check the
setup section):

~~~
$ module load anaconda
$ conda env create -f environment.yml
~~~
{: .language-bash}

The above command will make anaconda available in your environment to allow you
to run conda to install a virtual environment called *machine-learning* with
the packages listed in the file environment.yml. To activate the environment
use:

~~~
$ conda activate machine-learning
$ (machine-learning) $ conda list
~~~
{: .language-bash}

The last command above is useful to confirm that your environment has all the 
necessary libraries.

## Writing a job script.
One of the most challenging parts of working on an HPC system like Hawk is
getting used to write files know as *job scripts* that are simply text files
with *Bash* commands that are executed by *SLURM* (a job scheduler) that
decides in which order resources are allocated to users. For this training
course you can use the following template (pay attention to your instructor in
case modifications are required):

{::options parse_block_html="true" /}

<details><summary markdown="span"> A very simple job script for CPUs
(**Click to see the code**)  
</summary>
```bash
#!/bin/bash
#SBATCH --job-name=ml.example
#SBATCH --error=%x.e.%J
#SBATCH --output=%x.o.%J
#SBATCH --partition=dev
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --account=scwXXXX

module purge
module load keras/2.3.1
module list

time python mnist-test.py
```
</details>
<br/>

{::options parse_block_html="false" /}

{::options parse_block_html="true" /}

<details><summary markdown="span"> A very simple job script for GPUs
(**Click to see the code**)  
</summary>
```bash
#!/bin/bash
#SBATCH --job-name=ml.example.gpu
#SBATCH --error=%x.e.%J
#SBATCH --output=%x.o.%J
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=20
#SBATCH --gres=gpu:1
#SBATCH --account=scwXXXX

module purge
module load keras/2.3.1
module list

time python mnist-test.py
```
</details>
<br/>

{::options parse_block_html="false" /}


{::options parse_block_html="true" /}

<details><summary markdown="span"> A more complete job script example
(**Click to see the code**)  
</summary>
```bash
#!/bin/bash --login
#SBATCH --job-name=mnist
#SBATCH --error=%x.e.%j
#SBATCH --output=%x.o.%j
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:2
#SBATCH --account=scwXXXX

clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_3_exclusive"

echo '----------------------------------------------------'
echo ' NODE USED = '$SLURM_NODELIST
echo ' SLURM_JOBID = '$SLURM_JOBID
echo ' OMP_NUM_THREADS = '$OMP_NUM_THREADS
echo ' ncores = '$NP
echo ' PPN = ' $PPN
echo '----------------------------------------------------'
#
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo SLURM job ID is $SLURM_JOBID
#
echo Number of Processing Elements is $NP
echo Number of mpiprocs per node is $PPN
env

test=mnist
input_dir=$HOME/training
WDPATH=/scratch/$USER/${test}.$SLURM_JOBID
rm -rf ${WDPATH}
mkdir -p ${WDPATH}
cd ${WDPATH}

cp ${input_dir}/${test}.py ${WDPATH}

module purge
module load anaconda
module list

source activate machine-learning

start="$(date +%s)"
time python -u -i ${test}.py
stop="$(date +%s)"
finish=$(( $stop-$start ))
echo deep learning ${test} $SLURM_JOBID Job-Time $finish seconds
echo Keras End Time is `date`
```  

</details>
<br/>

{::options parse_block_html="false" /}

Hopefully the above script will allow recreate the examples with which we have 
worked in the training course.

## Other challenges
We have so far used very simple test cases to demonstrate the development of 
machine and deep learning applications. Real world applications are more 
complex and new challenges are likely to appear. Two of the more common 
include:

- *Data storage*: more complex models often require huge databases that are not 
  always simple to store even in HPC systems.
- *Optimization*: HPC systems tend to have high-end hardware CPU and GPU
  devices that often require compiling libraries with specialized settings to
  exploit special hardware features.

If you are having trouble with any of these issues or with any other, contact 
us at arcca-help@cardiff.ac.uk and we will try to help you solve it.


[1]: https://arcca.github.io/introduction-to-python/05-code-migration-01/index.html
