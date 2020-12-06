---
title: Setup
---

For this workshop you will need to:
-  In general we will work with datasets that come preinstalled with python libraries
   but for lesson 2 we will use a couple of additional datasets. You can download 
   them from this [<ins>link</ins>]({{ page.root }}/files/lesson02-gdp-bli.zip).
-  Install Anaconda, so you have Python and the Jupyter notebook. You can follow 
   the instructions provided by [<ins>Software Carpentry</ins>](https://datacarpentry.org/python-ecology-lesson/setup.html).
-  Once anaconda is installed, be sure to have the following libraries (we have 
   tested the listed versions but other might also work).
     -    python=3.7
     -    keras=2.3
     -    matplotlib=3.3
     -    tensorflow=2.0
     -    tensorflow-gpu=2.0
     -    ipykernel=5.3
     -    scikit-learn=0.23.2
     -    pandas=1.1.3
     -    pip=20.2
     -    tensorflow=2.3
    
   Or if you prefer you can use [<ins>this yml file</ins>]({{ page.root }}/files/environment.yml) 
   to create a virtual environment with all the necessary libraries. You can use the
   following command to install and activate your environment:
   ~~~
   conda activate machine-learning
   ~~~
   {: .language-bash}
   If you wish to run this environment from Jupyter Notebooks you can run the 
   following command from the base environment:
   ~~~
   python -m ipykernel install --user --name=machine-learning
   ~~~
   {: .language-bash}
   This should make the environment available from Jupyter Notebooks.

{% include links.md %}
