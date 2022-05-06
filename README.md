# MLOps_MNIST_Pytorch

This project is a piece of a possible MLOps workflow that allows data scientist, researchers, students or independent developer to develop Machine Leaning project in Continuous Integration (CI) with most popular open-source or free tools.

-- Project Status: [Active]

## Project Intro/Objective

The purpose of this project is to present a set of practices and tools that are part of the MLOps philosophy to develop, deploy and maintain data science projects.
All the tools used in this repo are open source or free (for personal use) and are among the most used tools in data science.

Note: the main goal of this project is not to perform the classification task but rather to see how to create a python project for data science that is shareable and reproducible.

### Method used

In this repository, we develop a CNN model in [Pytorch Ignite](https://pytorch.org/ignite/index.html) to classify [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

We ensure the training monitoring with [TensorBoard](https://www.tensorflow.org/tensorboard) and/or [Neptune.ai](https://neptune.ai/).

We use GitHub Actions to run the training and testing.

We can use [DVC (Data Version Control)](https://dvc.org/) to data  versioning and DVC Gdrive remote for data storage.
We can use [CML (Continuous Machine Learning)](https://cml.dev/) as CI/CD machine learning.


### Technologies
- Python (Pytorch Ignite)
- neptune and tensorboard loggers
- MLOps workflow
- CI machine learning

## Project description /  Motivation

In the field of data science, developing a project requires a lot of organization, discipline and various tools.
For each step of the development of a machine learning project there are many tools, and it is difficult to know which one to choose.

That's why I propose in this repo a selection of tools that are quite easy to use, open source, free and among the most used in data science.
Note this is only a proposal of possible workflows and there are many others.

The classic steps in a data science project are the following:
1. Problem statement:
2. Project initialization:
3. Data preparation:
4. Model development:
5. Documentation:


### 1) Problem statement:

It is a very important step: to understand the problem and the data. 
  - What is your current process ? 
  - What do you want to predict ?
  - How do you get the data ?

### 2) Project initialization:

Define timeline, deadline and team for the project: 
- How long will it take ?
- Who will be the team ?
- When will you start ?
Jira is a good tool to manage this.

Create GitHub repository:
- Create a repository on GitHub initialized with cookiecutter.
GitLab is a good other option.

### 3) Data preparation:

- Understand and find the data:
  - Define data needed
  - What are your data source ?

- Prepare data:
  - Data extraction
  - Data pre-processing
  - Exploratory Data Analysis (EDA)
  - prepare data (Building train and test datasets)

- Storage and version data:
  - A good solution to versioning and storage data for free is [DVC](https://dvc.org/). For example, you can use [DVC Gdrive remote](https://dvc.org/docs/remote/gdrive/).

### 4) Model development:

An example of good environment in Linux to develop a model in python is:
- IDE: [Pycharm](https://www.jetbrains.com/pycharm/)
- Machine learning library: [Pytorch](https://pytorch.org/) + [Ignite](https://pytorch.org/ignite/)
- Monitoring: [TensorBoard](https://www.tensorflow.org/tensorboard/) and./or [Neptune.ai](https://neptune.ai/).
- Testing: pytest, pylint, flake8, black
- Notebook: [Jupyter](https://jupyter.org/) or [Datalor](https://datalore.jetbrains.com/) or google colab.
- Continuous integration (CI): [GitHub Actions]
- Cloud computing: [GCP](https://cloud.google.com/), [Azure](https://azure.microsoft.com/en-us/services/compute/), [AWS](https://aws.amazon.com/), [Google Colab](https://colab.research.google.com/).

An example of CI machine learning is:
- CI loop:
  - init CI with CML + DVC + GitHub action
  - Make change and runs experiments. Ensure changed code doesn't break functionalities by running sanity checks (local).
  - optimize model (autoPyTorch for example)
  - Developer creates pull request targeting the main branch
  - CI tests
  - CI server rerun the code. Submit the new code to the CI routine (CML + GitHub action) which will re-run the tests again on a neutral environment to ensure reproducibility.
  - CI server compare performance metrics with the main branch and reports back to the PR page on GitHub so other members can review it.
  - If  approved, the code is then merged to main.
  - CI server reruns the experiment again but this time the model is pushed to the remote storage for future deployment as well as a performance report for comparison.
  - If CI routine is too long, you can optimize it.

### 5) Documentation:

For a great and auto generated documentation, you can use [Sphinx](https://www.sphinx-doc.org/en/master/).


## Getting Started

In this repo you can find help to:
- create a CNN with Pytorch Ignite,
- define some sanity checks for machine learning,
- monitor training and validation results with TensorBoard and/or Neptune,
- initialise a data science project with a good architecture (cookiecutter),
- save and store data on GDrive with DVC,
- versioning data and model with DVC,
- create a Continuous Integration loop with GitHub Actions and CML.


## More resources

Nothing to add here....
