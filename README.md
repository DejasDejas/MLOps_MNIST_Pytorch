# MLOps_MNIST_Pytorch

This project is a piece of a possible MLOps workflow that allows data scientist, researchers, students or independent developer to develop Machine Leaning project in Continuous Integration (CI) with most popular open-source or free tools.

-- Project Status: [Active]

## Project Intro/Objective

The purpose of this project is to present a set of practices and tools that are part of the MLOps philosophy to develop, deploy and maintain data science projects.
All the tools used in this repo are open source or free (for personal use) and are among the most used tools in data science.

Note: the main goal of this project is not to perform the classification task but rather to see how to create a python project for data science that is shareable and reproducible.

### Method used

In this repository, we develop a CNN model in [Pytorch Ignite](https://pytorch.org/ignite/index.html) to classify [MNIST dataset](http://yann.lecun.com/exdb/mnist/.

We ensure the training monitoring with [TensorBoard](https://www.tensorflow.org/tensorboard) and/or [Neptune.ai](https://neptune.ai/).

We use [DVC (Data Version Control)](https://dvc.org/) to data  versioning and DVC Gdrive remote for data storage.
We use [CML (Continuous Machine Learning)](https://cml.dev/) as CI/CD machine learning. 

**TODO:** motivate my chosen method


### Technologies
- Python (Pytorch Ignite)
- neptune logger and tensorboard logger
- MLOps
- CI/CD

## Project description /  Motivation

In the field of data science, developing a project requires a lot of organization, discipline and various tools.
For each step of the development of a machine learning project there are many tools, and it is difficult to know which one to choose.

That's why I propose in this repo a selection of tools that are quite easy to use, open source, free and among the most used in data science.
Note that this is only a proposal of possible workflows and that there are many others.

The classic steps in a data science project are the following:
- Problem statement
- Define timeline, deadline and team for the project
- Create GitHub repository
- Data:
    - Define data needed
    - data extraction
    - data pre-processing
    - Exploratory Data Analysis (EDA)
    - prepare data (Building train and test datasets)
    - save data and init data versioning with DVC remote on GDrive.
- Model
    - define model pytorch ignite + tensorboard + neptune
    - Unitaire test (adapt for machine learning)
    - Training model for the first time (baseline)
    - Evaluate model
- CI loop:
    - init CI with CML + DVC + GitHub action (only for the first loop) [link](https://dvc.org/doc/user-guide/setup-google-drive-remote)
    - Make change and runs experiments. Ensure changed code doesn't break functionalities by running sanity checks (local).
    - optimize model (autoPyTorch)
    - Dev create pull request targeting the main branch
    - CI tests
    - CI server rerun the code. Submit the new code to the CI routine (CML + GitHub action) which will re-run the tests again on a neutral environment to ensure reproducibility.
    - CI server compare performance metrics with the main branch and reports back to the PR page on GitHub so other members can review it.
    - If  approved, the code is then merged to main.
    - CI server reruns the experiment again but this time the model is pushed to the remote storage for future deployment as well as a performance report for comparison.
    - Optim CI
- Deploy trained model (Model serving)
- Model monitoring
- Documentation
- Create a pip install for my application
- Maintain system:
  - tests:
    - speed
    - performance
    - etc...


## Getting Started

In this repo you can find help to:
- create a CNN with Pytorch Ignite,
- sanity tests for machine learning,
- monitor training and validation results with TensorBoard or Neptune,
- initialise a data science project with a good architecture (cookiecutter),
- save and store data on GDrive with DVC,
- versioning data and model with DVC,
- create a CI/CD loop with CML

**TODO:** add other readme.md with details for some folder (tests, cml.yaml, .dvc, etc....)

## More resources

Arrive soon...
