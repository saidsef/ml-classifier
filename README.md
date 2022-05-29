# Machine Learning - News Articles classification with sklearn [![CI](https://github.com/saidsef/ml-classifier/actions/workflows/docker.yml/badge.svg)](#deployment) [![Tagging](https://github.com/saidsef/ml-classifier/actions/workflows/tagging.yml/badge.svg)](#deployment) [![Release](https://github.com/saidsef/ml-classifier/actions/workflows/release.yml/badge.svg)](#deployment) [![Release](https://github.com/saidsef/ml-classifier/actions/workflows/analysis.yml/badge.svg)](#deployment)

Classify news articles into different categories using Machine Learning.  The dataset consists of 5000 documents and 40 categories.

My goal is to show you how to create a predictive model that will classify news articles.

## Objective

- To classify news articles
- Learn the basics of natural language processing
- Build models using sklearn and choose the best one
- Use sklearn's make_pipeline class
- Learn how to turn it into a service
- Learn how to make it composable and portable
- Profit?

## Prerequisite

- Python >= v3.9
- Jupyter Notebook
- Some knowledge of Machine Learning

## Python Libs

- NumPy
- Pandas
- SciPy
- Matplotlib
- Jupyter
- Scikit-learn (the library that we will use later in this post when creating the predictive models)

## We Will

- Apply some preprocessing steps to prepare the data.
- Then, we will perform a descriptive analysis of the data to better understand the main characteristics that they have
- We will continue by practicing how to train different machine learning models using scikit-learn. It is one of the most popular python libraries for machine learning
- We will also use a subset of the dataset for training purposes
- Then, we will iterate and evaluate the learned models by using unseen data. Later, we will compare them until we find a good model that meets our expectations
- Once we have chosen the candidate model, we will use it to perform predictions and to create a simple web application that consumes this predictive model

## Getting started with the machine learning tutorial

See [Jupyter Notebook](https://machinelearningmastery.com/start-here/)

## Deployment

As a container:

```shell
docker run -d -p 7070:7070 docker.io/saidsef/ml-classifier:latest
```

As a Python application:

```shell

pip install -r requirements.txt

PORT=7070 classifier-ml.py
```

## JSON Format

The payload should be [JSON format](test/test.json)

```shell
{
  "body": "text-goes-here"
}
```

## The Request

The quest must be `POST` method:

```shell
curl -XPOST http://lcoalhost:7070/api/v1/news -H 'Content-Type: application/json' -d @test/test.json
```

And the response will look like:

```json
{
  "score": 1,
  "category": "Arts & Life"
}
```

## Kubernetes

```shell
kubectl apply -k ./deployment
```
