---
toc: true
layout: post
description: How to define algorithmic fairness and bias?
categories: [markdown]
title: AI Fairness
---
# Introduction

The concept of equity is implicitly learned by humans since childhood. This results in an approximate and subjective understanding that may emphasize different aspects and considerations: impartiality, merit, etc.

Machine Learning techniques are increasingly used in decision-making contexts with important societal implications. These automated decision-making mechanisms act under explicit or implicit conceptions of equity.

In some cases, a decision based on irrelevant elements can be characterized as biased (for example, ethnicity or gender in a hiring process), but this can also be the case for a completely arbitrary decision in other contexts (for example, giving the trophy to a random player).

A list of artificial intelligence projects showing these drifts has emerged and is growing month by month, ranging from a racist chatbot to a sexist recruitment algorithm.

One of the best known cases is COMPAS (Correctional Management Profiling for Alternative Sanctions), a decision-support software used in the United States to determine the likelihood that an accused may become a repeat offender. In an article published in 2016, ProPublica demonstrates a blatant bias in COMPAS decision making towards certain populations by systematically attributing a significantly higher risk to them.

The primary objective of this article is to introduce and raise awareness about the complexity of the issue of fairness in artificial intelligence: what is the state of legislation on the issue? How to define and quantify algorithmic biases? And in practice, how to correct them?


# European legislation closely monitoring the fairness of machine learning models
Since the implementation of the GDPR (General Regulation on Data Protection) in May 2018, work has been carried out by the European Commission to promote an ecosystem of trust for artificial intelligence. This work was written and published in February 2020 in a white paper "Public consultation towards a European approach for excellence and trust". The objective is to extend the regulation of GDPR to AI during 2021 by building on the 7 requirements listed in this white paper  by encouraging innovation around ethical AI.

This willingness to establish a regulatory framework encourages companies to take the necessary steps to develop fair algorithms, especially those in sectors most prone to discrimination.

Algorithmic biases have a greater impact in certain areas in particular. In the medical field, for example, the use of data collected from certain populations could have serious consequences on under-represented or absent populations. Some studies have already established that many data sets used in the medical field are biased. One example is the 23andme genome dataset, which is said to have only 3% African people.

In the financial sector, equity issues are increasingly emphasized. Recent work by the ACPR (French Prudential Supervision and Resolution Authority) shows that few actors have measured and sought to correct existing biases: "Exploratory work carried out by the ACPR, even when complemented by a more general study of the financial sector, has shown that only a few actors in the financial sector have begun to address the issue of detecting and correcting model biases". The existence of biases in this sector can also have far-reaching consequences, for example in the granting of loans, a case that is very frequently studied when looking at biases.


# How to define algorithmic fairness?

There is no universal definition of fairness today. In the algorithmic framework of decision making, Mehrabi et al. give the following very general definition which includes two essential elements that are discrimination and group distinction :

“Algorithmic fairness refers to the absence of any favouritism or discrimination against an individual or group formed by innate or acquired characteristics.”

It is therefore a matter of verifying and evaluating the absence of any harm, a discrimination that could be caused by decisions made by an algorithm. The challenge is then to precisely define these biases that one wishes to avoid in order to set up metrics to measure and correct them.


## Protected / sensitive variables

The groups mentioned in the previous definition can be defined using variables called protected variables or sensitive variables. These variables allow the socio-cultural characterization of each piece of data. Gender, skin color, nationality, religion, family status, age are characteristics explicitly and legally defined as sensitive.

An interesting remark concerning these variables is that there may be other non-sensitive variables linked to these first ones indirectly. There may indeed be a correlation between a sensitive variable and another non-sensitive variable that influences the final decision. For example, gender may be strongly related to occupation. Occupation is not a protected variable, but an imbalance of this variable in the data used could lead to gender bias in the decisions made by the algorithm.

Therefore, what to do with sensitive or indirectly sensitive variables? Removing directly or indirectly sensitive variables when training a Machine Learning model is not necessarily a good strategy. Indeed, an algorithmic bias may still remain at the output of the model. Removing them can even worsen existing biases.  

It is necessary to cross-reference the sensitive variables with the decisions of the model in order to be able to identify potential existing biases corresponding to discriminatory decisions made by the model. However, the use of these variables is not necessarily easy when considering of the General Data Protection Regulation (GDPR).


# Measures of Algorithmic Bias

There are a multitude of aspects of algorithmic fairness, each of which is associated with mathematical measures to quantify it.

One could imagine a set of metrics forming a universal framework for determining the decision biases introduced by an algorithm, but in reality, combining some of these metrics often proves difficult or even impossible.

Thus, there is no universal measure, nor are some metrics better than others. Each metric must be considered according to the subject matter. For example, when recruiting, the goal is to ensure that a man has exactly the same chance as a woman to be hired.

On the other hand, in the context of automobile insurance, knowing that men have on average more accidents than women, one may want to check that for a given score, we observe the same probability of having actually had an accident for men and women. We thus evaluate groups calibration (defined below).

Thus, it is necessary in the course of each project to identify the risks and issues associated with decision making process of the algorithm, and to deduce the most appropriate metrics. Some of the most commonly used definitions are given below.

For this introductory article, we present only the case of binary classification. In addition to its simplicity, it is also the most studied case in the literature and finds multiple applications (credit granting, hiring, customer scoring, etc.).

In order to clarify these definitions, they are illustrated by the example of the granting of credit, which consists in predicting the probability that a loaned sum will be well repaid by the individual, i.e. calculating the risk associated with credit default.


The following notations are introduced beforehand :

Y: the target class that the algorithm tries to predict. More precisely, in the example of the granting of credit, Y is 1 if the person has repaid the loan and 0 otherwise.

S: the variable representing the score associated with the algorithm's prediction (value between 0 and 1). In the example, the closer the score is to 0, the higher the risk that the person will not repay the loan.

Ŷ: the decision made by the algorithm. This variable is linked to the score and is equal to 0 if the score is below a defined threshold, or 1 otherwise. In the example, if Ŷ is 0 then the algorithm predicts a credit default.

G: the sensitive variable that defines the groups for which we want to measure the different biases. In the example, G corresponds to the gender of the person. We want to verify that the algorithm does not favor either men or women.


## 1. Independence
Les métriques qui cherchent à satisfaire le critère d’independence mesurent l’influence des groupes définis par la variable sensible sur la classe prédite.
Parmi ces mesures, on retrouve notamment la définition suivante :
Statistical Parity
Cette métrique évalue si chaque groupe a la même probabilité d’appartenir à la classe prédite positive.

### Statistical parity
This metric evaluates whether each group has the same probability of belonging to the positive predicted class


In the case of credit granting, for example, the model compares the probabilities of being able to repay a loan (given by the model) according to whether one is a man or a woman. If 4 men and 12 women apply for a loan, and if the algorithm decides to grant 8 loans, we want 2 men and 6 women to be selected (2/4 = 6/12).

## 2. Separation
The independence criteria do not take into account the target variable Y.
Definitions based on the separation criterion take this target variable into account in order to measure the independence between the score obtained by the algorithm and the sensitive variable conditioned by the target variable. These definitions allow G to be explanatory of Y, contrary to the previous definitions which assume that Y is independent of G. They thus take into account a potential influence of the sensitive variable on the target variable.

### Equal opportunity
This metric compares the rates of true positives in the different groups.

In our previous example, we thus observe the probabilities that the model predicts credit repayment, depending on whether it is for a man or a woman, knowing that the person has actually repaid the amount borrowed.

We consider a sample of 100 people containing 50 women and 50 men. Of the 50 women, 40 have repaid their loans, and only 20 of the men have repaid their loans. We imagine that among the persons selected by the algorithm, 30 persons have actually repaid their credit (the other persons selected are therefore errors in the algorithm). Then the algorithm satisfies the introduced metric if among these 30 persons, 20 are women and 10 are men (20/40=10/20).

This metric therefore takes into account the information provided by the sensitive variable on the target variable, i.e. in this example men are less reliable than women when granting credit.


## 3. Sufficiency

The metrics grouped around the sufficiency criterion seek to measure the independence of the target variable Y with respect to group G conditioned by the score variable S. This family of metrics is incompatible with the equilibrium metrics of the positive and negative classes.


### Test fairness or calibration

We test here that the probability of belonging to the positive class is the same for a given score:

In the previous example, we consider a score given by the model, and for all the persons having obtained this score, we compare the probabilities that this person has actually repaid the credit according to whether he is a man or a woman.


# And in practice, how can these biases be corrected?

The analysis and correction of algorithmic biases can be integrated into the different levels of a Machine Learning project: pre-processing, in-processing, post-processing.

Pre-processing (data)
This step allows to analyze and process the distribution of the model training data (discriminating sensitive variables, class imbalance). The objective is to apply transformations on the input data to mitigate biases and make them more equitable.

In-processing (modeling/optimization)
In this step, the main treatments consist in defining equity metrics to be optimized in conjunction with performance metrics during model training.

Post-processing (results)
The objective at this stage is to analyze and transform the model results to identify and address potential biases related to protected variables or subgroups related to these variables.



As pre-processing and post-processing methods act only on input data or results, they do not require transparency of the model and can treat it as a black box. In particular, this allows the Machine Learning libraries to be used directly without explicit modification. On the other hand, this kind of processing can make the model less interpretable.



# Conclusion

As Machine Learning techniques are increasingly used in important decision-making contexts, it becomes necessary to ensure that this does not generate unforeseen impact.
with negative societal implications. These should be specified during the project to identify existing biases in the data in order to derive the right indicators. A multitude of methods exist to correct these biases and can be integrated into the different stages of a Machine Learning project. The choice of these methods depends, among other things, on the metrics chosen and the interpretability constraints. These metrics and correction methods can already be found in open-source python libraries. Nevertheless, there is still a lack of scientific and technological consensus on the choice of the right correction algorithm according to the business use case, while keeping a good fair/performance compromise, even if some have already tried it.


# References

https://arxiv.org/pdf/1908.09635.pdf
https://github.com/wikistat/Fair-ML-4-Ethical-AI#annexe-extraits-du-guide-des-experts-de-la-ce-pour-une-ia-digne-de-confiance
https://papers.nips.cc/paper/2020/file/7ec2442aa04c157590b2fa1a7d093a33-Paper.pdfhttps://arxiv.org/pdf/1609.05807.pdf
https://arxiv.org/abs/1609.05807
https://arxiv.org/abs/1908.09635
https://www.quantmetry.com/blog/ia-de-confiance-exigence-et-opportunite-europeenne/
https://ec.europa.eu/info/sites/info/files/commission-white-paper-artificial-intelligence-feb2020_fr.pdf
https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing


### Remarque
This is a modified version.
This original article is in French. Written by: Salah Chadli, Philippe Neveux, Thibaud Real Del Sarte.

