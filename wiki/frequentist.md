---
title: Frequentist Stats
shorttitle: Frequentist Stats
layout: wiki
noline: 1
---

$$
\newcommand{\Ex}{\mathbb{E}}
\newcommand{\Var}{\mathrm{Var}}
\newcommand{\Cov}{\mathrm{Cov}}
\newcommand{\SampleAvg}{\frac{1}{N({S})} \sum_{s \in {S}}}
\newcommand{\indic}{\mathbb{1}}
\newcommand{\avg}{\overline}
\newcommand{\est}{\hat}
\newcommand{\trueval}[1]{#1^{*}}
\newcommand{\Gam}[1]{\mathrm{Gamma}#1}
$$

## What is data?

What is data? **Frequentist statistics** is one answer to this philosophical question. It treats data as a **sample** from an existing **population**.

This notion is probably clearest to you from elections, where some companies like Zogby or CNN take polls. The sample in these polls maybe a 1000 people, but they "represent" the electoral population at large. We attempt to draw inferences about how the population will vote based on these samples.

What did we just do? We made a 'point estimate' of the scale or rate parameter as a compression of our data.

### Point estimate: Maximum Likelihood Estimation

One of the techniques used to estimate parameters in frequentist statistics is **maximum likelihood estimation**. Briefly, the idea behind it is:

The product

$$
L(\lambda) = \prod_{i=1}^n P(x_i | \lambda)
$$

gives us a measure of how likely it is to observe values $x_1,...,x_n$ given the parameters $\lambda$. Maximum likelihood fitting consists of choosing the appropriate "likelihood" function $L=P(X \mid \lambda)$ to maximize for a given set of observations. How likely are the observations if the model is true?

An image can explain this better. We want to choose the distribution that maximizes the product of the vertical lines. Here the blue does better, but it is not clear if the blue is the best.

![](images/gaussmle.png)

Often it is easier and numerically more stable to maximise the log likelyhood:

$$
\ell(\lambda) = \sum_{i=1}^n ln(P(x_i \mid \lambda))
$$


## Multiple Point estimates

In frequentist statistics, the data we have in hand, is viewed as a **sample** from a population. So if we want to estimate some parameter of the population, like say the mean, we estimate it on the sample.

This is because we've been given only one sample. Ideally we'd want to see the population, but we have no such luck.

The parameter estimate is computed by applying an estimator $F$ to some data $D$, so $\est{\lambda} = F(D)$.


**The parameter is viewed as fixed and the data as random, which is the exact opposite of the Bayesian approach which you will learn later in this class. **

If you assume that your model describes the true generating process for the data, then there is some true $\trueval{\lambda}$ which defines the model. We dont know this. The best we can do to start with is to estimate  a lambda from the data set we have, say by using MLE, which we denote $\est{\lambda}$.

Now, imagine that I let you peek at the entire population in this way: I gave you some M data sets **drawn** from the population, and you can now find $\lambda$ on each such dataset, of which the one we have here is one.
So, we'd have M estimates of the parameter.

Thus if we had many replications of this data set: that is, data from other days, an **ensemble** of data sets, for example, we can compute other $\est{\lambda}$, and begin to construct what is called the **sampling distribution** of $\lambda$.

But we dont.

### Sampling Distribution of the parameter

What you are doing is sampling M Data Sets $D_i$ from the true population. We will now calculate M $\est{\lambda}_i$, one for each dataset. As we let $M \rightarrow \infty$, the distribution induced on $\est{\lambda}$ is the **sampling distribution of the estimator**.

We can use the sampling distribution to put confidence intervals on the estimation of the parameters, for example.

## Bootstrap

Bootstrap tries to approximate our sampling distribution. If we knew the true parameters of the population, we could generate M fake datasets. Then we could compute the parameter (or another estimator) on each one of these, to get a empirical sampling distribution of the parameter or estimator.

### Parametric Bootstrap

But we dont have the true parameter. So we generate these samples, using the parameter we calculated. This is the **parametric bootstrap**. The process is illustrated in the diagram below, taken from Shalizi:



![](images/parabootstrap.png)

There are 3 sources of error with respect to the sampling distribution that come from the bootstrap:

- simulation error: the number of samples M is finite. This can be made arbitrarily small by making M large
- statistical error: resampling from an estimated parameter is not the "true" data generating process. Often though, the distribution of an estimator from the samples around the truth is more invariant, so subtraction is a good choice in reducing the sampling error
- specification error: the model isnt quite good.

### Non-parametric bootstrap

To address specification error, alteratively, we sample with replacement the X from our original sample D, generating many fake datasets, and then compute the distribution on the parameters as before. This is the **non parametric bootstrap**. We want to sample with replacement, for if we do so, more typical values will be represented more often in the multiple datasets we create.

Here we are using the **empirical distribution**, since it comes without any model preconceptions. This process may be illustrated so:

![](images/nonparabootstrap.png)
