---
title: Mathlab (TM)
shorttitle: Mathlab (TM)
notebook: Lab2.ipynb
noline: 1
keywords: ['gradient descent', 'integration']
layout: wiki
---


## Contents
{:.no_toc}
* 
{: toc}

Here we will go through some examples of solving math questions in calculus, vectors and matrices. We'll revisit these at other points in the course, as needed, but hopefully this notebook provides you some practice, some cheatsheet, etc

A lot of the material here is taken from Daume's "Math for machine learning" and  Bengio's book on deep learning.

## Differential calculus

The derivative of a function is the slope of the function at that point. For example, velocity is the derivative of the position.

Its defined thus

$$\frac{df}{dx}\Big{\vert}_{x=x_0} = \lim_{h \to 0}\frac{f(x_0 + h) - f(x_0)}{h}$$

We'll abuse the notation $\partial_x$ for derivative with respect to x for now, as its more concise.

Some formulae:

![](https://www.dropbox.com/s/ouihdyln5nneh90/Screenshot%202017-02-03%2002.41.02.png?dl=1)

Let's try some functions. You will want to write the answer here and you can check it against the lab notebook

**Q1.** Differentiate $xe^{-x^2}$

**Q2.** Differentiate  $\frac{e^x + 1}{e^{-x}}$


A function is said to be **convex** if it looks like a bowl. In other words, you are guaranteed a **global** minimum. Its second derivative must be positive everywhere.

A **concave function** is an inverted bown. It has a global laximum, and its second derivative must be negative everywhere.

Critical points in 1D, from Bengio.

![](https://www.dropbox.com/s/196s7o9cbrjbbjx/Screenshot%202017-02-03%2003.20.30.png?dl=1)

**Q3.** What kind of a function is $2x^2 - 3x + 1$.

## Integral Calculus

There are two separate concepts here. One is the anti-derivative, or "find the function whose derivative I am". The second concept is the area under the curve, also called the integral. The fundamental theorem of calculus tells us that the antiderivative is the integral. 

This diagram, taken from wikipedia, illustrates this theorem:

![](https://upload.wikimedia.org/wikipedia/commons/e/e6/FTC_geometric.svg)

**Q4.** Integrate  $\int_{-\infty}^{\infty} e^{-x^2}$

**Q5.** What is the antiderivative of $x^2$; ie $\int x^2 dx$.

## Vectors

I am going to assume that you are familiar with vectors like  ${\bf v}$, and their dot product ${\bf v} \cdot {\bf w}$. 

**Q6.** Write down the vector which has equal weight in all three directions, and is a unit vector (ie of length 1.)

You can learn much more about vectors here http://www.feynmanlectures.caltech.edu/I_11.html .

## Multidimensional Calculus

This is where the partial symbol comes into its own. If $f$ is a function like $x^2 + y^2$, you can take derivatives with respect to both x and y, which gives us:

$$\frac{\partial{f}}{\partial{x}} = 2x$$


You can combine partials into a vector. This vector is known as the gradient vector

$$\nabla f = (2x, 2y)$$

The gradient will always give you the direction of the greatest change of a function. Why is this?

** Q7.** Compute the gradient of $f(\mathbf{w}) = \mathbf{w}^T\mathbf{w}$

** Q8** We saw the Jacobian in class on thursday. Compute the 3-D jacobian for a tranformation from cartesian $=(x,y,z)$ co-ordinates to cylindrical ones $(r, \phi, z)$, and find its determinant.

## The Jacobian and the Hessian

We saw the Jacobian. The Hessian is a matrix of second derivatives, and can be used to probe curvature

**Q9.** Calculate the Hessian of $$f(x) = x_1^2 + x_2^2$$. Make a 3D plot of this function to see what it looks like.

## Continuity

**Q10.** What is the derivative of $f(x) = \vert x \vert$. 

**Q11.** What is the derivative of the step function which is 0 for all x less than 0 and 1 otherwise

## Taylor Series

The taylor series expansion for a function about a point a is given as

$$f(x) = \sum_i  \frac{f^{(n)}(x-a)}{n!}x^n$$

where $f^{(n)}$ is the nth derivative.

**Q11** Expand $e^x$ in a taylor series

**Q12** Approximately calculate $1.05^5$ using a taylor series.

**Q13** Show that a log likelihood function of one parameter $\theta$ is normally distributed near its maximum.



```python

```

