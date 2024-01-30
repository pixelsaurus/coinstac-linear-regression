# COINSTAC: Interactive Linear Regression

**Synopsis:** This is a COINSTAC computation which approximates a linear regression analysis run on data from multiple sites using multiple rounds of communication.

**Analytical Description:** 
Suppose there are $S$ sites. Each site $s$ has a a collection of $n_s$ covariate-response pairs $\{ (x^{(s)}_j, y^{(s)}_j) : j = 1, 2, \ldots, n_s\}$ where each $x^{(s)}_j$ is a $d$-dimensional vector of covariates (real numbers) stored as a $d \times 1$ array and $y^{(s)}_j$ is a scalar response. Define the total number of samples as $N = \sum_{s=1}^{S} n_s$.

This computation uses an interactive protocol for computing the least-squares regression coefficients corresponding to a single aggregated data set:

$$\beta = \mathop{\mathrm{argmin}}_{\beta} \sum_{s=1}^{S} \sum_{j=1}^{n_s} (y^{(s)}_j - \beta^{\top} x^{(s)}_j )^2$$

This computation uses an interactive protocol to do this by emulating gradient descent (GD) at the aggregator. The aggregator sends an initialized $$\beta_1$$ to all the sites to start. At the $t$-th iteration the sites compute the gradient of their "local objective":

$$F_s(\beta_t) = \sum_{j=1}^{n_s} (y^{(s)}_j - \beta_t^{\top} x^{(s)}_j )^2$$

Each site $s$ computes $g_{s,t} = \nabla F_s(\beta_t)$ and sends it to the aggregator. The aggregator updates its global model

$$\beta_{t+1} = \beta_{t} - \eta_t \sum_{s=1}^{S} g_{s,t}$$

The aggregator sends $\beta_{t+1}$ to the sites to use in the next iteration.

## Required Preprocessing

In a linear regression model, we are given covariate-response pairs $\{ (v_j, y_j) : j = 1, 2, \ldots, n\}$ and try to fit a model

$$y = b_0 + b_1 v(1) + b_2 v(2) + \ldots b_{d-1} v(d-1)$$

using least squares. To simplify the problem, we append a 1 to the covariate vector and define
	$$\begin{bmatrix} x_j(1) \\ x_j(2) \\ \vdots \\ x_j(d) \end{bmatrix} = \begin{bmatrix} v_j(1) \\ v_j(2) \\ \vdots \\ v_j(d-1) \\ 1 \end{bmatrix}$$

so that the model is 

$$y = \beta^{\top} x$$

where $\beta = [b, b_0]$. This computation assumes that this preprocessing step has been done already.


## Local and aggregator computations

### Local script

The local script at site $s$ does the following at initialization:

1. Reads a local data set $\{ (x^{(s)}_j, y^{(s)}_j) : j = 1, 2, \ldots, n_s\}$ from disk.

At each iteration $t$ the site does the following:

1. Receives $\beta_t$ from the aggregator.
2. Computes $g_s = \nabla F_s(\beta_t)$.
3. Sends $g_s$ to the aggregator.
4. Deletes the previous iterate $\beta_t$.

### Aggregator script

The aggregator does the following:

1. Receives $\{ g_{s,t} : s = 1, 2, \ldots, S\}$ from the $S$ sites.
2. Updates the coefficients $\beta_{t+1} = \beta_{t} - \eta_t \sum_{s=1}^{S} g_{s,t}$.
3. Sends $\beta_{t+1}$ to all sites.
4. Deletes the messages $\{ g_{s,t} : s =1, 2, \ldots, S\}$.

## Communication and storage specification

Let $T$ be the total number of iterations.


**What data must sites provide?**

* Each site $s$ needs to provide access to their covariate-response pairs $\{ (x^{(s)}_j, y^{(s)}_j) : j = 1, 2, \ldots, n_s\}$.

**What is shared from the sites to the aggregator?**

* The site ID
* The local gradients $\{ g_{s,t} : t = 1, 2, \ldots, T\}$ from each iteration.

**What intermediate resultes are stored locally at the sites?**

* The sites receive the sequence of coefficient updates $\{ \beta_t : t = 1, 2, \ldots, T\}$.

**What intermediate results are stored at the aggregator?**

* The aggregator deletes $\{ g_{s,t} : s = 1, 2, \ldots S\}$ after computing $\beta_{t+1}$ at each iteration.

**what is the output from the computation?**

This computation produces a single output file:

* *format*: CSV
* *content*: 
  * $d$-dimensional vector $\beta_{T+1}$
  * ...



