Variational Autoencoders
========================
(c) Deniz Yuret, 2019-11-12, last updated: 2021-08-25

The goal of VAE is to model your data $X$ coming from a complicated distribution $P(X)$ using a latent (unobserved, hypothesized) variable $Z$:

$$P(x) = \int P(x|z) P(z) dz$$

This identity is true for any distribution $P$ and any value $x$. VAE takes $P(Z)$ to be the multivariate standard normal. Note that this identity can also be written as an expectation: 

$$P(x) = E_{z\sim P(Z)}[P(x|z)]$$

and can be approximated by sampling $z_n$ from $P(Z)$: 

$$P(x) \approx \frac{1}{N} \sum_{z_n\sim P(Z)} P(x|z_n)$$

However for high dimensional spaces (images, text) typically modeled by VAE, this would be a poor approximation because for a given $x$ value, $P(x|z)$ would be close to 0 almost everywhere. Randomly sampling from $P(Z)$ would be unlikely to hit regions of $Z$ space where $P(x|z)$ is high. Say we had a distribution $Q(Z|X)$ which is more likely to give us $z$ values where $P(x|z)$ is high. We could rewrite our former identity as:

$$P(x) = \int P(x|z) P(z) Q(z|x) / Q(z|x) dz$$

Note that this identity can also be expressed as an expectation:

$$P(x) = E_{z\sim Q(Z|x)}[P(x|z) P(z) / Q(z|x)]$$

and can be approximated by sampling $z_n$ from $Q(Z|x)$ (this is called importance sampling and would converge faster because $Q$ gives us better $z$ values): 

$$P(x) \approx \frac{1}{N} \sum_{z_n\sim Q(Z|x)} P(x|z_n) P(z_n) / Q(z_n|x)$$

To train a VAE model we pick some parametric functions $P_\theta(X|Z)$ (i.e. decoder, likelihood, generative network) and $Q_\phi(Z|X)$ (i.e. encoder, posterior, inference network) and fiddle with their parameters to maximize the likelihood of the training data $D=\{x_1,\ldots,x_M\}$. Actually, instead of likelihood $P(D) = \prod P(x_m)$ we use log likelihood: $\log P(D) = \sum\log P(x)$ because it nicely decomposes as a sum over each example. We now have to figure out how to approximate $\log P(X)$.

$$\log P(x) = \log E_{z\sim Q(Z|x)}[P(x|z) P(z) / Q(z|x)]$$

Jensen's inequality tells us that log of an expectation is greater than or equal to the expectation of the log:

$$\log P(x) \geq E_{z\sim Q(Z|x)}\log[P(x|z) P(z) / Q(z|x)]$$

The RHS of this inequality is what is known in the business as ELBO (evidence lower bound), more typically written as:

$$\log P(x) \geq E_{z\sim Q(Z|x)}[\log P(x|z)] - D_{KL}[Q(Z|x)\,\|\,P(Z)]$$

This standard expression tells us more directly what to compute but obscures the intuition that ELBO is just the expected log of an importance sampling term.

To see the exact difference between the two sides of this inequality we can use the integral version:

$$\begin{align*}
\log & P(x) - \int \log[P(x|z) P(z) / Q(z|x)] Q(z|x) dz \\
= & \int [\log P(x) - \log P(x|z) - \log P(z) + \log Q(z|x)] Q(z|x) dz \\
= & \int [\log Q(z|x) - \log P(z|x)] Q(z|x) dz \\
= & D_{KL}[Q(Z|x)\,\|\,P(Z|x)] 
\end{align*}$$

This allows us to write an exact equation, indicating the error of our approximation is given by the KL divergence between $Q(Z|x)$ and $P(Z|x)$:

$$\begin{align*} 
\log & P(x) - D_{KL}[Q(Z|x)\,\|\,P(Z|x)] = \\
   & E_{z\sim Q(Z|x)}[\log P(x|z)] - D_{KL}[Q(Z|x)\,\|\,P(Z)] 
\end{align*}$$

**Reference:** [Tutorial on Variational Autoencoders by Carl Doersch](https://arxiv.org/abs/1606.05908)
