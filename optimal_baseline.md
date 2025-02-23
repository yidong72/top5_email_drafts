## Policy Gradient Theorem and Baseline Derivation

### Policy Gradient Theorem

The policy gradient theorem provides a way to compute the gradient of the expected return with respect to the policy parameters. Let's derive it step by step:

1. Define the objective function:
   $$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \gamma^t r_t\right]$$

2. Express the gradient:
   $$\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \gamma^t r_t\right]$$

3. Apply the likelihood ratio trick:
   $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \gamma^t r_t \nabla_\theta \log \pi_\theta(\tau)\right]$$

4. Expand the trajectory probability:
   $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \gamma^t r_t \sum_{t'=0}^T \nabla_\theta \log \pi_\theta(a_{t'}|s_{t'})\right]$$

5. Rearrange the summation:
   $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{t'=t}^T \gamma^{t'} r_{t'}\right]$$

6. Define the Q-function:
   $$Q^\pi(s_t, a_t) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t'=t}^T \gamma^{t'-t} r_{t'} | s_t, a_t\right]$$

7. Express the final policy gradient theorem:
   $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) Q^\pi(s_t, a_t)\right]$$

### Policy Gradient Theorem with Baseline

Now, let's introduce a baseline b(s_t) to reduce variance:

1. Add the baseline to the gradient:
   $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) (Q^\pi(s_t, a_t) - b(s_t))\right]$$

2. Prove that the baseline doesn't introduce bias:
   $$\mathbb{E}_{a_t \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t)] = b(s_t) \mathbb{E}_{a_t \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a_t|s_t)] = 0$$

   This is because $\mathbb{E}_{a_t \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a_t|s_t)] = 0$ (score function property).

## Optimal Baseline Derivation

To derive the optimal baseline that minimizes variance:

1. Express the variance of the gradient estimator:
   $$\text{Var}(\hat{g}) = \mathbb{E}[\|\hat{g}\|^2] - \|\mathbb{E}[\hat{g}]\|^2$$

2. Focus on minimizing $\mathbb{E}[\|\hat{g}\|^2]$ as $\|\mathbb{E}[\hat{g}]\|^2$ is independent of the baseline:
   $$\mathbb{E}[\|\hat{g}\|^2] = \mathbb{E}[\|\nabla_\theta \log \pi_\theta(a_t|s_t) (Q^\pi(s_t, a_t) - b(s_t))\|^2]$$

3. Expand the expectation:
   $$\mathbb{E}[\|\hat{g}\|^2] = \mathbb{E}[\|\nabla_\theta \log \pi_\theta(a_t|s_t)\|^2 (Q^\pi(s_t, a_t) - b(s_t))^2]$$

4. Differentiate with respect to b(s_t) and set to zero:
   $$\frac{\partial}{\partial b(s_t)} \mathbb{E}[\|\hat{g}\|^2] = -2\mathbb{E}[\|\nabla_\theta \log \pi_\theta(a_t|s_t)\|^2 (Q^\pi(s_t, a_t) - b(s_t))] = 0$$

5. Solve for b(s_t):
   $$b^*(s_t) = \frac{\mathbb{E}[\|\nabla_\theta \log \pi_\theta(a_t|s_t)\|^2 Q^\pi(s_t, a_t)]}{\mathbb{E}[\|\nabla_\theta \log \pi_\theta(a_t|s_t)\|^2]}$$

This optimal baseline minimizes the variance of the gradient estimator.

## Practical Implementation of Optimal Baseline

To use the optimal baseline practically:

1. Collect data from n rollouts:
   - For each state s_t, store action a_t, gradient $\nabla_\theta \log \pi_\theta(a_t|s_t)$, and estimated Q-value.

2. Compute the baseline for each unique state s_t:
   $$b^*(s_t) \approx \frac{\sum_{i=1}^{n_s} \|\nabla_\theta \log \pi_\theta(a_t^i|s_t)\|^2 Q^\pi(s_t, a_t^i)}{\sum_{i=1}^{n_s} \|\nabla_\theta \log \pi_\theta(a_t^i|s_t)\|^2}$$
   where n_s is the number of samples for state s_t.

3. Optimization techniques:
   - Use vectorized operations for faster computation.
   - Pre-compute and store $\|\nabla_\theta \log \pi_\theta(a_t|s_t)\|^2$ for each sample.
   - Group data by unique states to avoid redundant calculations.

4. Update policy using the estimated gradient:
   $$\hat{g} = \nabla_\theta \log \pi_\theta(a_t|s_t) (Q^\pi(s_t, a_t) - b^*(s_t))$$

## Derivation of the Optimal Baseline for Off-Policy RL Loss

For off-policy reinforcement learning, samples are drawn from a behavior policy $\pi_{\text{old}}$ while the target policy is $\pi_\theta$. Using the simplified formulation, the off-policy policy gradient becomes

$$
\nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}}\!\left[\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \left( Q^\pi(s,a)-b(s)\right)\right].
$$

### Variance Analysis Using the Simplified Formula

The estimator for a single sample is

$$
\hat{g} = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \left(Q^\pi(s,a)-b(s)\right).
$$

Thus, its squared norm is

$$
\left\|\hat{g}\right\|^2 = \left\|\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\right\|^2 \left(Q^\pi(s,a)-b(s)\right)^2.
$$

To minimize the variance of this estimator, we focus on the expectation

$$
\mathcal{L}(b(s)) = \mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\left\|\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\right\|^2 \left(Q^\pi(s,a)-b(s)\right)^2\right].
$$

Taking the derivative with respect to $b(s)$ and setting it to zero:

$$
\frac{\partial}{\partial b(s)}\,\mathcal{L}(b(s)) = -2\,\mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\left\|\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\right\|^2 \left(Q^\pi(s,a)-b(s)\right)\right] = 0.
$$

Solving for the optimal baseline yields

$$
b^*(s) = \frac{\mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\left\|\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\right\|^2 Q^\pi(s,a)\right]}{\mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\left\|\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\right\|^2\right]}.
$$

This baseline optimally reduces the variance of the off-policy policy gradient estimator while maintaining its unbiasedness.

------------------------------------------------------------

## Practical Implementation of the Off-Policy Baseline

In practice, one can estimate the optimal off-policy baseline from data obtained by following the behavior policy $\pi_{\text{old}}$.

Assume that for each state $s$ in the collected data you store:
- The action $a$,
- The gradient $\nabla_\theta \pi_\theta(a|s)$,
- The old policy probability $\pi_{\text{old}}(a|s)$,
- The estimated action value $Q^\pi(s,a)$.

Then, the empirical estimate for the baseline at state $s$ is

$$
b^*(s) \approx \frac{\sum_{i=1}^{n_s} \left\|\frac{\nabla_\theta \pi_\theta(a_i|s)}{\pi_{\text{old}}(a_i|s)}\right\|^2\, Q^\pi(s,a_i)}{\sum_{i=1}^{n_s} \left\|\frac{\nabla_\theta \pi_\theta(a_i|s)}{\pi_{\text{old}}(a_i|s)}\right\|^2},
$$

where $n_s$ is the number of samples for state $s$.


## Summary

- **Policy Gradient Theorem:** We derived the policy gradient using the likelihood-ratio trick.
- **Policy Gradient with Baseline:** Introducing a baseline $b(s_t)$ reduces the variance of the estimator without bias.
- **Optimal Baseline (On-Policy):** We derived
  $$
  b^*(s_t)=\frac{\mathbb{E}_{a_t \sim \pi_\theta}\!\Big[\|\nabla_\theta \log \pi_\theta(a_t|s_t)\|^2\, Q^\pi(s_t,a_t)\Big]}{\mathbb{E}_{a_t \sim \pi_\theta}\!\Big[\|\nabla_\theta \log \pi_\theta(a_t|s_t)\|^2\Big]}.
  $$
- **Off-Policy Derivation (Simplified Formula):** With samples from $\pi_{\text{old}}$ and using
  $$
  \nabla_\theta J(\theta) = \mathbb{E}_{(s,a) \sim \pi_{\text{old}}}\!\left[\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \left(Q^\pi(s,a)-b(s)\right)\right],
  $$
  the optimal baseline is derived as
  $$
  b^*(s)=\frac{\mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\left\|\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\right\|^2 Q^\pi(s,a)\right]}{\mathbb{E}_{a \sim \pi_{\text{old}}}\!\left[\left\|\frac{\nabla_\theta \pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\right\|^2\right]}.
  $$
- **Practical Implementation:** Both on-policy and off-policy baselines can be efficiently estimated using Monte Carlo samples, with the off-policy estimator accounting for the probability ratio via the simplified formulation.

This approach enables variance reduction for both on- and off-policy gradient estimates, leading to more stable and data-efficient learning.

