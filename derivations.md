# Derivations for Alternating Least Squares Models

This document provides the mathematical derivations underlying the alternating least squares (ALS) recommendation models implemented throughout the notebooks.

The derivations begin with the standard matrix factorisation framework incorporating user and item biases, before extending the formulation to include feature-informed item embeddings. The resulting update equations form the basis of the optimisation procedures used in subsequent experiments.

The presentation focuses on deriving the closed-form ALS updates from the regularised log-likelihood objective, illustrating how alternating optimisation naturally emerges from the structure of the problem.

---

# Standard Alternating Least Squares

The standard ALS approach learns user and item parameters by iteratively minimising the regularised squared reconstruction error between observed ratings and predicted ratings.

At each step, the algorithm alternates between:
- updating user parameters while holding item parameters fixed,
- updating item parameters while holding user parameters fixed.

This process continues until convergence, yielding optimal biases and latent embeddings for the observed ratings.

The predicted rating is:

$$
\hat r_{mn} = \mathbf{u}_m^T \mathbf{v}_n + b_m^{(u)} + b_n^{(i)}
$$

where:
- $\mathbf{u}_m$ is the latent embedding for user $m$,
- $\mathbf{v}_n$ is the latent embedding for item $n$,
- $b_m^{(u)}$ is the user bias,
- $b_n^{(i)}$ is the item bias.

The regularised log-likelihood function is:

$$
\begin{aligned}
LL
&=
-\frac{\lambda}{2}
\sum_{mn}
\left(
r_{mn}
-
(\mathbf{u}_m^T \mathbf{v}_n + b_m^{(u)} + b_n^{(i)})
\right)^2
\\
&\quad
-
\frac{\tau}{2}
\sum_m
\mathbf{u}_m^T \mathbf{u}_m
-
\frac{\tau}{2}
\sum_n
\mathbf{v}_n^T \mathbf{v}_n
\\
&\quad
-
\frac{\gamma}{2}
\sum_m
b_m^{(u)\,2}
-
\frac{\gamma}{2}
\sum_n
b_n^{(i)\,2}
+
\text{const}
\end{aligned}
$$

---

# Bias Update Derivation

To derive the user bias update, differentiate the likelihood with respect to $b_m^{(u)}$ and set the result equal to zero.

$$
\begin{aligned}
\frac{\partial LL}{\partial b_m^{(u)}}
&=
-\lambda
\sum_n
\left(
r_{mn}
-
(
\mathbf{u}_m^T \mathbf{v}_n
+
b_m^{(u)}
+
b_n^{(i)}
)
\right)
-
\gamma b_m^{(u)}
\\
&=
-\lambda
\sum_n
(
r_{mn}
-
\mathbf{u}_m^T \mathbf{v}_n
-
b_n^{(i)}
)
-
\lambda |\Omega(m)| b_m^{(u)}
-
\gamma b_m^{(u)}
=
0
\end{aligned}
$$

Rearranging gives:

$$
\lambda
\sum_n
\left(
r_{mn}
-
(
\mathbf{u}_m^T \mathbf{v}_n + b_n^{(i)}
)
\right)
=
b_m^{(u)}
(
\lambda |\Omega(m)| + \gamma
)
$$

Hence:

$$
\boxed{
b_m^{(u)}
=
\frac{
\lambda
\sum_{n \in \Omega(m)}
\left(
r_{mn}
-
(
\mathbf{u}_m^T \mathbf{v}_n + b_n^{(i)}
)
\right)
}{
\lambda |\Omega(m)| + \gamma
}
}
$$

In the bias-only formulation, the latent embeddings are removed, giving:

$$
\boxed{
b_m^{(u)}
=
\frac{
\lambda
\sum_{n \in \Omega(m)}
(
r_{mn} - b_n^{(i)}
)
}{
\lambda |\Omega(m)| + \gamma
}
}
$$

The item bias update is derived analogously.

---

# Embedding Update Derivation

To derive the user embedding update, differentiate the likelihood with respect to $\mathbf{u}_m$.

$$
\begin{aligned}
\frac{\partial LL}{\partial \mathbf{u}_m}
&=
-\lambda
\sum_n
-
\left(
r_{mn}
-
(
\mathbf{u}_m^T \mathbf{v}_n
+
b_m^{(u)}
+
b_n^{(i)}
)
\right)
\mathbf{v}_n
-
\tau \mathbf{u}_m
\\
&=
\lambda
\sum_n
(
r_{mn}
-
b_m^{(u)}
-
b_n^{(i)}
)
\mathbf{v}_n
-
\lambda
\sum_n
(
\mathbf{v}_n \mathbf{v}_n^T
)
\mathbf{u}_m
-
\tau \mathbf{u}_m
\\
&=
\lambda
\sum_n
(
r_{mn}
-
b_m^{(u)}
-
b_n^{(i)}
)
\mathbf{v}_n
-
\left(
\lambda
\sum_n
\mathbf{v}_n \mathbf{v}_n^T
+
\tau \mathbf{I}
\right)
\mathbf{u}_m
=
0
\end{aligned}
$$

Rearranging:

$$
\lambda
\sum_n
(
r_{mn}
-
b_m^{(u)}
-
b_n^{(i)}
)
\mathbf{v}_n
=
\left(
\lambda
\sum_n
\mathbf{v}_n \mathbf{v}_n^T
+
\tau \mathbf{I}
\right)
\mathbf{u}_m
$$

Therefore:

$$
\boxed{
\mathbf{u}_m
=
\left(
\lambda
\sum_n
\mathbf{v}_n \mathbf{v}_n^T
+
\tau \mathbf{I}
\right)^{-1}
\left(
\lambda
\sum_n
(
r_{mn}
-
b_m^{(u)}
-
b_n^{(i)}
)
\mathbf{v}_n
\right)
}
$$

The update for $\mathbf{v}_n$ follows symmetrically.

---

# Adding Features

To incorporate side information, item embeddings are regularised toward feature-derived representations.

Define:
- $\mathbf{f}_\iota$ as the embedding of feature $\iota$,
- $F_n$ as the number of features associated with item $n$,
- $\bar{\mathbf f}_n$ as the average feature embedding for item $n$:

$$
\bar{\mathbf f}_n
=
\frac{1}{\sqrt{F_n}}
\sum_\iota
\mathbf f_\iota
$$

The modified likelihood becomes:

$$
\begin{aligned}
LL
&=
-\frac{\lambda}{2}
\sum_{mn}
(
r_{mn}
-
(
\mathbf{u}_m^T \mathbf{v}_n
+
b_m^{(u)}
+
b_n^{(i)}
)
)^2
\\
&\quad
-
\frac{\tau}{2}
\sum_n
\left(
\mathbf v_n
-
\bar{\mathbf f}_n
\right)^T
\left(
\mathbf v_n
-
\bar{\mathbf f}_n
\right)
\\
&\quad
-
\frac{\tau}{2}
\sum_m
\mathbf u_m^T \mathbf u_m
-
\frac{\tau}{2}
\sum_\iota
\mathbf f_\iota^T \mathbf f_\iota
\\
&\quad
-
\frac{\gamma}{2}
\sum_m
b_m^{(u)\,2}
-
\frac{\gamma}{2}
\sum_n
b_n^{(i)\,2}
+
\text{const}
\end{aligned}
$$

---

# Feature-Regularised Item Embedding Update

Differentiating with respect to $\mathbf v_n$:

$$
\begin{aligned}
\frac{\partial LL}{\partial \mathbf v_n}
&=
\lambda
\sum_m
(
r_{mn}
-
b_m^{(u)}
-
b_n^{(i)}
)
\mathbf u_m
\\
&\quad
-
\left(
\lambda
\sum_m
\mathbf u_m \mathbf u_m^T
+
\tau \mathbf I
\right)
\mathbf v_n
+
\tau \bar{\mathbf f}_n
=
0
\end{aligned}
$$

Hence:

$$
\boxed{
\mathbf v_n
=
\left(
\lambda
\sum_m
\mathbf u_m \mathbf u_m^T
+
\tau \mathbf I
\right)^{-1}
\left(
\lambda
\sum_m
(
r_{mn}
-
b_m^{(u)}
-
b_n^{(i)}
)
\mathbf u_m
+
\tau \bar{\mathbf f}_n
\right)
}
$$

---

# Feature Embedding Update

Differentiating with respect to feature embedding $\mathbf f_\iota$:

$$
\begin{aligned}
\frac{\partial LL}{\partial \mathbf f_\iota}
&=
\tau
\sum_n
\frac{1}{\sqrt{F_n}}
\left(
\mathbf v_n
-
\frac{1}{\sqrt{F_n}}
\sum_j
\mathbf f_j
\right)
-
\tau \mathbf f_\iota
=
0
\end{aligned}
$$

Rearranging:

$$
\sum_n
\frac{1}{F_n}
\sum_j
\mathbf f_j
+
\mathbf f_\iota
=
\sum_n
\frac{1}{\sqrt{F_n}}
\mathbf v_n
$$

Define the weighted feature matrix $W$:

$$
W_{n\iota}
=
\begin{cases}
\frac{1}{\sqrt{F_n}}
&
\text{if feature } \iota \text{ belongs to item } n
\\
0
&
\text{otherwise}
\end{cases}
$$

Let:
- $\mathbf F$ denote the feature embedding matrix,
- $\mathbf V$ denote the item embedding matrix.

Then:

$$
(W^T W + I)\mathbf F = W^T \mathbf V
$$

giving the closed-form solution:

$$
\boxed{
\mathbf F
=
(W^T W + I)^{-1} W^T \mathbf V
}
$$

All remaining ALS updates remain unchanged.
