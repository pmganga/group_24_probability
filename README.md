# Probability Distributions, Bayesian Probability, and Gradient Descent
 
## Group 24 Members
 
- Philip Mganga
- Methode Duhujubumwe
- Nathanaella Hirwa

African Leadership University
BSE - Machine Learning
Mathematics for Machine Learning
 
---
 
## Project links

| What | Link |
|---|---|
| Notebook (Google Colab) | [Open the notebook in Colab](https://colab.research.google.com/drive/1-e_MzHVl_e3LmuHOe1zGwFofLqYXSV6Z?usp=sharing) |
| Team task sheet (live) | [Open the task sheet](https://docs.google.com/spreadsheets/d/12UGM8M3DCWtZxfs28Zg4xkIGg_Qt2Brxz2Dv-NWcc40/edit?usp=sharing) |
| Notebook (in this repo) | [notebooks/Group24_Formative3_Notebook.ipynb](notebooks/Group24_Formative3_Notebook.ipynb) |
| Task sheet (PDF copy) | [docs/Task_Sheet_Formative3_Team24.pdf](docs/Task_Sheet_Formative3_Team24.pdf) |
| Handwritten Part 3 | [Manual_Gradient_Descent.pdf](Manual_Gradient_Descent.pdf) |

## Project Overview
 
This repository contains our submission for Formative Assignment 3, covering probability distributions, Bayesian probability, and gradient descent. The work is organized into four parts:
 
1. Expectation-Maximization (EM) Algorithm
2. Bayesian Probability
3. Manual Gradient Descent
4. Gradient Descent Implementation in Python
Each part builds a different piece of statistical reasoning from first principles, using only NumPy, pandas, and SciPy, with no prebuilt machine learning libraries for the core algorithms.
 
---
 
## Project Structure
 
```
Project/
|
├── README.md
|
├── Codes/
│   └── Formative_3_Probability_Distributions_Bayesian_Probability_and_Gradient_Descent_Implementation.ipynb
|
├── Manual_Gradient_Descent.pdf
|
├── data/
│   ├── GaltonFamilies.csv
│   └── IMDB Dataset.csv
├── BSE Group Assignments _ Task Sheet_[Mathematics for ML_Formative 3_Cohort 2_Team24] - 1.pdf
```
 
Parts 1, 2, and 4 are implemented in a single notebook to keep the statistical narrative and the code that produces it side by side, since each part builds on the same modeling habits (defining a small set of reusable functions, then calling them repeatedly). Part 3 is submitted as a separate handwritten PDF, as required.
 
---
 
## Part 1: Expectation-Maximization (EM) Algorithm
 
### Objective
 
Implement the Expectation-Maximization algorithm from scratch to recover two hidden Gaussian populations from a single pooled, unlabeled set of heights.
 
### Dataset
 
Galton Families dataset (father and child heights). We compared father heights against child heights, treating the two as an unlabeled mixture of two Gaussian distributions.
 
### Approach
 
- Removed duplicate father entries per family, then pooled father heights and child heights into a single unlabeled array of 1,139 values.
- Initialized the two cluster means using the 25th and 75th percentiles of the pooled data, with equal starting variance and equal mixing weights.
- Implemented the E-step, computing each point's responsibility toward each cluster from the Gaussian density and the current mixing weight.
- Implemented the M-step, updating each cluster's mean, variance, and mixing weight as a responsibility-weighted average.
- Tracked the log-likelihood at every iteration and stopped once its change fell below a fixed tolerance.
### Results
 
The algorithm converged after 25 iterations. The tracking table below shows the initialization step and the first two updates.
 
| Iteration | mu1 (Children) | mu2 (Parents) | sigma1^2 | sigma2^2 | pi1 | pi2 | Log-Likelihood |
|---|---|---|---|---|---|---|---|
| 0 | 64.5000 | 70.0000 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | -3119.82 |
| 1 | 65.2205 | 69.2054 | 9.0603 | 8.4468 | 0.5011 | 0.4989 | -3057.91 |
| 2 | 65.1609 | 69.2600 | 8.9420 | 8.1040 | 0.5005 | 0.4995 | -3056.97 |
 
At convergence, EM recovered a child mean of 64.50 and a parent mean of 69.79, compared to the true group means of 66.75 and 69.32. The recovered values are close to, though not identical to, the true group means, which is expected given the real overlap between the two populations.
 
### Classification Demonstration
 
Given a new height, the model reports the exact posterior probability of it belonging to each population. For a test height of 73 inches, the model returned:
 
- P(Child) = 0.0053
- P(Parent) = 0.9947
### Should We Simply Split at the Global Mean?
 
We tested this directly. Splitting the pooled data at its global mean produced 60.8 percent accuracy against the true labels, compared to 60.5 percent for EM's hard assignment. The two methods land close together on this particular dataset, but this is a property of this dataset being reasonably balanced, not evidence that the mean split is a sound method in general. The mean split assumes equal group sizes and equal spread, neither of which holds here, since children outnumber fathers by more than four to one and the two groups have different variances. It also forces a hard decision even for heights that sit in the ambiguous overlap zone, where EM instead reports a genuine, close-to-even probability. EM is the principled choice because it directly maximizes the likelihood of the observed data and accounts for unequal group sizes and unequal spread, rather than relying on a single arbitrary cutoff.
 
---
 
## Part 2: Bayesian Probability
 
### Objective
 
Apply Bayes' Theorem to estimate the probability that a movie review is positive, given the presence of a specific keyword.
 
### Dataset
 
IMDb Movie Reviews dataset, 50,000 reviews, loaded and used without any external machine learning libraries.
 
### Selected Keywords
 
Positive keywords:
- excellent
- amazing
- wonderful
- masterpiece
Negative keywords:
- worst
- terrible
- awful
- boring
These were chosen for being strong, unambiguous sentiment markers that are unlikely to appear in neutral or sarcastic contexts, which keeps the keyword-level probability estimates meaningful.
 
### Conditional Probability Direction
 
We computed P(Positive | keyword) exclusively, as instructed, rather than computing both directions.
 
### Computation
 
For each keyword we computed all four required quantities directly from the data using basic Python and pandas operations:
 
- Prior: P(Positive)
- Likelihood: P(keyword | Positive)
- Marginal: P(keyword)
- Posterior: P(Positive | keyword), via Bayes' Theorem
### Results
 
Prior probability across the dataset: P(Positive) = 0.5000
 
| Keyword | Type | Likelihood P(keyword\|Positive) | Marginal P(keyword) | Posterior P(Positive\|keyword) |
|---|---|---|---|---|
| excellent | Positive | 0.1147 | 0.0710 | 0.8074 |
| amazing | Positive | 0.0672 | 0.0432 | 0.7780 |
| wonderful | Positive | 0.0903 | 0.0556 | 0.8122 |
| masterpiece | Positive | 0.0351 | 0.0241 | 0.7274 |
| worst | Negative | 0.0164 | 0.0887 | 0.0927 |
| terrible | Negative | 0.0153 | 0.0540 | 0.1418 |
| awful | Negative | 0.0114 | 0.0577 | 0.0985 |
| boring | Negative | 0.0236 | 0.0610 | 0.1937 |
 
The posterior probabilities line up with expectation: reviews containing our chosen positive keywords are far more likely to be positive (roughly 73 to 81 percent), while reviews containing our chosen negative keywords are correspondingly unlikely to be positive (roughly 9 to 19 percent).
 
---
 
## Part 3: Manual Gradient Descent
 
### Objective
 
Manually compute gradient descent updates for a linear regression model with two input features, using matrix multiplication rather than treating parameters as scalars.
 
### Contents
 
The full handwritten derivation, including the chain rule applied to both partial derivatives and the intermediate results after each update, is submitted separately as a PDF in this repository, as required. Each group member completed at least one iteration by hand.
 
---
 
## Part 4: Gradient Descent in Python
 
### Objective
 
Convert the manual gradient descent process into working Python code, using SciPy to compute derivatives numerically rather than deriving them by hand in code.
 
### Approach
 
- Defined the linear model as a matrix operation, y_pred = X @ m + b.
- Defined the Mean Squared Error cost function.
- Used SciPy's approx_fprime to compute the gradient of the cost function with respect to m and b at each iteration.
- Updated the parameters using the gradient and a fixed learning rate, printing every intermediate value (predictions, errors, MSE, and gradient) at each step so no calculation is hidden.
### Results and Trend Observed

Across the four manual iterations, the MSE did not decrease — it grew substantially,
With a learning rate of 0.1, the parameter updates grew rapidly rather than settling toward a minimum. The mean squared error rose from 61.0 at initialization to over 21 billion by the fourth iteration. This is a direct consequence of the chosen learning rate being too large for the scale of this data: each update overshoots the minimum by a wider margin than the previous one, so the error compounds instead of shrinking. This illustrates why the learning rate is one of the most sensitive hyperparameters in gradient descent, and why a smaller learning rate, or a normalization step on the input features, would be needed for this particular problem to converge. (In the background, we have found that the learning rate of 0.01 would be better to minimize the error)
 
Two plots are included in the notebook: one tracking m1, m2, and b across iterations, and one tracking the mean squared error across the same iterations. Both visibly show the divergence described above.
 
---
 
## Technologies Used
 
- Python
- NumPy
- pandas
- SciPy
- Matplotlib
- Google Colab
- GitHub
---
 
## Code Design
 
All four parts are built from small, single-purpose functions that are reused rather than rewritten, in line with the DRY principle:
 
- gaussian_pdf, e_step, m_step, and log_likelihood are each defined once and called repeatedly across every EM iteration.
- predict, mse_equation, and compute_gradient_with_scipy are each defined once and reused across every gradient descent iteration.
This keeps each notebook cell focused on a single step of the algorithm, and makes the underlying formulas easy to check against the mathematical derivations they represent.
 
---
 
## How to Run the Project
 
1. Clone the repository.
```
git clone https://github.com/yohan2330/ALU-Formative3
```
 
2. Install the required libraries.
```
pip install numpy pandas scipy matplotlib
```
 
3. Place GaltonFamilies.csv and IMDB Dataset.csv in the data folder, or update the file paths in the notebook to match their location. ( you would want to change some part of the notebook where dataset is loaded to match the correct path of the dataset
4. Open the notebook in Jupyter or Google Colab.
5. Run all cells in order, from top to bottom.
---
 
## Group Contributions
 
| Member | Contribution |
|---|---|
| Philip Mganga | EM Algorithm Implementation, Live Classification Demo, Manual GD Iteration 1 |
| Methode Duhujubumwe | Bayesian Probability, Data Cleaning & Preprocessing, Manual GD Iteration 2 |
| Nathanaella Hirwa | SciPy Gradient Descent Code, Matplotlib Visualizations, Manual GD Iteration 3 |
 
A detailed breakdown of individual contributions is included in the separate contributions PDF submitted alongside this repository. All members contributed to the implementation and are prepared to present and answer questions on every part of the project.
 
---
 
## Conclusion
 
This project applies three related ideas in statistical machine learning to real data. The EM algorithm recovered two overlapping Gaussian populations from a single unlabeled set of heights, and we used our own results to argue, with evidence, against the simpler approach of splitting at the global mean. Bayes' Theorem gave us a principled way to estimate sentiment from individual keywords in real movie reviews. Gradient descent, implemented first by hand and then in code, showed both how parameter updates are computed from the cost function's gradient and what happens when a learning rate is chosen too aggressively for the data at hand.