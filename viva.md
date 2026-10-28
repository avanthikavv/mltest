# ML Practical Buddy Notes 🤝

Bro, this is your quick guide for ML practicals and viva. For every parameter, understand **what the value means** first, then remember what changing it does. The ranges below are common values to try, not magic values that work best for every dataset.

---

# 1. First: Classification vs Regression

- **Classification:** predicts a category, like `spam/not spam` or `Class_A/Class_B/Class_C`.
- **Regression:** predicts a number, like price, temperature, or remaining useful life.

Use a classifier when `y` contains classes. Use a regressor when `y` is a continuous numerical value.

# 2. Decision Tree Classifier 🌳

A Decision Tree asks a series of questions about the features. Each answer sends a row down a branch until it reaches a predicted class.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `criterion` | The rule used to judge how well a split separates classes. | `"gini"`, `"entropy"`, `"log_loss"` |
| `max_depth` | How many levels deep the tree is allowed to grow. `None` means no preset depth limit. | `None`, `1, 2, 3, ...`; commonly try `3–20` |
| `min_samples_split` | How many rows a node must contain before the tree is allowed to split it. | Integer `>= 2`, or fraction `(0, 1]`; e.g. `2, 5, 10` or `0.1` |
| `min_samples_leaf` | The minimum number of training rows that must end up in a leaf. | Integer `>= 1`, or fraction `(0, 0.5]`; e.g. `1, 2, 4` |
| `max_features` | How many features the tree checks when looking for a split. | `None`, `"sqrt"`, `"log2"`, integer, or fraction |
| `max_leaf_nodes` | The maximum number of final leaf nodes the tree may have. | `None` or integer `>= 2` |
| `class_weight` | A weight assigned to each class during training. `"balanced"` gives more weight to less frequent classes. | `None`, `"balanced"`, or dictionary |
| `random_state` | The seed used for randomized choices, so results can be repeated. | Any integer, e.g. `42`, or `None` |

**How to think about it:** A very deep tree with tiny leaves can memorize the training data. Try limiting `max_depth` or increasing `min_samples_leaf` if that happens.

# 3. Decision Tree Regressor

Same tree idea, but the answer at the leaf is a number instead of a class.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `criterion` | The error rule used to judge regression splits. | `"squared_error"`, `"friedman_mse"`, `"absolute_error"`, `"poisson"` (some criteria have data requirements) |
| `max_depth` | Maximum number of levels in the tree. | `None` or positive integer |
| `min_samples_split` | Minimum rows required before a node can split. | Integer `>= 2` or fraction `(0, 1]` |
| `min_samples_leaf` | Minimum rows allowed in a leaf. | Integer `>= 1` or fraction `(0, 0.5]` |
| `max_features` | Number of features considered for a split. | `None`, `"sqrt"`, `"log2"`, integer, or fraction |

# 4. Random Forest 🌲🌲🌲

Random Forest builds lots of decision trees and combines their answers. For classification it votes; for regression it averages.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `n_estimators` | Number of trees in the forest. | Positive integer; try `100, 200, 300` |
| `max_depth` | Depth limit for each tree. | `None` or positive integer |
| `min_samples_split` | Minimum rows needed to split a node. | Integer `>= 2` or fraction `(0, 1]` |
| `min_samples_leaf` | Minimum rows in a leaf. | Integer `>= 1` or fraction `(0, 0.5]` |
| `max_features` | Number of features each tree considers at a split. | Classifier default is commonly `"sqrt"`; also `None`, `"log2"`, integer, or fraction |
| `bootstrap` | Whether each tree trains on a random sample drawn with replacement. Some rows can appear more than once in a tree's sample. | `True` or `False` |
| `class_weight` | Class weighting during classification. | `None`, `"balanced"`, `"balanced_subsample"`, or dictionary |

**Viva line:** Random Forest is an ensemble of decision trees. It uses bagging and random feature selection to make the trees less alike.

# 5. Gradient Boosting Classifier / Regressor 🚀

Gradient Boosting adds small trees one after another. Each new tree helps improve the predictions made by the ensemble so far.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `n_estimators` | Number of boosting stages/trees added sequentially. | Positive integer; commonly try `50, 100, 200, 500` |
| `learning_rate` | How much each new tree changes the ensemble's prediction. | Positive float; commonly try `0.01, 0.05, 0.1, 0.2` |
| `max_depth` | How deep each individual tree can grow. | Non-negative integer; commonly try `1–5` |
| `subsample` | Fraction of training rows randomly used for each boosting stage. `0.8` means 80% of rows are sampled for a stage. | `(0, 1]`; try `0.5, 0.7, 0.8, 1.0` |
| `min_samples_split` | Minimum rows needed before an individual tree node can split. | Integer `>= 2` or fraction `(0, 1]` |
| `min_samples_leaf` | Minimum rows that must stay in a leaf. | Integer `>= 1` or fraction `(0, 0.5]` |
| `max_features` | Number of features checked for each tree split. | `None`, `"sqrt"`, `"log2"`, integer, or fraction |
| `loss` | The objective the boosting model tries to reduce. Available choices depend on classifier vs regressor. | Check the estimator's `loss` options for your scikit-learn version |

**Easy memory trick:** `learning_rate` is the size of each step; `n_estimators` is how many steps the model takes. A smaller learning rate often needs more estimators. `subsample=1.0` uses all training rows at each stage; values below `1.0` add randomness.

# 6. K-Nearest Neighbors (KNN) 📍

KNN looks at the closest training examples. For classification it takes a vote; for regression it combines their target values.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `n_neighbors` | Number of nearby rows used for a prediction. | Positive integer; try `3, 5, 7, 9, 11` |
| `weights` | Whether all neighbors have equal influence or closer neighbors count more. | `"uniform"`, `"distance"` |
| `metric` | The distance formula used to decide which rows are closest. | Commonly `"minkowski"`, `"euclidean"`, `"manhattan"` |
| `p` | The distance power when using Minkowski distance. | `1` = Manhattan; `2` = Euclidean |

**Important:** Scaling usually matters for KNN. A feature measured in thousands can otherwise dominate one measured between 0 and 1.

# 7. Logistic Regression

Despite its name, Logistic Regression is a **classification** algorithm. It estimates class probabilities and uses them to predict classes.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `C` | Inverse regularization strength. Smaller `C` means stronger regularization; larger `C` means weaker regularization. | Positive float; try `0.01, 0.1, 1, 10, 100` |
| `penalty` | The type of coefficient penalty used. L1 can set some coefficients to zero; L2 shrinks coefficients. | `"l1"`, `"l2"`, `"elasticnet"`, or `None`, depending on solver/version |
| `solver` | The numerical method used to find the model's coefficients. | Examples: `"lbfgs"`, `"liblinear"`, `"saga"` |
| `max_iter` | Maximum number of optimization steps. | Positive integer; try `100, 500, 1000, 2000` |
| `class_weight` | Changes how much each class contributes to training. | `None`, `"balanced"`, or dictionary |

Not every `penalty` works with every `solver`. Scaling is often helpful.

# 8. Naive Bayes 🧮

Naive Bayes uses Bayes' theorem to estimate which class is most likely. Its simplifying assumption is that features are conditionally independent given the class.

- **`GaussianNB`:** for continuous numeric features; models each feature within each class with a Gaussian distribution.
- **`MultinomialNB`:** commonly used for non-negative counts, such as word counts in text.
- **`BernoulliNB`:** for binary features, such as whether a word appears or does not appear.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `var_smoothing` (`GaussianNB`) | Adds a small amount to feature variances for numerical stability. | Small positive floats, e.g. `1e-9`, `1e-8`, `1e-7` |
| `alpha` (`MultinomialNB`, `BernoulliNB`) | Smoothing amount so unseen feature/class combinations do not get zero probability. | Non-negative float; commonly try `0.01, 0.1, 1, 10` |
| `binarize` (`BernoulliNB`) | Threshold above which a numeric feature is treated as 1; below it becomes 0. | Float or `None` |

MultinomialNB expects non-negative inputs. GaussianNB generally needs dense data, not a sparse one-hot matrix.

# 9. Support Vector Machine (SVC / SVR)

- **SVC** predicts classes.
- **SVR** predicts numbers.
- An SVM tries to find a boundary/function with a margin; kernels can represent nonlinear patterns.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `C` | How strongly the model penalizes training errors. Large `C` tries harder to fit training examples; small `C` allows more regularization. | Positive float; try `0.1, 1, 10, 100` |
| `kernel` | The kind of boundary/similarity function used. | `"linear"`, `"poly"`, `"rbf"`, `"sigmoid"` |
| `gamma` | How far the influence of a training point reaches for kernels such as RBF. Large gamma creates more local influence. | `"scale"`, `"auto"`, or positive float |
| `degree` | Polynomial degree, used only with `kernel="poly"`. | Integer; commonly `2, 3, 4, 5` |
| `epsilon` (`SVR`) | Width of the zone where prediction errors are not penalized by the epsilon-insensitive loss. | Non-negative float; try `0.01, 0.1, 0.2, 0.5` |

Scaling is usually important for SVC/SVR.

# 10. Linear, Ridge, and Lasso Regression

- **Linear Regression:** fits a linear relationship between features and a numerical target.
- **Ridge:** adds an L2 penalty, which shrinks coefficients.
- **Lasso:** adds an L1 penalty, which can shrink some coefficients all the way to zero.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `fit_intercept` | Whether the model estimates a constant offset/intercept. | `True` or `False` |
| `alpha` (Ridge/Lasso) | Strength of the regularization penalty. | Non-negative; commonly try `0.001, 0.01, 0.1, 1, 10, 100` |
| `max_iter` (Lasso) | Maximum optimization iterations. | Positive integer; increase if convergence warnings occur |

# 11. PCA (Principal Component Analysis)

PCA makes new features called components. The first component captures as much variance as possible; later components capture as much remaining variance as possible while staying orthogonal to earlier ones.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `n_components` | Number of components to keep, or (with supported solvers) a variance fraction such as 95%. | Integer `1` to number of available features; or float in `(0, 1)` for variance-based selection |
| `svd_solver` | The algorithm used to calculate components. | `"auto"`, `"full"`, `"arpack"`, `"randomized"` |
| `random_state` | Seed for solvers that use randomness. | Integer or `None` |

Scale features first when their units/ranges differ significantly. Fit PCA on training data only; use that fitted PCA to transform test data.

# 12. Train-Test Split and Cross-Validation

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `test_size` | How much data is held out for testing. `0.2` means 20%. | Float between 0 and 1, or a sample count |
| `random_state` | Seed so the same random split can be reproduced. | Integer such as `42`, or `None` |
| `stratify=y` | Tries to keep class proportions similar in train and test sets. | Target labels or `None` |
| `n_splits` | Number of folds used in cross-validation. | Integer `>= 2`; commonly `5` |
| `shuffle` | Whether data is shuffled before splitting into folds. | `True` or `False` |

**Gotcha:** `stratify=y` fails when a class has too few samples. For `StratifiedKFold(n_splits=5)`, every class generally needs at least 5 samples.

# 13. GridSearchCV and RandomizedSearchCV 🔎

They try different hyperparameter settings and use cross-validation to find a promising combination.

| Parameter | What does it mean? | Values to know |
|---|---|---|
| `param_grid` | Dictionary of parameter names and candidate lists for GridSearchCV. | Example: `{"clf__max_depth": [3, 5, None]}` |
| `param_distributions` | Candidate values or distributions for RandomizedSearchCV. | Dictionary |
| `cv` | The cross-validation strategy or number of folds. | Commonly `5` or a splitter such as `StratifiedKFold` |
| `scoring` | The score used to compare candidates. | Classification: `"accuracy"`, `"f1"`, `"roc_auc"`; regression: `"neg_mean_absolute_error"`, `"neg_root_mean_squared_error"`, `"r2"` |
| `n_iter` | Number of combinations randomly sampled by RandomizedSearchCV. | Positive integer |
| `n_jobs` | Number of parallel jobs. `-1` asks for all available CPU cores. | `None`, `-1`, or integer |

If the model is inside a Pipeline step named `clf`, parameters need the prefix `clf__`, such as `clf__max_depth`.

# 14. Bagging, Boosting, Random Forest, and Stacking

## Bagging — models work separately

Bagging means **Bootstrap Aggregating**. It makes different training samples by sampling rows with replacement, trains multiple models independently (so they can often train in parallel), and combines their predictions.

Think: **many models trained separately, then vote/average.**

- Main idea: often reduces variance and makes predictions more stable.
- Example: `BaggingClassifier`, `BaggingRegressor`.

## Random Forest — bagging with trees

Random Forest is a bagging ensemble made of decision trees. Each tree sees a bootstrap sample and considers a random subset of features at each split.

Think: **a crowd of slightly different trees voting together.**

## Boosting — models work one after another

Boosting adds models sequentially. Later models try to improve what the earlier ensemble is getting wrong.

Think: **each new model tries to fix/improve the current result.**

- Main idea: often reduces bias and improves predictive performance.
- Can overfit if the model is too complex or poorly tuned.

## AdaBoost

In the classic AdaBoost approach, training examples misclassified by earlier learners receive more attention in later rounds. It often uses shallow decision trees as weak learners.

## Gradient Boosting

Gradient Boosting adds trees one by one to reduce a chosen loss function. The gradients of that loss guide what the next tree learns.

- `learning_rate`: size of each update.
- `n_estimators`: number of updates/trees.
- `subsample`: fraction of rows used at each stage.

## Stacking

Stacking trains several different base models, then trains a final model (meta-model) to combine their predictions. Cross-validated/out-of-fold predictions should be used to train the meta-model to avoid leakage.

## XGBoost

XGBoost is a fast, regularized implementation of gradient-boosted trees. It is often strong on tabular data but requires the `xgboost` package and has additional tuning parameters.

# 15. Evaluation Metrics

## Classification
- **Accuracy:** share of all predictions that are correct.
- **Precision:** among predictions called positive, how many really are positive.
- **Recall:** among actual positives, how many the model finds.
- **F1-score:** balances precision and recall.
- **Confusion matrix:** table of correct and incorrect predictions for each class.
- **ROC curve:** plots true-positive rate against false-positive rate at different thresholds.
- **ROC-AUC:** summarizes how well the model ranks positive examples above negative ones.

## Regression
- **MAE:** average absolute difference between actual and predicted values.
- **MSE:** average squared difference; large errors count extra heavily.
- **RMSE:** square root of MSE, expressed in the target's units.
- **R²:** compares the model with predicting the target mean; it can be negative.
- **MAPE:** average percentage error; problematic when actual values are zero or close to zero.

# 16. Quick Viva Questions

**Q: Classification vs regression?**  
Classification predicts categories; regression predicts numerical values.

**Q: What is overfitting?**  
The model learns the training data too closely, including noise, and performs worse on unseen data.

**Q: What is underfitting?**  
The model is too simple to learn the important patterns, so it performs poorly even on training data.

**Q: Bagging vs boosting?**  
Bagging trains models independently on varied samples and combines them. Boosting builds models sequentially, with later models improving the current ensemble.

**Q: Random Forest vs Decision Tree?**  
A Decision Tree is one tree. Random Forest combines many randomized trees, usually making predictions more stable.

**Q: Why use scaling?**  
Features measured on very different scales can dominate distance- or margin-based models such as KNN and SVM.

**Q: Why use PCA?**  
To reduce feature dimensions by representing the data with fewer components that preserve as much variance as possible.

**Q: Why use cross-validation?**  
To estimate how well a model performs across different train/validation splits instead of trusting one split alone.

**Q: Why use GridSearchCV?**  
To compare specified hyperparameter combinations with cross-validation and select the best-scoring combination.

**Q: What is data leakage?**  
When information that should not be available during training (often test data or future information) influences model training or selection.

**Q: Why use a Pipeline?**  
It bundles preprocessing and the model together so each CV fold learns preprocessing only from its training portion.

**Q: What does `subsample=0.8` mean in Gradient Boosting?**  
Each boosting stage uses a randomly selected 80% of the training rows.

**Q: Does a high training score mean the model is good?**  
Not necessarily. Check validation/test performance to see how well it generalizes.

---

# Last-minute practical checklist ✅

1. Set the correct target column.
2. Check missing values, data types, duplicates, target class counts, and outliers.
3. Decide whether the target requires classification or regression.
4. Split data before learning preprocessing transformations.
5. Use a Pipeline for preprocessing plus model when using cross-validation.
6. For classification, make sure each class has enough samples for stratified splitting/folds.
7. Tune using training/CV results; keep the test set for final evaluation.
8. If a parameter errors, inspect `model.get_params()` and your installed scikit-learn version.
