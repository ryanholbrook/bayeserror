
<!-- README.md is generated from README.Rmd. Please edit that file -->

# bayeserror

<!-- badges: start -->

<!-- badges: end -->

This package (will eventually) implement several estimators of the
[Bayes error rate](https://en.wikipedia.org/wiki/Bayes_error_rate) of a
data set for a classification task. The Bayes error rate is a measure of
how well the features of the data can discriminate among the classes. It
gives the *best possible* error rate attainable from the data. If the
distributions of the classes overlap, then the Bayes error rate must be
greater than zero.

There are three parametric estimates: 1. Mahalanobis distance 2.
Bhattacharyya distance 3. Chernoff bound

And three non-parametric estimates: 4. Nearest Neighbor 5. Classifier
Ensembles 6. Plurality Error
