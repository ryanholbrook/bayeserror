test_that("predict_on_fold returns a good array", {
    set.seed(31415)
    x <- matrix(rnorm(10), 5, 2)
    y <- sample(c(0, 1), 10, replace = TRUE)
    fold <- origami::make_folds(x, origami::folds_vfold, V = 2)[[1]]
    ps <- predict_on_fold(x, y, fold, c(naivebayes_classify, svm_classify))
    checkmate::expect_array(x, d = 3, any.missing = FALSE)
})
