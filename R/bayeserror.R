## bayeserror.R

randomForest_classify <- function(x, y, newdata) {
    fit <- randomForest::randomForest(x = x, y = factor(y))
    preds <- predict(fit, newdata = newdata, type = "prob")
    return(preds)
}
attr(randomForest_classify, "name") <- "randomForest"
attr(randomForest_classify, "shortname") <- "RF"

naivebayes_classify <- function(x, y, newdata) {
    fit <- naivebayes::naive_bayes(x = x, y = factor(y), )
    preds <- predict(fit, newdata = newdata, type = "prob")
    return(preds)
}
attr(naivebayes_classify, "name") <- "naiveBayes"
attr(naivebayes_classify, "shortname") <- "NB"

svm_classify <- function(x, y, newdata) {
    fit <- e1071::svm(x = x, y = factor(y), probability = TRUE)
    preds <- attr(predict(fit, newdata = newdata, probability = TRUE),
                  "probabilities")
    return(preds)
}
attr(svm_classify, "name") <- "svm"
attr(svm_classify, "shortname") <- "SVM"

#' Generate predictions on a cross-validation fold.
#'
#' @return an array of class probability predictions
#'
#' @export
#' @keywords internal
predict_on_fold <- function(x, y, fold, classifiers) {
    checkmate::assert_matrix(x,
                             mode = "numeric",
                             any.missing = FALSE)
    checkmate::assert_integerish(y,
                                 lower = 0L, upper = 1L,
                                 any.missing = FALSE,
                                 len = nrow(x))
    checkmate::assert_list(fold)
    checkmate::assert_list(classifiers,
                           type = c("character", "function"))
    ## Get training and prediction folds.
    x_train <- origami::training(x = x, fold = fold)
    x_val <- origami::validation(x = x, fold = fold)
    y_train <- origami::training(x = y, fold = fold)
    y_val <- origami::validation(x = y, fold = fold)
    ## Make predictions on prediction set, returning a 3-dimensional
    ## array of probabilities: observation x class x classifier
    ps <- abind::abind(lapply(classifiers,
                              function(classify)
                                  classify(x_train, y_train, x_val)),
                       along = 3L)
    checkmate::assert_array(ps,
                            mode = "numeric",
                            d = 3L,
                            any.missing = FALSE)
    ## Prediction probabilities for ensemble classifier. Compute by
    ## averaging class predictions across the classifiers.
    p_ave <- apply(ps,
                   MARGIN = c(1L, 2L),
                   FUN = mean)
    ## Collect predictions
    classifier_names <- lapply(classifiers,
                               function(classify)
                                   attr(classify, "shortname"))
    ps <- abind::abind(ps, p_ave,
                       along = 3L,
                       new.names = list(NULL, NULL,
                                        c(classifier_names, "AVE")))
    attr(ps, "index") <- fold$validation
    return(ps)
}

#' Merge folded data
#'
#' @param folded_data a list of arrays containing the results of a
#'     computation on folded data, one array for each fold.
#'
#' @export
#' @keywords internal
merge_folds <- function(folded_data) {
    checkmate::assert_list(folded_data, types = "array")
    lapply(folded_data,
           function(fold)
               assertthat::assert_that(assertthat::has_attr(fold, "index"),
                                       msg = "Fold is missing index."))
    merged_array <- array(data = NA,
                          dim = attr(folded_data, "ds"))
    for(arr in folded_data) {
        index <- as.integer(attr(arr, "index"))
        merged_array[index, , ] <- arr
    }
    dimnames(merged_array) <- attr(folded_data, "dnames")
    ## checkmate
    return(merged_array)
}

#' Generate class prediction probabilities
#'
#' This function will generate class prediction probabilities on a
#' dataset from a list of classifiers using a specified resampling
#' method.
#' 
#' @param x A matrix with observations as rows and features as
#'     columns.
#'
#' @param y A vector of 0/1 response classes
#'
#' @param classifiers A list of classifier functions. Each function
#'     should be of the form `f(x, y, newdata)` and have the attribute
#'     `shortname` to identify its predictions.
#'
#' @param v A number specifying the resampling method to use when
#'     makining predictions.
#'
#' @return A three-dimensional array containing class prediction
#'     probabilities. Its dimesions are as (observation, class,
#'     classifier). The last entry on the third dimension is `"AVE"`,
#'     the predictions for the ensemble classifier.
#'
#' @param v For k-fold cross-validation, use an integer to specify the
#'     number of folds. For instance, `v = 10` means to use 10-fold
#'     CV. For a holdout split, use a number between 0.0 to 1.0
#'     specifying the percent to be held out to make predictions. For
#'     in-sample predictions, use `v = 0`.
#' @export
predict_ensemble <- function(x, y, classifiers, v = 10L) {
    checkmate::assert_matrix(x)
    checkmate::assert_numeric(y)
    checkmate::assert_list(classifiers)
    ## checkmate
    folds <- origami::make_folds(x,
                                 ## 10-fold CV
                                 fold_fun = origami::folds_vfold, V = v,
                                 ## Balance samples across classes
                                 strata_ids = y)
    folded_data <- lapply(folds,
                          function(fold)
                              predict_on_fold(x, y, fold, classifiers))
    attr(folded_data, "ds") <- c(nrow(x),
                                 2, # binary classification
                                 length(classifiers) + 1)
    attr(folded_data, "dnames") <- list(NULL, NULL,
                                        append(lapply(classifiers,
                                                      function(c)
                                                          attr(c, "shortname")),
                                               "AVE"))
    merged_data <- merge_folds(folded_data)
    ## checkmate
    return(merged_data)
}

#' Compute classifier error from a matrix of class predictions.
ensemble_errors <- function(predictions, true_classes, method = "accuracy") {
    checkmate::assert_matrix(predictions, mode = "integerish")
    checkmate::assert_integerish(predictions,
                                 lower = 0L, upper = 1L)
    checkmate::assert_integerish(true_classes,
                                 lower = 0L, upper = 1L)
    loss <- function(y_pred) {
        MLmetrics::Accuracy(y_pred, true_classes)
    }
    err <- apply(X = predictions,
                 MARGIN = 2,
                 FUN = loss)
    ## checkmate
    return(err)
}

#' Dual Total Correlation for Binary Data
#'
#' @param X a matrix whose columns are 0/1 class vectors
dual_total_correlation <- function(X, normalized = FALSE) {
    checkmate::assert_matrix(X, mode = "integerish")
    entropy <- infotheo::entropy(X)
    cond_entropy <-
        lapply(seq_len(ncol(X)),
               function(i)
                   infotheo::condentropy(X=X[, i], Y=X[, -i]))
    cond_entropy <- unlist(cond_entropy)
    sum_cond_entropy <- sum(cond_entropy)
    dtc <- entropy - sum_cond_entropy
    result <- ifelse(normalized,
                     dtc / entropy,
                     dtc)
#    checkmate::assert_number(result, lower = 0.0, upper = 1.0)
    return(result)
}

average_mutual_information <- function(X, ave) {
    checkmate::assert_matrix(X, mode = "integerish")
    checkmate::assert_integerish(ave)
    mi <- apply(X = X,
                MARGIN = 2,
                FUN = function (x) infotheo::mutinformation(x, ave))
    ent <- infotheo::entropy(X)
    sum_mi <- sum(mi)
    ami <- sum_mi / ent
#    checkmate::assert_number(ami, lower = 0.0, upper = 1.0)
    return(ami)
}

#' Get mutual-information based correlated-error for predicted classes.
#'
#' @param total matrix of base classifier 0/1 predictions
#' @param ave vector of ensemble 0/1 predictions
#' @param method the method to use to compute the correlation
error_correlation <- function(total, ave, method = "ami") {
    checkmate::assert_matrix(total, mode = "numeric")
    checkmate::assert_numeric(ave)
    # TODO - if not integer, discretize
    X <- as.matrix(infotheo::discretize(total))
    ave <- as.matrix(infotheo::discretize(ave))
    delta <- dual_total_correlation(X, normalized = TRUE)
#    checkmate::assert_number(delta, lower = 0.0, upper = 1.0)
    return(delta)
}

bayeserror_formula <- function(N, e_total, e_ave, delta) {
    checkmate::assert_int(N, lower = 2L)
    checkmate::assert_number(e_total, lower = 0.0, upper = 1.0)
    checkmate::assert_number(e_ave, lower = 0.0, upper = 1.0)
#    checkmate::assert_number(delta, lower = 0.0, upper = 1.0)
    be <- ((N * e_ave - ((N - 1L) * delta + 1L) * e_total) /
           ((N - 1L) * (1L - delta)))
    return(be)
}

#' Estimate the Bayes error for a data set using a list of classifiers.
#'
#' @param x a matrix of features
#' @param y a vector of the observed classes
#' @param classifiers a list of classifier functions or else their names
#' @param err_method classification loss method
#' @param corr_method correlation of error method
#' @param v the resampling method
bayeserror <- function(x,
                       y,
                       classifiers,
                       err_method="accuracy",
                       corr_method="ami",
                       v = 10L) {
    ## Check input.
    checkmate::assert_matrix(x,
                             mode = "numeric")
    checkmate::assert_integerish(y,
                                 lower = 0L, upper = 1L,
                                 any.missing = FALSE,
                                 len = nrow(x))
    checkmate::assert_list(classifiers,
                           types = c("character", "function"))

    ## Posterior class probabilities.
    pred_prob <- predict_ensemble(x, y, v = v, classifiers = classifiers)
    ## Convert to 0/1 class predictions.
    pred_class <- round(pred_prob[, 1, ]) # array with dims (obs, classifier)
    ## Compute errors of the predictions
    errs <- ensemble_errors(pred_class, y, err_method)
    ## Form bayes error estimator terms.
    ## Number of base classifiers
    N <- length(classifiers)
    ## Class predictions of base classifiers
    class_total <- pred_class[, 1:N]
    ## Class predictions of ensemble classifier
    class_ave <- pred_class[, N + 1]
    ## Mean error of base classifiers
    e_total <- mean(head(errs, -1))
    ## Error of ensemble classifier
    e_ave <- tail(errs, 1)
    names(e_ave) <- NULL
    ## Estimated error correlation
    delta <- error_correlation(class_total, class_ave, corr_method)
    e_bayes <- bayeserror_formula(N, e_total, e_ave, delta)
    return(list("errors" = errs, "e_total" = e_total,
                "e_ave" = e_ave, "delta" = delta,
                "e_bayes" = e_bayes))
}
