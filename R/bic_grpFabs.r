#' A Group Forward and Backward Stagewise (GFabs) algorithm for Group penalized problem.
#'
#' @useDynLib Fabs, .registration = TRUE
#' @param W The design matrix.
#' @param y The survival outcome.
#' @param group The grouping vector.
#' @param status The censoring indicator.
#' @param sigma The smoothing parameter in SPR.
#' @param weight The weight vector of groups.
#' @param model The loss function used.
#' @param back The indicator of whether to take backward steps.
#' @param stoping The indicator of whether to stop iteration when lambda is less than lambda.min.
#' @param eps The step size for GFabs.
#' @param xi The threshhold for GFabs.
#' @param iter The maximum number of outer-loop iterations allowed.
#' @param lambda.min The smallest value for lambda, as a fraction of lambda.max.
#'
#' @return A list.
#' \itemize{
#'   \item Beta - The standardized estimation of covariates.
#'   \item beta - The optimal standardized estimation of covariates.
#'   \item lambda - Lambda sequence.
#'   \item direction - Direction of GFabs.
#'   \item active - Active set for each step.
#'   \item iter - Iterations.
#'   \item BIC - The bic for each solution.
#'   \item group - The grouping vector.
#'   \item opt - Position of the optimal tuning based on BIC.
#' }
#' @export
#'
#' @examples
#' sigma = outer(1:20, 1:20, FUN = function(x, y) 0.3^(abs(x - y)))
#' x     = matrix(rnorm(100*20), 100, 20) %*% Matrix::chol(sigma)
#' b     = c(5, -5, 5, -5, rep(0, 16))
#' error = c(0.7*rnorm(100)+0.3*rcauchy(100))
#' y     = x %*% b + error
#' group = seq(1, 20)
#' fit   <- GFabs(x, y, group)

GFabs = function(W, y, group, status=NULL, sigma=NULL, weight=NULL,
                      model=c("spr", "square", "cox", "logistic"), back=TRUE, stoping=TRUE,
                      eps = 0.01, xi = 10^-6, iter=10^4, lambda.min = NULL)
{
  ## Reorder groups, if necessary
  gf    = as.factor(group)
  g     =  as.numeric(gf)
  G     = max(g)
  g.ord = order(g)
  g     = g[g.ord]
  W     = W[,g.ord]
  K     = as.numeric(table(g))
  K1    = cumsum(K)
  K0    = c(1, K1[-G]+1)

  ## standard
  std    = standard(W, group)
  W.std  = std[[1]]
  center = std[[2]]
  scale  = std[[3]]
  y.std  = y - mean(y)

  n = nrow(W.std)
  p = ncol(W.std)
  pg = length(K0)
  param = c(n, p, pg, 0)
  if (is.null(status))         status = rep(1, n)
  if (is.null(sigma))           sigma = 1/sqrt(n)
  if (is.null(lambda.min)) lambda.min = {if (n > p) 1e-4 else .02}
  if (is.null(weight))         weight = sqrt(K)
  
  model = match.arg(model)
  if (model == "cox") {
    y.order = order(y)
    W.std   = W.std[y.order, ]
    y.std   = y[y.order]
    status  = status[y.order]
  }

  #if (type == "L2") type = 2
  type = 2

  VC    = FALSE

  fit <- .Call("BIC_grpFabs",
               as.numeric(t(W.std)),
               as.numeric(y.std),
               as.numeric(weight),
               as.integer(status),
               as.integer(K0-1),
               as.integer(K1-1),
               as.character(model),
               as.numeric(sigma),
               as.numeric(eps),
               as.numeric(lambda.min),
               as.numeric(xi),
               as.integer(type),
               as.integer(back),
               as.integer(stoping),
               as.integer(iter),
               as.integer(param),
               as.integer(VC) )

  beta = matrix(fit$beta, nrow = p)

  # unstandardization
  K1    = cumsum(tabulate(group))
  K0    = c(1, K1[-length(K1)]+1)
  Beta = matrix(0, nrow=nrow(beta), ncol=ncol(beta))
  for (i in 1:length(K0)) {
    idx = c(K0[i]:K1[i])
    Beta[idx,] = scale[[i]] %*% beta[idx,]
  }

  val = list(Beta      = Beta,
             beta      = Beta[,fit$opt],
             lambda    = fit$lambda,
             direction = fit$direction,
             active    = fit$active,
             iter      = fit$iter,
             group     = group,
             BIC       = fit$bic,
             opt       = fit$opt)
  return(val)
}




