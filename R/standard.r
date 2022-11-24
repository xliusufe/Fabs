#' Within group standardization
#'
#' @param W A matrix.
#' @param group The grouping vector.
#'
#' @return A list
#' \itemize{
#'   \item xx - The standardized matrix.
#'   \item center - The colmeans of the original matrix.
#'   \item scale - A list whose i^{th} element is the transformation matrix of the i^{th} group.
#' }
#'
#' @examples
#' W = matrix(rnorm(200), 10, 20)
#' group <- rep(c(1:10), each = 2)
#' W_tilde <- standard(W, group)

standard = function(W, group)
{
  p = ncol(W)
  n = nrow(W)
  K1    = cumsum(tabulate(group))
  K0    = c(1, K1[-length(K1)]+1)

  center = colMeans(W)
  W.mean = W - matrix(rep(center, n), n, p, byrow=T)

  scale <- vector("list", length(K0))
  xx = matrix(0, n, p)
  for (i in 1:length(K0)) {
    idx = c(K0[i]:K1[i])
    xx[,idx] = W[,idx]
    scale[[i]] = solve(chol(t(xx[,idx]) %*% xx[,idx]/n))
    xx[,idx] = xx[,idx] %*% scale[[i]]
  }
  list(xx = xx, center = center, scale = scale)
}
