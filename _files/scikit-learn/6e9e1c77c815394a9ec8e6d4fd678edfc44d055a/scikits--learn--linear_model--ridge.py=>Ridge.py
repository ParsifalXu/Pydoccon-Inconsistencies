class Ridge(LinearModel):
    """
    Ridge regression.

    Parameters
    ----------
    alpha : float
        Small positive values of alpha improve the coditioning of the
        problem and reduce the variance of the estimates.

    fit_intercept : boolean
        wether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    Examples
    --------
    >>> from scikits.learn.linear_model import Ridge
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> y = np.random.randn(n_samples)
    >>> X = np.random.randn(n_samples, n_features)
    >>> clf = Ridge(alpha=1.0)
    >>> clf.fit(X, y)
    Ridge(alpha=1.0, fit_intercept=True)
    """

    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y, solver='default', **params):
        """
        Fit Ridge regression model

        Parameters
        ----------
        X : numpy array of shape [n_samples,n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values

        solver : 'default' | 'cg'
            Solver to use in the computational routines. 'default'
            will use the standard scipy.linalg.solve function, 'cg'
            will use the a conjugate gradient solver as found in
            scipy.sparse.linalg.cg.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_params(**params)
        self.solver = solver
        X = safe_asanyarray(X, dtype=np.float)
        y = np.asanyarray(y, dtype=np.float)
        X, y, Xmean, ymean = LinearModel._center_data(X, y, self.fit_intercept)
        if sp.issparse(X):
            self._solve_sparse(X, y)
        else:
            self._solve_dense(X, y)
        self._set_intercept(Xmean, ymean)
        return self

    def _solve_dense(self, X, y):
        n_samples, n_features = X.shape
        if n_samples > n_features:
            A = np.dot(X.T, X)
            A.flat[::n_features + 1] += self.alpha
            self.coef_ = self._solve(A, np.dot(X.T, y))
        else:
            A = np.dot(X, X.T)
            A.flat[::n_samples + 1] += self.alpha
            self.coef_ = np.dot(X.T, self._solve(A, y))

    def _solve_sparse(self, X, y):
        n_samples, n_features = X.shape
        if n_samples > n_features:
            I = sp.lil_matrix((n_features, n_features))
            I.setdiag(np.ones(n_features) * self.alpha)
            self.coef_ = self._solve(X.T * X + I, X.T * y)
        else:
            I = sp.lil_matrix((n_samples, n_samples))
            I.setdiag(np.ones(n_samples) * self.alpha)
            c = self._solve(X * X.T + I, y)
            self.coef_ = X.T * c

    def _solve(self, A, b):
        if self.solver == 'cg':
            sol, error = sp_linalg.cg(A, b)
            if error:
                raise ValueError('Failed with error code %d' % error)
            return sol
        else:
            if sp.issparse(A):
                A = A.todense()
            return linalg.solve(A, b, sym_pos=True, overwrite_a=True)