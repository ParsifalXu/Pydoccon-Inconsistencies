class _BinaryGaussianProcessClassifierLaplace(BaseEstimator):
    """Binary Gaussian process classification based on Laplace approximation.

    The implementation is based on Algorithm 3.1, 3.2, and 5.1 of
    ``Gaussian Processes for Machine Learning'' (GPML) by Rasmussen and
    Williams.

    Internally, the Laplace approximation is used for approximating the
    non-Gaussian posterior by a Gaussian.

    Currently, the implementation is restricted to using the logistic link
    function.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    optimizer : 'fmin_l_bfgs_b' or callable, default='fmin_l_bfgs_b'
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the  signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'L-BFGS-B' algorithm from scipy.optimize.minimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.

    max_iter_predict : int, default=100
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.

    warm_start : bool, default=False
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization. See :term:`the Glossary
        <warm_start>`.

    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int or RandomState, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term: `Glossary <random_state>`.

    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).

    y_train_ : array-like of shape (n_samples,)
        Target values in training data (also required for prediction)

    classes_ : array-like of shape (n_classes,)
        Unique class labels.

    kernel_ : kernl instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_ : array-like of shape (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in X_train_

    pi_ : array-like of shape (n_samples,)
        The probabilities of the positive class for the training points
        X_train_

    W_sr_ : array-like of shape (n_samples,)
        Square root of W, the Hessian of log-likelihood of the latent function
        values for the observed labels. Since W is diagonal, only the diagonal
        of sqrt(W) is stored.

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    """

    def __init__(self, kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None):
        self.kernel = kernel
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    def fit(self, X, y):
        """Fit Gaussian process classification model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.

        y : array-like of shape (n_samples,)
            Target values, must be binary

        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:
            self.kernel_ = C(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed')
        else:
            self.kernel_ = clone(self.kernel)
        self.rng = check_random_state(self.random_state)
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        label_encoder = LabelEncoder()
        self.y_train_ = label_encoder.fit_transform(y)
        self.classes_ = label_encoder.classes_
        if self.classes_.size > 2:
            raise ValueError('%s supports only binary classification. y contains classes %s' % (self.__class__.__name__, self.classes_))
        elif self.classes_.size == 1:
            raise ValueError('{0:s} requires 2 classes; got {1:d} class'.format(self.__class__.__name__, self.classes_.size))
        if self.optimizer is not None and self.kernel_.n_dims > 0:

            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=False)
                    return (-lml, -grad)
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)
            optima = [self._constrained_optimization(obj_func, self.kernel_.theta, self.kernel_.bounds)]
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError('Multiple optimizer restarts (n_restarts_optimizer>0) requires that all bounds are finite.')
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = np.exp(self.rng.uniform(bounds[:, 0], bounds[:, 1]))
                    optima.append(self._constrained_optimization(obj_func, theta_initial, bounds))
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(self.kernel_.theta)
        K = self.kernel_(self.X_train_)
        _, (self.pi_, self.W_sr_, self.L_, _, _) = self._posterior_mode(K, return_temporaries=True)
        return self

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """
        check_is_fitted(self)
        K_star = self.kernel_(self.X_train_, X)
        f_star = K_star.T.dot(self.y_train_ - self.pi_)
        return np.where(f_star > 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute ``classes_``.
        """
        check_is_fitted(self)
        K_star = self.kernel_(self.X_train_, X)
        f_star = K_star.T.dot(self.y_train_ - self.pi_)
        v = solve(self.L_, self.W_sr_[:, np.newaxis] * K_star)
        var_f_star = self.kernel_.diag(X) - np.einsum('ij,ij->j', v, v)
        alpha = 1 / (2 * var_f_star)
        gamma = LAMBDAS * f_star
        integrals = np.sqrt(np.pi / alpha) * erf(gamma * np.sqrt(alpha / (alpha + LAMBDAS ** 2))) / (2 * np.sqrt(var_f_star * 2 * np.pi))
        pi_star = (COEFS * integrals).sum(axis=0) + 0.5 * COEFS.sum()
        return np.vstack((1 - pi_star, pi_star)).T

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,), default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : ndarray of shape (n_kernel_params,),                 optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when `eval_gradient` is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluated for theta!=None')
            return self.log_marginal_likelihood_value_
        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta
        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)
        Z, (pi, W_sr, L, b, a) = self._posterior_mode(K, return_temporaries=True)
        if not eval_gradient:
            return Z
        d_Z = np.empty(theta.shape[0])
        R = W_sr[:, np.newaxis] * cho_solve((L, True), np.diag(W_sr))
        C = solve(L, W_sr[:, np.newaxis] * K)
        s_2 = -0.5 * (np.diag(K) - np.einsum('ij, ij -> j', C, C)) * (pi * (1 - pi) * (1 - 2 * pi))
        for j in range(d_Z.shape[0]):
            C = K_gradient[:, :, j]
            s_1 = 0.5 * a.T.dot(C).dot(a) - 0.5 * R.T.ravel().dot(C.ravel())
            b = C.dot(self.y_train_ - pi)
            s_3 = b - K.dot(R.dot(b))
            d_Z[j] = s_1 + s_2.T.dot(s_3)
        return (Z, d_Z)

    def _posterior_mode(self, K, return_temporaries=False):
        """Mode-finding for binary Laplace GPC and fixed kernel.

        This approximates the posterior of the latent function values for given
        inputs and target observations with a Gaussian approximation and uses
        Newton's iteration to find the mode of this approximation.
        """
        if self.warm_start and hasattr(self, 'f_cached') and (self.f_cached.shape == self.y_train_.shape):
            f = self.f_cached
        else:
            f = np.zeros_like(self.y_train_, dtype=np.float64)
        log_marginal_likelihood = -np.inf
        for _ in range(self.max_iter_predict):
            pi = expit(f)
            W = pi * (1 - pi)
            W_sr = np.sqrt(W)
            W_sr_K = W_sr[:, np.newaxis] * K
            B = np.eye(W.shape[0]) + W_sr_K * W_sr
            L = cholesky(B, lower=True)
            b = W * f + (self.y_train_ - pi)
            a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
            f = K.dot(a)
            lml = -0.5 * a.T.dot(f) - np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum() - np.log(np.diag(L)).sum()
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml
        self.f_cached = f
        if return_temporaries:
            return (log_marginal_likelihood, (pi, W_sr, L, b, a))
        else:
            return log_marginal_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == 'fmin_l_bfgs_b':
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B', jac=True, bounds=bounds)
            _check_optimize_result('lbfgs', opt_res)
            theta_opt, func_min = (opt_res.x, opt_res.fun)
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError('Unknown optimizer %s.' % self.optimizer)
        return (theta_opt, func_min)