from . import SimStage
from .tsne import TSNEStage
from .tsne import TSNESaveEmbedding

import inspect
import logging
import multiprocessing
from collections import Iterable
from types import SimpleNamespace
from time import time

import numpy as np
from sklearn.base import BaseEstimator

import openTSNE
from openTSNE import _tsne
from openTSNE import initialization as initialization_scheme
from openTSNE.affinity import Affinities, PerplexityBasedNN
from openTSNE.quad_tree import QuadTree
from openTSNE import utils

from .bhumap import estimate_negative_gradient_bh, estimate_negative_gradient_elastic

EPSILON = np.finfo(np.float64).eps

log = logging.getLogger(__name__)


def _handle_nice_params(embedding: np.ndarray, optim_params: dict) -> None:
    """Convert the user friendly params into something the optimizer can
    understand."""
    # Handle callbacks
    optim_params["callbacks"] = openTSNE.tsne._check_callbacks(
        optim_params.get("callbacks")
    )
    optim_params["use_callbacks"] = optim_params["callbacks"] is not None

    # Handle negative gradient method
    negative_gradient_method = optim_params.pop("negative_gradient_method")
    if callable(negative_gradient_method):
        negative_gradient_method = negative_gradient_method
    elif negative_gradient_method in {"umap", "UMAP", "bhumap"}:
        negative_gradient_method = bh_umap
    elif negative_gradient_method in {"elastic"}:
        negative_gradient_method = bh_elastic
    elif negative_gradient_method in {"bh", "BH", "barnes-hut"}:
        negative_gradient_method = openTSNE.kl_divergence_bh
    elif negative_gradient_method in {"fft", "FFT", "interpolation"}:
        negative_gradient_method = openTSNE.kl_divergence_fft
    else:
        raise ValueError(
            "Unrecognized gradient method. Please choose one of "
            "the supported methods or provide a valid callback."
        )
    # `gradient_descent` uses the more informative name `objective_function`
    optim_params["objective_function"] = negative_gradient_method

    # Handle number of jobs
    n_jobs = optim_params.get("n_jobs", 1)
    if n_jobs < 0:
        n_cores = multiprocessing.cpu_count()
        # Add negative number of n_jobs to the number of cores, but increment by
        # one because -1 indicates using all cores, -2 all except one, and so on
        n_jobs = n_cores + n_jobs + 1

    # If the number of jobs, after this correction is still <= 0, then the user
    # probably thought they had more cores, so we'll default to 1
    if n_jobs <= 0:
        log.warning(
            "`n_jobs` receieved value %d but only %d cores are available. "
            "Defaulting to single job." % (optim_params["n_jobs"], n_cores)
        )
        n_jobs = 1

    optim_params["n_jobs"] = n_jobs

    # Determine learning rate if requested
    if optim_params.get("learning_rate", "auto") == "auto":
        optim_params["learning_rate"] = max(200, embedding.shape[0] / 12)


def bh_umap(
    embedding,
    P,
    dof,
    bh_params,
    reference_embedding=None,
    eps=0.001,
    should_eval_error=False,
    n_jobs=1,
    **_,
):
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself. We've also got to make sure that the points'
    # interactions don't interfere with each other
    pairwise_normalization = reference_embedding is None
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    tree = QuadTree(reference_embedding)
    sum_Q = estimate_negative_gradient_bh(
        tree,
        embedding,
        gradient,
        **bh_params,
        eps=eps,
        num_threads=n_jobs,
        pairwise_normalization=pairwise_normalization,
    )
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.estimate_positive_gradient_nn(
        P.indices,
        P.indptr,
        P.data,
        embedding,
        reference_embedding,
        gradient,
        dof,
        num_threads=n_jobs,
        should_eval_error=should_eval_error,
    )

    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


def bh_elastic(
    embedding,
    P,
    dof,
    bh_params,
    reference_embedding=None,
    elastic_const=10000,
    eps=1.0,
    should_eval_error=False,
    n_jobs=1,
    **_,
):
    gradient = np.zeros_like(embedding, dtype=np.float64, order="C")

    # In the event that we wish to embed new points into an existing embedding
    # using simple optimization, we compute optimize the new embedding points
    # w.r.t. the existing embedding. Otherwise, we want to optimize the
    # embedding w.r.t. itself. We've also got to make sure that the points'
    # interactions don't interfere with each other
    pairwise_normalization = reference_embedding is None
    if reference_embedding is None:
        reference_embedding = embedding

    # Compute negative gradient
    tree = QuadTree(reference_embedding)
    sum_Q = estimate_negative_gradient_elastic(
        tree,
        embedding,
        gradient,
        **bh_params,
        eps=eps,
        elastic_const=elastic_const,
        num_threads=n_jobs,
        pairwise_normalization=pairwise_normalization,
    )
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_ = _tsne.estimate_positive_gradient_nn(
        P.indices,
        P.indptr,
        P.data,
        embedding,
        reference_embedding,
        gradient,
        dof,
        num_threads=n_jobs,
        should_eval_error=should_eval_error,
    )

    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


class UMAPEmbedding(openTSNE.TSNEEmbedding):
    def __new__(
        cls,
        embedding,
        affinities,
        reference_embedding=None,
        dof=1,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        negative_gradient_method="umap",
        random_state=None,
        optimizer=None,
        **gradient_descent_params,
    ):
        # init_checks.num_samples(embedding.shape[0], affinities.P.shape[0])

        obj = np.asarray(embedding, dtype=np.float64, order="C").view(UMAPEmbedding)

        obj.reference_embedding = reference_embedding
        obj.P = affinities.P
        obj.gradient_descent_params = gradient_descent_params
        obj.affinities = affinities  # type: Affinities
        obj.gradient_descent_params = gradient_descent_params  # type: dict
        obj.gradient_descent_params.update(
            {
                "negative_gradient_method": negative_gradient_method,
                "n_interpolation_points": n_interpolation_points,
                "min_num_intervals": min_num_intervals,
                "ints_in_interval": ints_in_interval,
                "dof": dof,
            }
        )
        obj.random_state = random_state

        if optimizer is None:
            optimizer = gradient_descent()
        elif not isinstance(optimizer, openTSNE.tsne.gradient_descent):
            raise TypeError(
                "`optimizer` must be an instance of `%s`, but got `%s`."
                % (openTSNE.tsne.gradient_descent.__class__.__name__, type(optimizer))
            )
        obj.optimizer = optimizer

        obj.kl_divergence = None

        # Interpolation grid variables
        obj.interp_coeffs = None
        obj.box_x_lower_bounds = None
        obj.box_y_lower_bounds = None

        return obj

    def optimize(
        self,
        n_iter,
        inplace=False,
        propagate_exception=False,
        **gradient_descent_params,
    ):
        """Run optmization on the embedding for a given number of steps.

        Parameters
        ----------
        n_iter: int
            The number of optimization iterations.

        learning_rate: Union[str, float]
            The learning rate for t-SNE optimization. When
            ``learning_rate="auto"`` the appropriate learning rate is selected
            according to max(200, N / 12), as determined in Belkina et al.
            "Automated optimized parameters for t-distributed stochastic
            neighbor embedding improve visualization and analysis of large
            datasets", 2019. Note that this should *not* be used when adding
            samples into existing embeddings, where the learning rate often
            needs to be much lower to obtain convergence.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        negative_gradient_method: str
            Specifies the negative gradient approximation method to use. For
            smaller data sets, the Barnes-Hut approximation is appropriate and
            can be set using one of the following aliases: ``bh``, ``BH`` or
            ``barnes-hut``. For larger data sets, the FFT accelerated
            interpolation method is more appropriate and can be set using one of
            the following aliases: ``fft``, ``FFT`` or ``ìnterpolation``.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        inplace: bool
            Whether or not to create a copy of the embedding or to perform
            updates inplace.

        propagate_exception: bool
            The optimization process can be interrupted using callbacks. This
            flag indicates whether we should propagate that exception or to
            simply stop optimization and return the resulting embedding.

        random_state: Union[int, RandomState]
            The random state parameter follows the convention used in
            scikit-learn. If the value is an int, random_state is the seed used
            by the random number generator. If the value is a RandomState
            instance, then it will be used as the random number generator. If
            the value is None, the random number generator is the RandomState
            instance used by `np.random`.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        callbacks: Callable[[int, float, np.ndarray] -> bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        PartialTSNEEmbedding
            An optimized partial t-SNE embedding.

        Raises
        ------
        OptimizationInterrupt
            If a callback stops the optimization and the ``propagate_exception``
            flag is set, then an exception is raised.

        """
        # Typically we want to return a new embedding and keep the old one intact
        if inplace:
            embedding = self
        else:
            embedding = UMAPEmbedding(
                np.copy(self),
                self.reference_embedding,
                self.P,
                optimizer=self.optimizer.copy(),
                **self.gradient_descent_params,
            )

        # If optimization parameters were passed to this funciton, prefer those
        # over the defaults specified in the TSNE object
        optim_params = dict(self.gradient_descent_params)
        optim_params.update(gradient_descent_params)
        optim_params["n_iter"] = n_iter
        # this calls the function I patched in this file, not the one
        # from openTSNE
        _handle_nice_params(embedding, optim_params)

        try:
            # Run gradient descent with the embedding optimizer so gains are
            # properly updated and kept
            error, embedding = embedding.optimizer(
                embedding=embedding,
                reference_embedding=self.reference_embedding,
                P=self.P,
                **optim_params,
            )

        except openTSNE.OptimizationInterrupt as ex:
            log.info("Optimization was interrupted with callback.")
            if propagate_exception:
                raise ex
            error, embedding = ex.error, ex.final_embedding

        embedding.kl_divergence = error

        return embedding


class UMAPBH(SimStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        learning_rate="umap",
        n_iter_early=250,
        n_iter=500,
        exaggeration=1,
        early_exaggeration=1,
        eps=0.001,
        early_momentum=0.5,
        momentum=0.8,
        n_jobs=-1,
        save_iter_freq=250,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
        )
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_early = n_iter_early
        self.exaggeration = exaggeration
        self.early_exaggeration = early_exaggeration
        self.eps = eps
        self.early_momentum = early_momentum
        self.momentum = momentum

        self.n_jobs = n_jobs
        self.saver = TSNESaveEmbedding(self.outdir)
        self.save_iter_freq = save_iter_freq

    def transform(self):
        affinities = Affinities()
        affinities.P = self.data.tocsr()

        if self.learning_rate == "auto":
            n = self.init.shape[0]
            self.learning_rate = n / self.exaggeration
        elif self.learning_rate == "umap":
            self.learning_rate = 1 / self.exaggeration

        umap = UMAPEmbedding(
            embedding=self.init,
            affinities=affinities,
            negative_gradient_method=bh_umap,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            callbacks_every_iters=self.save_iter_freq,
            callbacks=self.saver,
        )

        self.data_ = umap.optimize(
            n_iter=self.n_iter_early,
            exaggeration=self.early_exaggeration,
            eps=self.eps,
            momentum=self.early_momentum,
            inplace=True,
            n_jobs=self.n_jobs,
            propagate_exception=True,
        )

        self.data_ = umap.optimize(
            n_iter=self.n_iter,
            exaggeration=self.exaggeration,
            eps=self.eps,
            momentum=self.momentum,
            inplace=True,
            n_jobs=self.n_jobs,
            propagate_exception=True,
        )

        return self.data_


class TSNEElastic(TSNEStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        learning_rate="auto",
        n_iter_early=250,
        n_iter=500,
        exaggeration=1,
        early_exaggeration=12,
        elastic_const=10000,
        eps=1,
        early_momentum=0.5,
        momentum=0.8,
        negative_gradient_method=bh_elastic,
        n_jobs=-1,
        save_iter_freq=250,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
            learning_rate=learning_rate,
            n_iter=n_iter,
            exaggeration=exaggeration,
            momentum=momentum,
            negative_gradient_method=negative_gradient_method,
            n_jobs=n_jobs,
            save_iter_freq=save_iter_freq,
        )
        self.early_exaggeration = early_exaggeration
        self.n_iter_early = n_iter_early
        self.early_momentum = early_momentum
        self.elastic_const = elastic_const
        self.eps = eps

    def transform(self):
        affinities = Affinities()
        affinities.P = self.data.tocsr()

        if self.learning_rate == "auto":
            n = self.init.shape[0]
            self.learning_rate = n / self.early_exaggeration

        tsne = UMAPEmbedding(
            embedding=self.init,
            affinities=affinities,
            negative_gradient_method=self.negative_gradient_method,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            callbacks_every_iters=self.save_iter_freq,
            callbacks=self.saver,
        )

        self.data_ = tsne.optimize(
            n_iter=self.n_iter_early,
            exaggeration=self.early_exaggeration,
            momentum=self.early_momentum,
            elastic_const=self.elastic_const,
            eps=self.eps,
            inplace=True,
            n_jobs=self.n_jobs,
            propagate_exception=True,
        )

        self.data_ = tsne.optimize(
            n_iter=self.n_iter,
            exaggeration=self.exaggeration,
            momentum=self.momentum,
            elastic_const=self.elastic_const,
            eps=self.eps,
            inplace=True,
            n_jobs=self.n_jobs,
            propagate_exception=True,
        )

        return self.data_


class gradient_descent(openTSNE.tsne.gradient_descent):
    def __call__(
        self,
        embedding,
        P,
        n_iter,
        objective_function,
        learning_rate=200,
        momentum=0.5,
        exaggeration=None,
        dof=1,
        min_gain=0.01,
        max_grad_norm=None,
        max_step_norm=5,
        eps=0.001,
        elastic_const=10000,
        theta=0.5,
        n_interpolation_points=3,
        min_num_intervals=50,
        ints_in_interval=1,
        reference_embedding=None,
        n_jobs=1,
        use_callbacks=False,
        callbacks=None,
        callbacks_every_iters=50,
        verbose=False,
        **kwargs,
    ):
        """Perform batch gradient descent with momentum and gains.

        Parameters
        ----------
        embedding: np.ndarray
            The embedding :math:`Y`.

        P: array_like
            Joint probability matrix :math:`P`.

        n_iter: int
            The number of iterations to run for.

        objective_function: Callable[..., Tuple[float, np.ndarray]]
            A callable that evaluates the error and gradient for the current
            embedding.

        learning_rate: Union[str, float]
            The learning rate for t-SNE optimization. When
            ``learning_rate="auto"`` the appropriate learning rate is selected
            according to max(200, N / 12), as determined in Belkina et al.
            "Automated optimized parameters for t-distributed stochastic
            neighbor embedding improve visualization and analysis of large
            datasets", 2019.

        momentum: float
            Momentum accounts for gradient directions from previous iterations,
            resulting in faster convergence.

        exaggeration: float
            The exaggeration factor is used to increase the attractive forces of
            nearby points, producing more compact clusters.

        dof: float
            Degrees of freedom of the Student's t-distribution.

        min_gain: float
            Minimum individual gain for each parameter.

        max_grad_norm: float
            Maximum gradient norm. If the norm exceeds this value, it will be
            clipped. This is most beneficial when adding points into an existing
            embedding and the new points overlap with the reference points,
            leading to large gradients. This can make points "shoot off" from
            the embedding, causing the interpolation method to compute a very
            large grid, and leads to worse results.

        max_step_norm: float
            Maximum update norm. If the norm exceeds this value, it will be
            clipped. This prevents points from "shooting off" from
            the embedding.

        theta: float
            This is the trade-off parameter between speed and accuracy of the
            tree approximation method. Typical values range from 0.2 to 0.8. The
            value 0 indicates that no approximation is to be made and produces
            exact results also producing longer runtime.

        n_interpolation_points: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The number of interpolation points to use within each grid
            cell for interpolation based t-SNE. It is highly recommended leaving
            this value at the default 3.

        min_num_intervals: int
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. The minimum number of grid cells to use, regardless of the
            ``ints_in_interval`` parameter. Higher values provide more accurate
            gradient estimations.

        ints_in_interval: float
            Only used when ``negative_gradient_method="fft"`` or its other
            aliases. Indicates how large a grid cell should be e.g. a value of 3
            indicates a grid side length of 3. Lower values provide more
            accurate gradient estimations.

        reference_embedding: np.ndarray
            If we are adding points to an existing embedding, we have to compute
            the gradients and errors w.r.t. the existing embedding.

        n_jobs: int
            The number of threads to use while running t-SNE. This follows the
            scikit-learn convention, ``-1`` meaning all processors, ``-2``
            meaning all but one, etc.

        use_callbacks: bool

        callbacks: Callable[[int, float, np.ndarray] -> bool]
            Callbacks, which will be run every ``callbacks_every_iters``
            iterations.

        callbacks_every_iters: int
            How many iterations should pass between each time the callbacks are
            invoked.

        Returns
        -------
        float
            The KL divergence of the optimized embedding.
        np.ndarray
            The optimized embedding Y.

        Raises
        ------
        OptimizationInterrupt
            If the provided callback interrupts the optimization, this is raised.

        """
        assert isinstance(embedding, np.ndarray), (
            "`embedding` must be an instance of `np.ndarray`. Got `%s` instead"
            % type(embedding)
        )

        if reference_embedding is not None:
            assert isinstance(reference_embedding, np.ndarray), (
                "`reference_embedding` must be an instance of `np.ndarray`. Got "
                "`%s` instead" % type(reference_embedding)
            )

        # If we're running transform and using the interpolation scheme, then we
        # should limit the range where new points can go to
        should_limit_range = False
        if reference_embedding is not None:
            if reference_embedding.box_x_lower_bounds is not None:
                should_limit_range = True
                lower_limit = reference_embedding.box_x_lower_bounds[0]
                upper_limit = reference_embedding.box_x_lower_bounds[-1]

        update = np.zeros_like(embedding)
        if self.gains is None:
            self.gains = np.ones_like(embedding)

        bh_params = {"theta": theta}
        fft_params = {
            "n_interpolation_points": n_interpolation_points,
            "min_num_intervals": min_num_intervals,
            "ints_in_interval": ints_in_interval,
        }

        # Lie about the P values for bigger attraction forces
        if exaggeration is None:
            exaggeration = 1

        if exaggeration != 1:
            P *= exaggeration

        # Notify the callbacks that the optimization is about to start
        if isinstance(callbacks, Iterable):
            for callback in callbacks:
                # Only call function if present on object
                getattr(callback, "optimization_about_to_start", lambda: ...)()

        timer = utils.Timer(
            "Running optimization with exaggeration=%.2f, lr=%.2f for %d iterations..."
            % (exaggeration, learning_rate, n_iter),
            verbose=verbose,
        )
        timer.__enter__()

        if verbose:
            start_time = time()

        for iteration in range(n_iter):
            should_call_callback = (
                use_callbacks and (iteration + 1) % callbacks_every_iters == 0
            )
            # Evaluate error on 50 iterations for logging, or when callbacks
            should_eval_error = should_call_callback or (
                verbose and (iteration + 1) % 50 == 0
            )

            error, gradient = objective_function(
                embedding,
                P,
                dof=dof,
                bh_params=bh_params,
                fft_params=fft_params,
                eps=eps,
                elastic_const=elastic_const,
                reference_embedding=reference_embedding,
                n_jobs=n_jobs,
                should_eval_error=should_eval_error,
                **kwargs,
            )

            # Clip gradients to avoid points shooting off. This can be an issue
            # when applying transform and points are initialized so that the new
            # points overlap with the reference points, leading to large
            # gradients
            if max_grad_norm is not None:
                norm = np.linalg.norm(gradient, axis=1)
                coeff = max_grad_norm / (norm + 1e-6)
                mask = coeff < 1
                gradient[mask] *= coeff[mask, None]

            # Correct the KL divergence w.r.t. the exaggeration if needed
            if should_eval_error and exaggeration != 1:
                error = error / exaggeration - np.log(exaggeration)

            if should_call_callback:
                # Continue only if all the callbacks say so
                should_stop = any(
                    (bool(c(iteration + 1, error, embedding)) for c in callbacks)
                )
                if should_stop:
                    # Make sure to un-exaggerate P so it's not corrupted in future runs
                    if exaggeration != 1:
                        P /= exaggeration
                    raise openTSNE.tsne.OptimizationInterrupt(
                        error=error, final_embedding=embedding
                    )

            # Update the embedding using the gradient
            grad_direction_flipped = np.sign(update) != np.sign(gradient)
            grad_direction_same = np.invert(grad_direction_flipped)
            self.gains[grad_direction_flipped] += 0.2
            self.gains[grad_direction_same] = (
                self.gains[grad_direction_same] * 0.8 + min_gain
            )
            update = momentum * update - learning_rate * self.gains * gradient

            # Clip the update sizes
            if max_step_norm is not None:
                update_norms = np.linalg.norm(update, axis=1, keepdims=True)
                mask = update_norms.squeeze() > max_step_norm
                update[mask] /= update_norms[mask]
                update[mask] *= max_step_norm

            embedding += update

            # Zero-mean the embedding only if we're not adding new data points,
            # otherwise this will reset point positions
            if reference_embedding is None:
                embedding -= np.mean(embedding, axis=0)

            # Limit any new points within the circle defined by the interpolation grid
            if should_limit_range:
                if embedding.shape[1] == 1:
                    mask = (lower_limit < embedding) & (embedding < upper_limit)
                    np.clip(embedding, lower_limit, upper_limit, out=embedding)
                elif embedding.shape[1] == 2:
                    r = np.linalg.norm(embedding, axis=1)
                    phi = np.arctan2(embedding[:, 0], embedding[:, 1])
                    mask = (lower_limit < embedding) & (embedding < upper_limit)
                    mask = np.any(mask, axis=1)
                    np.clip(r, lower_limit, upper_limit, out=r)
                    embedding[:, 0] = r * np.cos(phi)
                    embedding[:, 1] = r * np.sin(phi)
                # Zero out the momentum terms for the points that hit the boundary
                self.gains[~mask] = 0

            if verbose and (iteration + 1) % 50 == 0:
                stop_time = time()
                print(
                    "Iteration %4d, KL divergence %6.4f, 50 iterations in %.4f sec"
                    % (iteration + 1, error, stop_time - start_time)
                )
                start_time = time()

        timer.__exit__()

        # Make sure to un-exaggerate P so it's not corrupted in future runs
        if exaggeration != 1:
            P /= exaggeration

        # The error from the loop is the one for the previous, non-updated
        # embedding. We need to return the error for the actual final embedding, so
        # compute that at the end before returning
        error, _ = objective_function(
            embedding,
            P,
            dof=dof,
            bh_params=bh_params,
            fft_params=fft_params,
            reference_embedding=reference_embedding,
            n_jobs=n_jobs,
            should_eval_error=True,
        )

        return error, embedding
