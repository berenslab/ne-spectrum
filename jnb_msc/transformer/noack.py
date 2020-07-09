from . import SimStage
from .tsne import TSNESaveEmbedding, TSNEStage
from .umap_bh import gradient_descent as GDOpt_

import numpy as np

import numba

from pathlib import Path
from collections import Iterable

import openTSNE
from openTSNE import _tsne
from openTSNE import initialization as initialization_scheme
from openTSNE.affinity import Affinities, PerplexityBasedNN
from openTSNE.quad_tree import QuadTree

from .bhnoack import estimate_negative_gradient_bh, estimate_positive_gradient_nn

EPSILON = np.finfo(np.float64).eps


def bh_noack(
    embedding,
    P,
    dof,
    bh_params,
    reference_embedding=None,
    a=1.0,
    r=-1.0,
    elastic_const=1,
    eps=0.0,  # a, r, eps are FA2 default
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
        r=r,
        eps=eps,
        elastic_const=elastic_const,
        num_threads=n_jobs,
        pairwise_normalization=pairwise_normalization,
    )
    del tree

    # Compute positive gradient
    sum_P, kl_divergence_ = estimate_positive_gradient_nn(
        P.indices,
        P.indptr,
        P.data,
        embedding,
        reference_embedding,
        gradient,
        dof,
        eps=eps,
        a=a,
        num_threads=n_jobs,
        should_eval_error=should_eval_error,
    )

    # Computing positive gradients summed up only unnormalized q_ijs, so we
    # have to include normalziation term separately
    if should_eval_error:
        kl_divergence_ += sum_P * np.log(sum_Q + EPSILON)

    return kl_divergence_, gradient


class GDAnneal(GDOpt_):
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

        orig_lr = learning_rate
        m_old = np.zeros_like(embedding)
        v_old = np.zeros_like(embedding)

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

            if False:
                b1 = 0.9
                b2 = 0.999
                _eps = 1e-8

                t = iteration + 1

                m_bias = b1 * m_old + (1 - b1) * gradient
                v_bias = b2 * v_old + (1 - b2) * gradient ** 2
                m = m_bias / (1 - b1 ** t)
                v = v_bias / (1 - b2 ** t)
                update = -learning_rate * m / (v ** 0.5 + _eps)

                m_old = m_bias
                v_old = v_bias
            else:
                # Update the embedding using the gradient
                grad_direction_flipped = np.sign(update) != np.sign(gradient)
                grad_direction_same = np.invert(grad_direction_flipped)
                self.gains[grad_direction_flipped] += 0.2
                self.gains[grad_direction_same] = (
                    self.gains[grad_direction_same] * 0.8 + min_gain
                )
                update = momentum * update - learning_rate * self.gains * gradient

                # momentum gradient with no gains
                # # Update the embedding using the gradient
                # update = momentum * update - learning_rate * gradient
                # learning_rate = orig_lr * (1 - iteration / n_iter)
            # else:
            #     update = momentum * update - learning_rate * gradient

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


class BHARModel(TSNEStage):
    """Implementation of the (a, r)-energy model by Noack.  Using
    Barnes-Hut for optimization in order to speed up the computation."""

    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        learning_rate=0.1,
        n_iter=750,
        a=1.0,
        r=-1.0,
        eps=0.0,
        elastic_const=1.0,
        exaggeration=1,
        momentum=0.5,
        negative_gradient_method=bh_noack,
        max_step_norm=10,  # increase the default step limit
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
        self.a = a
        self.r = r
        self.eps = eps
        self.elastic_const = elastic_const
        self.max_step_norm = max_step_norm

    def transform(self):
        affinities = Affinities()
        affinities.P = self.data.tocsr()

        if self.learning_rate == "auto":
            n = self.init.shape[0]
            self.learning_rate = n / self.early_exaggeration

        emb = openTSNE.TSNEEmbedding(
            embedding=self.init,
            affinities=affinities,
            negative_gradient_method=self.negative_gradient_method,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            max_step_norm=self.max_step_norm,
            random_state=self.random_state,
            callbacks_every_iters=self.save_iter_freq,
            callbacks=self.saver,
            optimizer=GDAnneal(),
        )

        self.data_ = emb.optimize(
            n_iter=self.n_iter,
            exaggeration=self.exaggeration,
            momentum=self.momentum,
            eps=self.eps,
            a=self.a,
            r=self.r,
            elastic_const=self.elastic_const,
            inplace=True,
            n_jobs=self.n_jobs,
            propagate_exception=True,
        )

        return self.data_


class AttractionRepulsionModel(SimStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        knn_indices_name="knn_indices.npy",
        knn_dists_name="knn_dists.npy",
        a=1,
        r=-1,  # FA2 default
        n_iter=250,
        repulsion=1,
        learning_rate=0.1,
        momentum=0.8,
        anneal="linear",
        max_grad_norm=1000,
    ):
        super().__init__(path, dataname, initname, n_components, random_state)
        self.knn_indices_name = knn_indices_name
        self.knn_dists_name = knn_dists_name

        self.a = a
        self.r = r
        self.n_iter = n_iter
        self.repulsion = repulsion
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.anneal = anneal
        self.max_grad_norm = max_grad_norm

        self.saver = TSNESaveEmbedding(self.outdir)
        self.gradsaver = TSNESaveEmbedding(self.outdir / "grad")
        self.lrs = []

        if self.anneal == "linear":
            self.lr_adjust = lambda lr, i, n: lr * (1 - i / n)
        elif self.anneal == "linhalf":
            self.lr_adjust = lambda lr, i, n: lr * (1 - (i / n) / 2)
        else:
            self.lr_adjust = lambda lr, i, n: lr

    def get_datadeps(self):
        return [
            self.indir / self.knn_indices_name,
            self.indir / self.knn_dists_name,
            self.indir / self.initname,
        ]

    def load(self):
        inlist = self.get_datadeps()

        self.knn_indices = np.load(inlist[0])
        self.knn_dists = np.load(inlist[1])
        self.knn_dists[self.knn_dists != 0] = 1
        self.data = self.knn_indices, self.knn_dists
        self.init = np.load(inlist[2])[:, : self.n_components]
        # This is the format that is needed for the numba functions
        self.init = np.array(self.init, copy=False, dtype=np.float32, order="C")

    def transform(self):
        self.data_ = self.init

        import sys

        grads = np.zeros((2, *self.data_.shape))
        self.saver(-1, np.nan, self.data_)  # save the initial layout
        for n in range(self.n_iter):
            grad = grads[n % 2]
            grad_old = grads[(n + 1) % 2]
            grad = energy_gradient(
                grad,
                self.knn_indices,
                self.knn_dists,
                self.data_,
                self.a,
                self.r,
                self.max_grad_norm,
                self.repulsion,
            )
            # print(grad[n % grad.shape[0]], file=sys.stderr)
            print((grad ** 2).mean(), file=sys.stderr)
            self.saver(n, np.nan, self.data_)
            self.gradsaver(n, np.nan, grad)

            self.data_ += self.learning_rate * (grad + self.momentum * grad_old)

            self.lrs.append(self.learning_rate)
            self.learning_rate = self.lr_adjust(self.learning_rate, n, self.n_iter)
        return self.data_

    def save(self):
        super().save()
        self.save_lambda(self.outdir / "lrs.npy", self.lrs, np.save)


@numba.njit(parallel=True)
def energy_gradient(
    grad, knn_indices, knn_dists, layout, a, r, max_grad_norm=1000, repulsion_factor=1
):
    grad[:] = 0
    for i in numba.prange(layout.shape[0]):
        for j in range(layout.shape[0]):
            if i == j:
                continue

            diff = layout[j] - layout[i]
            diffnorm = np.linalg.norm(diff)

            if diffnorm <= 1e-120:
                continue

            # attraction
            if (knn_indices[i] == j).any() or (knn_indices[j] == i).any():
                grad[i] += diff / diffnorm * 1 * diffnorm ** a

            # repulsion
            # knn_dists[i].sum()  # degree of node i
            grad[i] -= (
                knn_dists[i].sum()
                * knn_dists[j].sum()
                * diff
                / diffnorm
                * diffnorm ** r
                * repulsion_factor
            )

            # clip gradients
            grad_i = grad[i]
            grad_i[grad[i] > max_grad_norm] = max_grad_norm
            grad_i[grad[i] < -max_grad_norm] = -max_grad_norm
    return grad
