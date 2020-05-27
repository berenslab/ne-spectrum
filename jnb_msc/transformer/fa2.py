from .transformer import SimStage

import random
import numpy as np
import scipy.sparse

import fa2
import fa2.fa2util as fa2util

from tqdm import tqdm


class FA2(SimStage):
    def __init__(
        self,
        path,
        dataname="nns.mtx",
        initname="data.npy",
        n_components=2,
        random_state=None,
        n_iter=750,
        repulsion=2,
        jitter_tol=1,
        theta=1.2,
        use_degrees=True,
    ):
        super().__init__(
            path,
            dataname=dataname,
            initname=initname,
            n_components=n_components,
            random_state=random_state,
        )

        self.n_iter = n_iter
        self.repulsion = repulsion
        self.jitter_tol = jitter_tol
        self.theta = theta
        self.use_degrees = use_degrees

    def transform(self):

        fa2 = ForceAtlas2(
            verbose=False,
            jitterTolerance=self.jitter_tol,
            scalingRatio=self.repulsion,
            barnesHutTheta=self.theta,
            degreeRepulsion=self.use_degrees,
        )

        saver = (lambda i, pos: np.save(self.outdir / str(i), pos),)

        self.data_ = fa2.forceatlas2(
            self.data,
            pos=self.init,
            iterations=self.n_iter,
            callbacks_every_iters=1,
            callbacks=saver,
        )


class ForceAtlas2(fa2.ForceAtlas2):
    def init(
        self,
        G,  # a graph in 2D np ndarray format (or) scipy sparse matrix format
        pos=None,  # Array of initial positions
    ):
        isSparse = False
        if isinstance(G, np.ndarray):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert np.all(
                G.T == G
            ), "G is not symmetric.  Currently only undirected graphs are supported"
            assert isinstance(pos, np.ndarray) or (
                pos is None
            ), "Invalid node positions"
        elif scipy.sparse.issparse(G):
            # Check our assumptions
            assert G.shape == (G.shape[0], G.shape[0]), "G is not 2D square"
            assert isinstance(pos, np.ndarray) or (
                pos is None
            ), "Invalid node positions"
            G = G.tolil()
            isSparse = True
        else:
            assert False, "G is not np ndarray or scipy sparse matrix"

        # Put nodes into a data structure we can understand
        nodes = []
        for i in range(0, G.shape[0]):
            n = fa2util.Node()
            if isSparse and self.degreeRepulsion:
                n.mass = 1 + len(G.rows[i])
            elif self.degreeRepulsion:
                n.mass = 1 + np.count_nonzero(G[i])
            else:
                n.mass = 1
            n.old_dx = 0
            n.old_dy = 0
            n.dx = 0
            n.dy = 0
            if pos is None:
                n.x = random.random()
                n.y = random.random()
            else:
                n.x = pos[i][0]
                n.y = pos[i][1]
            nodes.append(n)

        # Put edges into a data structure we can understand
        edges = []
        es = np.asarray(G.nonzero()).T
        for e in es:  # Iterate through edges
            if e[1] <= e[0]:
                continue  # Avoid duplicate edges
            edge = fa2util.Edge()
            edge.node1 = e[0]  # The index of the first node in `nodes`
            edge.node2 = e[1]  # The index of the second node in `nodes`
            edge.weight = G[tuple(e)]
            edges.append(edge)

        return nodes, edges

    # Given an adjacency matrix, this function computes the node positions
    # according to the ForceAtlas2 layout algorithm.  It takes the same
    # arguments that one would give to the ForceAtlas2 algorithm in Gephi.
    # Not all of them are implemented.  See below for a description of
    # each parameter and whether or not it has been implemented.
    #
    # This function will return a list of X-Y coordinate tuples, ordered
    # in the same way as the rows/columns in the input matrix.
    #
    # The only reason you would want to run this directly is if you don't
    # use networkx.  In this case, you'll likely need to convert the
    # output to a more usable format.  If you do use networkx, use the
    # "forceatlas2_networkx_layout" function below.
    #
    # Currently, only undirected graphs are supported so the adjacency matrix
    # should be symmetric.
    def forceatlas2(
        self,
        G,  # a graph in 2D np ndarray format (or) scipy sparse matrix format
        pos=None,  # Array of initial positions
        iterations=100,  # Number of times to iterate the main loop
        callbacks_every_iters=0,
        callbacks=None,
    ):
        # Initializing, initAlgo()
        # ================================================================

        # speed and speedEfficiency describe a scaling factor of dx and dy
        # before x and y are adjusted.  These are modified as the
        # algorithm runs to help ensure convergence.
        speed = 1.0
        speedEfficiency = 1.0
        nodes, edges = self.init(G, pos)
        outboundAttCompensation = 1.0
        if self.outboundAttractionDistribution:
            outboundAttCompensation = np.mean([n.mass for n in nodes])
        # ================================================================

        # Main loop, i.e. goAlgo()
        # ================================================================

        barneshut_timer = fa2.Timer(name="BarnesHut Approximation")
        repulsion_timer = fa2.Timer(name="Repulsion forces")
        gravity_timer = fa2.Timer(name="Gravitational forces")
        attraction_timer = fa2.Timer(name="Attraction forces")
        applyforces_timer = fa2.Timer(name="AdjustSpeedAndApplyForces step")

        # transform into list if single func is passed
        if callable(callbacks):
            callbacks = [callbacks]

        # Each iteration of this loop represents a call to goAlgo().
        niters = range(iterations)
        if self.verbose:
            niters = tqdm(niters)
        for _i in niters:
            for i, n in enumerate(nodes):
                n.old_dx = n.dx
                n.old_dy = n.dy
                n.dx = 0
                n.dy = 0
                pos[i, 0] = n.x
                pos[i, 1] = n.y

            if callbacks_every_iters > 0 and (_i % callbacks_every_iters == 0):
                if callbacks is not None:
                    [c(_i, pos) for c in callbacks]

            # Barnes Hut optimization
            if self.barnesHutOptimize:
                barneshut_timer.start()
                rootRegion = fa2util.Region(nodes)
                rootRegion.buildSubRegions()
                barneshut_timer.stop()

            # Charge repulsion forces
            repulsion_timer.start()
            # parallelization should be implemented here
            if self.barnesHutOptimize:
                rootRegion.applyForceOnNodes(
                    nodes, self.barnesHutTheta, self.scalingRatio
                )
            else:
                fa2util.apply_repulsion(nodes, self.scalingRatio)
            repulsion_timer.stop()

            # Gravitational forces
            gravity_timer.start()
            fa2util.apply_gravity(
                nodes, self.gravity, useStrongGravity=self.strongGravityMode
            )
            gravity_timer.stop()

            # If other forms of attraction were implemented they would be selected here.
            attraction_timer.start()
            fa2util.apply_attraction(
                nodes,
                edges,
                self.outboundAttractionDistribution,
                outboundAttCompensation,
                self.edgeWeightInfluence,
            )
            attraction_timer.stop()

            # Adjust speeds and apply forces
            applyforces_timer.start()
            values = fa2util.adjustSpeedAndApplyForces(
                nodes, speed, speedEfficiency, self.jitterTolerance
            )
            speed = values["speed"]
            speedEfficiency = values["speedEfficiency"]
            applyforces_timer.stop()

        if self.verbose:
            if self.barnesHutOptimize:
                barneshut_timer.display()
            repulsion_timer.display()
            gravity_timer.display()
            attraction_timer.display()
            applyforces_timer.display()
        # ================================================================
        return [(n.x, n.y) for n in nodes]
