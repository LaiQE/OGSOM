# -*- coding: utf-8 -*-
""" 这个文件是dex-net中的源码修改的
"""
import logging
import numpy as np
try:
    import pyhull.convex_hull as cvh
except:
    logging.warning('Failed to import pyhull')
try:
    import cvxopt as cvx
except:
    logging.warning('Failed to import cvx')
import sys
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# turn off output logging
cvx.solvers.options['show_progress'] = False


class PointGraspMetrics3D:
    """ Class to wrap functions for quasistatic point grasp quality metrics.
    """

    @staticmethod
    def grasp_matrix(forces, torques, normals, soft_fingers=False,
                     finger_radius=0.005, params=None):
        """ Computes the grasp map between contact forces and wrenchs on the object in its reference frame.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        finger_radius : float
            the radius of the fingers to use
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        G : 6xM :obj:`numpy.ndarray`
            grasp map
        """
        if params is not None and 'finger_radius' in params.keys():
            finger_radius = params['finger_radius']
        num_forces = forces.shape[1]
        num_torques = torques.shape[1]
        if num_forces != num_torques:
            raise ValueError('Need same number of forces and torques')

        num_cols = num_forces
        if soft_fingers:
            num_normals = 2
            if normals.ndim > 1:
                num_normals = 2*normals.shape[1]
            # print(num_normals, normals.ndim)
            num_cols = num_cols + num_normals

        G = np.zeros([6, num_cols])
        for i in range(num_forces):
            G[:3, i] = forces[:, i]
            G[3:, i] = params['torque_scaling'] * torques[:, i]

        if soft_fingers:
            torsion = np.pi * finger_radius**2 * \
                params['friction_coef'] * normals * params['torque_scaling']
            pos_normal_i = int(-num_normals)
            neg_normal_i = int(-num_normals + num_normals / 2)
            # print(pos_normal_i, neg_normal_i)
            G[3:, pos_normal_i:neg_normal_i] = torsion
            G[3:, neg_normal_i:] = -torsion

        return G

    @staticmethod
    def force_closure(c1, c2, friction_coef, use_abs_value=True):
        """" Checks force closure using the antipodality trick.

        Parameters
        ----------
        c1 : :obj:`Contact3D`
            first contact point
        c2 : :obj:`Contact3D`
            second contact point
        friction_coef : float
            coefficient of friction at the contact point
        use_abs_value : bool
            whether or not to use directoinality of the surface normal (useful when mesh is not oriented)

        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        if c1.point is None or c2.point is None or c1.normal is None or c2.normal is None:
            return 0
        p1, p2 = c1.point, c2.point
        n1, n2 = -c1.normal, -c2.normal  # inward facing normals

        if (p1 == p2).all():  # same point
            return 0

        for normal, contact, other_contact in [(n1, p1, p2), (n2, p2, p1)]:
            diff = other_contact - contact
            normal_proj = normal.dot(diff) / np.linalg.norm(normal)
            if use_abs_value:
                normal_proj = abs(normal.dot(diff)) / np.linalg.norm(normal)

            if normal_proj < 0:
                return 0  # wrong side
            alpha = np.arccos(normal_proj / np.linalg.norm(diff))
            if alpha > np.arctan(friction_coef):
                return 0  # outside of friction cone
        return 1

    @staticmethod
    def force_closure_qp(forces, torques, normals, soft_fingers=False,
                         wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                         params=None):
        """ Checks force closure by solving a quadratic program (whether or not zero is in the convex hull)

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in force closure, 0 otherwise
        """
        if params is not None:
            if 'wrench_norm_thresh' in params.keys():
                wrench_norm_thresh = params['wrench_norm_thresh']
            if 'wrench_regularizer' in params.keys():
                wrench_regularizer = params['wrench_regularizer']

        G = PointGraspMetrics3D.grasp_matrix(
            forces, torques, normals, soft_fingers, params=params)
        min_norm, _ = PointGraspMetrics3D.min_norm_vector_in_facet(
            G, wrench_regularizer=wrench_regularizer)
        # if greater than wrench_norm_thresh, 0 is outside of hull
        return 1 * (min_norm < wrench_norm_thresh)

    @staticmethod
    def partial_closure(forces, torques, normals, soft_fingers=False,
                        wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                        params=None):
        """ Evalutes partial closure: whether or not the forces and torques can resist a specific wrench.
        Estimates resistance by sollving a quadratic program (whether or not the target wrench is in the convex hull).

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        int : 1 if in partial closure, 0 otherwise
        """
        force_limit = None
        if params is None:
            return 0
        force_limit = params['force_limits']
        target_wrench = params['target_wrench']
        if 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params['wrench_norm_thresh']
        if 'wrench_regularizer' in params.keys():
            wrench_regularizer = params['wrench_regularizer']

        # reorganize the grasp matrix for easier constraint enforcement in optimization
        num_fingers = normals.shape[1]
        num_wrenches_per_finger = forces.shape[1] / num_fingers
        G = np.zeros([6, 0])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            G_i = PointGraspMetrics3D.grasp_matrix(forces[:, start_i:end_i], torques[:, start_i:end_i], normals[:, i:i+1],
                                                   soft_fingers, params=params)
            G = np.c_[G, G_i]

        wrench_resisted, _ = PointGraspMetrics3D.wrench_in_positive_span(G, target_wrench, force_limit, num_fingers,
                                                                         wrench_norm_thresh=wrench_norm_thresh,
                                                                         wrench_regularizer=wrench_regularizer)
        return 1 * wrench_resisted

    @staticmethod
    def wrench_resistance(forces, torques, normals, soft_fingers=False,
                          wrench_norm_thresh=1e-3, wrench_regularizer=1e-10,
                          finger_force_eps=1e-9, params=None):
        """ Evalutes wrench resistance: the inverse norm of the contact forces required to resist a target wrench
        Estimates resistance by sollving a quadratic program (min normal contact forces to produce a wrench).

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite
        finger_force_eps : float
            small float to prevent numeric issues in wrench resistance metric
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of wrench resistance metric
        """
        force_limit = None
        if params is None:
            return 0
        force_limit = params['force_limits']
        target_wrench = params['target_wrench']
        if 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params['wrench_norm_thresh']
        if 'wrench_regularizer' in params.keys():
            wrench_regularizer = params['wrench_regularizer']
        if 'finger_force_eps' in params.keys():
            finger_force_eps = params['finger_force_eps']

        # reorganize the grasp matrix for easier constraint enforcement in optimization
        num_fingers = normals.shape[1]
        num_wrenches_per_finger = forces.shape[1] / num_fingers
        G = np.zeros([6, 0])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            G_i = PointGraspMetrics3D.grasp_matrix(forces[:, start_i:end_i], torques[:, start_i:end_i], normals[:, i:i+1],
                                                   soft_fingers, params=params)
            G = np.c_[G, G_i]

        # compute metric from finger force norm
        Q = 0
        wrench_resisted, finger_force_norm =\
            PointGraspMetrics3D.wrench_in_positive_span(G, target_wrench, force_limit, num_fingers,
                                                        wrench_norm_thresh=wrench_norm_thresh,
                                                        wrench_regularizer=wrench_regularizer)
        if wrench_resisted:
            Q = 1.0 / (finger_force_norm + finger_force_eps) - 1.0 / (2 * force_limit)
        return Q

    @staticmethod
    def min_singular(forces, torques, normals, soft_fingers=False, params=None):
        """ Min singular value of grasp matrix - measure of wrench that grasp is "weakest" at resisting.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of smallest singular value
        """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        min_sig = S[5]
        return min_sig

    @staticmethod
    def wrench_volume(forces, torques, normals, soft_fingers=False, params=None):
        """ Volume of grasp matrix singular values - score of all wrenches that the grasp can resist.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of wrench volume
        """
        k = 1
        if params is not None and 'k' in params.keys():
            k = params['k']

        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        sig = S
        return k * np.sqrt(np.prod(sig))

    @staticmethod
    def grasp_isotropy(forces, torques, normals, soft_fingers=False, params=None):
        """ Condition number of grasp matrix - ratio of "weakest" wrench that the grasp can exert to the "strongest" one.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model

        Returns
        -------
        float : value of grasp isotropy metric
        """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        max_sig = S[0]
        min_sig = S[5]
        isotropy = min_sig / max_sig
        if np.isnan(isotropy) or np.isinf(isotropy):
            return 0
        return isotropy

    @staticmethod
    def ferrari_canny_L1(forces, torques, normals, soft_fingers=False, params=None,
                         wrench_norm_thresh=1e-3,
                         wrench_regularizer=1e-10):
        """ Ferrari & Canny's L1 metric. Also known as the epsilon metric.

        Parameters
        ----------
        forces : 3xN :obj:`numpy.ndarray`
            set of forces on object in object basis
        torques : 3xN :obj:`numpy.ndarray`
            set of torques on object in object basis
        normals : 3xN :obj:`numpy.ndarray`
            surface normals at the contact points
        soft_fingers : bool
            whether or not to use the soft finger contact model
        params : :obj:`GraspQualityConfig`
            set of parameters for grasp matrix and contact model
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        float : value of metric
        """
        if params is not None and 'wrench_norm_thresh' in params.keys():
            wrench_norm_thresh = params['wrench_norm_thresh']
        if params is not None and 'wrench_regularizer' in params.keys():
            wrench_regularizer = params['wrench_regularizer']

        # create grasp matrix
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals,
                                             soft_fingers, params=params)
        s = time.time()
        # center grasp matrix for better convex hull comp
        hull = cvh.ConvexHull(G.T)
        # TODO: suppress ridiculous amount of output for perfectly valid input to qhull
        e = time.time()
        logging.debug('CVH took %.3f sec' % (e - s))

        debug = False
        if debug:
            fig = plt.figure()
            torques = G[3:, :].T
            ax = Axes3D(fig)
            ax.scatter(torques[:, 0], torques[:, 1], torques[:, 2], c='b', s=50)
            ax.scatter(0, 0, 0, c='k', s=80)
            ax.set_xlim3d(-1.5, 1.5)
            ax.set_ylim3d(-1.5, 1.5)
            ax.set_zlim3d(-1.5, 1.5)
            ax.set_xlabel('tx')
            ax.set_ylabel('ty')
            ax.set_zlabel('tz')
            plt.show()

        if len(hull.vertices) == 0:
            logging.warning('Convex hull could not be computed')
            return 0.0

        # determine whether or not zero is in the convex hull
        s = time.time()
        min_norm_in_hull, v = PointGraspMetrics3D.min_norm_vector_in_facet(
            G, wrench_regularizer=wrench_regularizer)
        e = time.time()
        logging.debug('Min norm took %.3f sec' % (e - s))

        # if norm is greater than 0 then forces are outside of hull
        if min_norm_in_hull > wrench_norm_thresh:
            logging.debug('Zero not in convex hull')
            return 0.0

        # if there are fewer nonzeros than D-1 (dim of space minus one)
        # then zero is on the boundary and therefore we do not have
        # force closure
        if np.sum(v > 1e-4) <= G.shape[0]-1:
            logging.debug('Zero not in interior of convex hull')
            return 0.0

        # find minimum norm vector across all facets of convex hull
        s = time.time()
        min_dist = sys.float_info.max
        closest_facet = None
        for v in hull.vertices:
            # because of some occasional odd behavior from pyhull
            if np.max(np.array(v)) < G.shape[1]:
                facet = G[:, v]
                dist, _ = PointGraspMetrics3D.min_norm_vector_in_facet(
                    facet, wrench_regularizer=wrench_regularizer)
                if dist < min_dist:
                    min_dist = dist
                    closest_facet = v
        e = time.time()
        logging.debug('Min dist took %.3f sec for %d vertices' % (e - s, len(hull.vertices)))

        return min_dist

    @staticmethod
    def wrench_in_positive_span(wrench_basis, target_wrench, force_limit, num_fingers=1,
                                wrench_norm_thresh=1e-4, wrench_regularizer=1e-10):
        """ Check whether a target can be exerted by positive combinations of
        wrenches in a given basis with L1 norm fonger force limit limit.

        Parameters
        ----------
        wrench_basis : 6xN :obj:`numpy.ndarray`
            basis for the wrench space
        target_wrench : 6x1 :obj:`numpy.ndarray`
            target wrench to resist
        force_limit : float
            L1 upper bound on the forces per finger (aka contact point)
        num_fingers : int
            number of contacts, used to enforce L1 finger constraint
        wrench_norm_thresh : float
            threshold to use to determine equivalence of target wrenches
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        int
            whether or not wrench can be resisted
        float
            minimum norm of the finger forces required to resist the wrench
        """
        num_wrenches = wrench_basis.shape[1]

        # quadratic and linear costs
        P = wrench_basis.T.dot(wrench_basis) + wrench_regularizer*np.eye(num_wrenches)
        q = -wrench_basis.T.dot(target_wrench)

        # inequalities
        lam_geq_zero = -1 * np.eye(num_wrenches)

        num_wrenches_per_finger = num_wrenches / num_fingers
        force_constraint = np.zeros([num_fingers, num_wrenches])
        for i in range(num_fingers):
            start_i = num_wrenches_per_finger * i
            end_i = num_wrenches_per_finger * (i + 1)
            force_constraint[i, start_i:end_i] = np.ones(num_wrenches_per_finger)

        G = np.r_[lam_geq_zero, force_constraint]
        h = np.zeros(num_wrenches+num_fingers)
        for i in range(num_fingers):
            h[num_wrenches+i] = force_limit

        # convert to cvx and solve
        P = cvx.matrix(P)
        q = cvx.matrix(q)
        G = cvx.matrix(G)
        h = cvx.matrix(h)
        sol = cvx.solvers.qp(P, q, G, h)
        v = np.array(sol['x'])

        min_dist = np.linalg.norm(wrench_basis.dot(v).ravel() - target_wrench)**2

        # add back in the target wrench
        return min_dist < wrench_norm_thresh, np.linalg.norm(v)

    @staticmethod
    def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
        """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.

        Parameters
        ----------
        facet : 6xN :obj:`numpy.ndarray`
            vectors forming the facet
        wrench_regularizer : float
            small float to make quadratic program positive semidefinite

        Returns
        -------
        float
            minimum norm of any point in the convex hull of the facet
        Nx1 :obj:`numpy.ndarray`
            vector of coefficients that achieves the minimum
        """
        dim = facet.shape[1]  # num vertices in facet

        # create alpha weights for vertices of facet
        G = facet.T.dot(facet)
        grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
        P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
        q = cvx.matrix(np.zeros((dim, 1)))
        G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
        h = cvx.matrix(np.zeros((dim, 1)))
        A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
        b = cvx.matrix(np.ones(1))         # combinations of vertices

        sol = cvx.solvers.qp(P, q, G, h, A, b)
        v = np.array(sol['x'])
        min_norm = np.sqrt(sol['primal objective'])

        return abs(min_norm), v
