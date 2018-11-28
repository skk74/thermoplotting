from __future__ import absolute_import
from __future__ import division

import scipy.spatial as spa
import scipy.linalg as lin
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from . import systems

def prepare_input(file_name):
    """Read in from list of files and output them as new .thin
    files that are compatible with the expected format:
    composition0, composition1, energy

    :file_name: name of file, presumably casm output
    :returns: void, but you'll end up with new a file "oldfile.thin"

    """
    pass


def composition(data_list, component):
    """Returns one composition column from the given data, which
    was presumably made using the standard clobber() function.

    :data_list: double numpy array
    :component: int specifying which composition to return (2=origin)
    :returns: list of first composition values

    """
    if component==3:
        oneslist=np.ones(data_list.shape[0])
        origincomp=oneslist-data_list[:,0]-data_list[:,1]-data_list[:,2]
        return origincomp

    else:
        return data_list[:,component]


def energy(data_list):
    """Returns energy column from the given data, which
    was presumably made using the standard clobber() function

    :data_list: double numpy array
    :returns: list of second composition values

    """
    return data_list[:,3]

def equil_trans(data_list):
    """Recursively goes through all the data points and applies shear matrix
    so that the composition space looks like an equilateral triangle

    :data_list: arbitrarily dimensioned array, can be clobber() or faces()
    :returns: double numpy array with transformed composition values

    """
    transmat = np.array([[1, 0.5, 0.5, 0], [0, 3**(0.5)/2, 0.5/3**(0.5), 0], [0, 0, (2/3)**(0.5),0],[0, 0, 0, 1]])

    # If we have 2 or 1 dimnesions, we can directly dot
    if data_list.ndim < 3:
        return np.dot(transmat, data_list.T).T
    # Otherwise use recursion to dot each entry in the data_list
    else:
        return np.array([equil_trans(dlist) for dlist in data_list])


def _digest_data(x, y, z, e):
    """Given 4D data, shear the x, y, and z values so that the
    space is a regular tetrahedron

    :x: list of composition
    :y: list of composition
    :z: list of composition
    :e: list of energy
    :returns: np array

    """
    digestable = np.array((np.ravel(x), np.ravel(y), np.ravel(z), np.ravel(e))).T
    digested = equil_trans(digestable)

    return digested

def _get_min_4d_points(points):
    """Given 4D data, return the points at each unique x y and z
    that have the minimum 4d value
    """
    zipped_coords=sorted(zip(points[:,0],points[:,1],points[:,2],points[:,3]))
    min_list=[0]
    for i in range(1,len(zipped_coords)):
        if zipped_coords[i][0]!=zipped_coords[i-1][0] or zipped_coords[i][1]!= zipped_coords[i-1][1] or zipped_coords[i][2]!= zipped_coords[i-1][2]:
            min_list.append(i)
    min_coords=[zipped_coords[i] for i in min_list]
    return min_coords




def _draw_projected_facet(ax, facet, kwargs):
    """Draw a single facet onto the composition plane

    :ax: mpl axis
    :facet: from scipy.Hull
    :kwargs: Polygon arguments
    :returns: ax

    """
    coords = facet[:, [0, 1, 2]]
    tetr = Poly3DCollection([coords], **kwargs)
    tetr.set_facecolor((1,1,1,0.01))
    ax.add_collection3d(tetr)
    return ax

def _scatter_projected_facet(ax, facet, **kwargs):
    """Scatter the corners of a facet onto the composition plane

    Parameters
    ----------
    :ax: mpl axis
    :facet: from scipy.Hull
    :kwargs: Polygon arguments
    :returns: ax


    Returns
    -------
    mpl axis

    """
    coords = facet[:, [0, 1, 2]]
    ax.scatter(facet[:,0],facet[:,1],facet[:,2],**kwargs)
    return ax

def pruned_hull_facets(data_list,tol=0.00001):
    """Calls facets() to get all facets of the convex hull, but then
    removes all facets on binary subspace, as well as any resulting
    facets above the reference endpoints.

    :data_list: numpy array of numpy array
    :returns: array of facets (collection of 3 points)

    """
    #TODO: Consolidate with routines in hull module
    new_tetr = spa.ConvexHull(data_list) #pylint: disable=no-member
    good_simplex=[]

    for eq,sx in zip(new_tetr.equations,new_tetr.simplices):
        if eq[3]<0-tol:
            good_simplex.append(sx)
    good_simplex=np.vstack(good_simplex)
    return new_tetr.points[good_simplex]


def draw_projected_convex_hull(ax,
                               x,
                               y,
                               z,
                               e,
                               kwargs={
                                   "edgecolor": "black",
                                   "linewidth": 1.5
                               }):
    """Draw the convex hull, but suppress the energy axis, yielding projected facets
    onto the composition plane

    :ax: matplotlib axis
    :kwargs: keyword arguments for the add_patch function
    :returns: matplotlib axis

    """
    digested = _digest_data(x, y, z, e)
    facets = pruned_hull_facets(digested)

    for f in facets:
        ax = _draw_projected_facet(ax, f, kwargs)

    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
    ax.set_aspect('equal')

    return ax

def hull_dist_contour(ax,
                               x,
                               y,
                               z,
                               e,
                               kwargs={
                                   "edgecolor": "black",
                                   "linewidth": 1.5
                               }):
    """Draw the convex hull, but suppress the energy axis, yielding projected facets
    onto the composition plane

    :ax: matplotlib axis
    :kwargs: keyword arguments for the add_patch function
    :returns: matplotlib axis

    """
    digested = _digest_data(x, y, z, e)
    closest_points=_get_min_4d_points(digested)
    closest_hull_dists=systems.hull.distances_from_hull(closest_points,ConvexHull(digested))
    print("max hull dist")
    print(max(closest_hull_dists))
    #closest_hull_dists=closest_hull_dists/.200
    #print("max hull dist")
    #print(max(closest_hull_dists))
    facets = pruned_hull_facets(digested)
    closest_xs=[c[0] for c in closest_points]
    closest_ys=[c[1] for c in closest_points]
    closest_zs=[c[2] for c in closest_points]
    for f in facets:
        ax = _draw_projected_facet(ax, f, kwargs)
    ax.scatter(closest_xs,closest_ys,closest_zs,c=closest_hull_dists,s=40,vmin=0,vmax=0.3,cmap='viridis')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
    ax.set_aspect('equal')

    return ax



def scatter_projected_convex_hull(ax, x, y, z, e, **kwargs):
    """Draw the points that make up the convex hull, but plot
    them projected onto the composition plane   

    Parameters
    ----------
    ax : mpl axis
    x : composition data
    y : composition data
    z : composition data
    :kwargs: keyword arguments for the add_patch function

    Returns
    -------
    mpl axis

    """
    digested = _digest_data(x, y, z, e)
    facets = pruned_hull_facets(digested)

    for f in facets:
        ax = _scatter_projected_facet(ax, f, **kwargs)

    #Set projected view? axis, aspect, etc?
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
    ax.set_aspect('equal')
    return ax

def hull_points(x, y, z, e):
    """Return the points that make up the convex hull
    Parameters
    ----------
    x : composition data
    y : composition data
    z : composition data
    z : energy data

    Returns
    -------
    list of points

    """
    digested = _digest_data(x, y, z, e)
    facets = pruned_hull_facets(digested)
    hull_points=[]
    for f in facets:
        for point in f:
            hull_points.append(tuple(point))
    hull_points=list(set(hull_points))
    for i,pt in enumerate(hull_points):
        hull_points[i]=np.array(pt)
    return hull_points



