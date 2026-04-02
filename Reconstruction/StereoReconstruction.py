import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as p3
import pylab as p
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay
from scipy import interpolate
from scipy.spatial import cKDTree
import networkx as nx
import pandas as pd
import math
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.cluster import KMeans
import hyperspy.api as hs
import tifffile as tif
from tifffile import imwrite

def get_spline(points,start,end,smooth):
    vor=Voronoi(points)
    inds=vor.ridge_points
    thresh=5
    pt1=np.array(points[inds][:,0])
    pt2=np.array(points[inds][:,1])
    dist = np.linalg.norm(pt1-pt2,axis=1)
    is_in = dist < thresh
    inds_thresh=inds[is_in,...]
    
    kdt = cKDTree(points) 
    dist0, idx0 = kdt.query(start) # Starting point of the path
    dist1, idx1 = kdt.query(end) # ending point of the path

# compute shortest weighted path
    edge_lengths = [np.linalg.norm(points[e[0], :] - points[e[1], :]) for e in inds_thresh]
    g = nx.Graph((i, j, {'weight': dist}) for (i, j), dist in zip(inds_thresh, edge_lengths))
    path_s = nx.shortest_path(g,source=idx0,target=idx1, weight='weight')
    
    tck,u = scipy.interpolate.splprep([points[path_s][:,0],points[path_s][:,1]],s=smooth)
    return tck,u

#everything explained in : https://stackoverflow.com/questions/77099540/how-to-convert-a-bspline-ppoly-object-into-an-array-of-polynomials-list
#indPoly is the poly from the indPolyth interval
def get_poly(polys,x,indPoly,find_y=True):
    if find_y==True : 
        i=1 #it will interpolate the y for a given x
    else:
        i=0 #it will interpolate the x for a given y
    poly1=polys[i].c[0,indPoly]*(x-polys[i].x[indPoly])**3 + polys[i].c[1,indPoly]*(x-polys[i].x[indPoly])**2 + polys[i].c[2,indPoly]*(x-polys[i].x[indPoly]) + polys[i].c[3,indPoly]
    deriv1=3*polys[i].c[0,indPoly]*(x-polys[i].x[indPoly])**2 + 2*polys[i].c[1,indPoly]*(x-polys[i].x[indPoly]) + polys[i].c[2,indPoly]
    deriv2=6*polys[i].c[0,indPoly]*(x-polys[i].x[indPoly]) + 2*polys[i].c[1,indPoly]
    return poly1

#version qui exclue les valeurs complexes
def get_points_from_polys(Ys,polys,tck) :
    roots_ok=np.ndarray((0,2))
    for i in range(polys[1].x.shape[0]-1):
        coeff=polys[1].c.T[i]
        xp=polys[1].x[i]
        for y in Ys:
            coeffs_new=np.array([coeff[0],(-coeff[0]*3*xp+coeff[1]),(3*coeff[0]*xp**2-2*coeff[1]*xp+coeff[2]),(-coeff[0]*xp**3+coeff[1]*xp**2-coeff[2]*xp+coeff[3])])
            coeffs_new[3]=coeffs_new[3]-y
            xs=np.roots(coeffs_new)
            x_roots=[x for x in xs if x>=polys[1].x[i] and x<polys[1].x[i+1]]
            if len(x_roots)>0:
                roots=[[interpolate.splev(root,tck)[0].reshape(1)[0],y] for root in x_roots if np.isreal(root)==True]
                if len(roots)>0:
                    roots_ok=np.append(roots_ok,roots,axis=0)
    return roots_ok

# permet de mettre les deux listes à la même taille
def shift_spline(points_l,points_r,tck_l,tck_r,smooth):

    Ux=np.linspace(0,1,100)
    
    diffs=np.linspace(np.mean(points_l[:,0])-np.mean(points_r[:,0])-10,np.mean(points_l[:,0])-np.mean(points_r[:,0])+10,100)  
    diff = 100
    sx_l,sy_l=interpolate.splev(Ux,tck_l)
    sx_r,sy_r=interpolate.splev(Ux,tck_r)
    for i in diffs:
        tck_l_test,u = scipy.interpolate.splprep([sx_l-i,sy_l],per=True,s=smooth)
        tck_r_test,u = scipy.interpolate.splprep([sx_r,sy_r],per=True,s=smooth)
        spl_l=scipy.interpolate.BSpline(tck_l_test[0],tck_l_test[1][0],tck_l_test[2])
        spl_r=scipy.interpolate.BSpline(tck_r_test[0],tck_r_test[1][0],tck_r_test[2])
        if abs(spl_l.integrate(0,1)-spl_r.integrate(0,1))<diff:
            diff=abs(spl_l.integrate(0,1)-spl_r.integrate(0,1))
            shifto=i
    return shifto

# faire du shift avec R²
def make_same_size(points_l,points_r,tck_l,tck_r,roots_l,roots_r,start_l,start_r,smooth):
    x_tot_big=[]
    x_tot_small=[]
    y_tot=[]
    diff=shift_spline(points_l,points_r,tck_l,tck_r,smooth)
    roots_l=roots_l-[diff,0]

    if len(roots_l)>len(roots_r): #we check which one is the smallest list
        big_list=roots_l
        small_list=roots_r
        ind=1
    else:
        big_list=roots_r
        small_list=roots_l
        ind=2
    
    small_list=small_list[small_list[:, 1].argsort()] #we sort along the y axis
    for i in range(len(small_list)):
        common=np.array(np.where(np.isin(big_list[:,1],small_list[i,1])==True))[0] #we check if there is a common y for the i th point of small list
        if len(common)==1: #on regarde si la liste ne contient qu'une valeur de y correspondante
            x_tot_big=np.append(x_tot_big,big_list[common[0]][0])
            x_tot_small=np.append(x_tot_small,small_list[i,0])
            y_tot=np.append(y_tot,small_list[i,1])
        elif len(common)>1: # if multiple values we check for the closest one
            ref=np.zeros((len(common),2))+small_list[i]
            to_test=big_list[common]
            dist = np.linalg.norm(ref-to_test,axis=1)
            min_dist=np.where(dist==min(dist))[0]
            x_tot_big=np.append(x_tot_big,big_list[common[min_dist]][0][0])
            x_tot_small=np.append(x_tot_small,small_list[i,0])
            y_tot=np.append(y_tot,small_list[i,1])

    if ind==1:
        roots_l_tot=np.array(list(zip(x_tot_big,y_tot)))+[diff,0]
        roots_r_tot=np.array(list(zip(x_tot_small,y_tot)))
    else:
        roots_r_tot=np.array(list(zip(x_tot_big,y_tot)))
        roots_l_tot=np.array(list(zip(x_tot_small,y_tot)))+[diff,0]
        
    return roots_l_tot,roots_r_tot

def make_same_size(roots_l,roots_r,start_l,start_r):
    x_tot_big=[]
    x_tot_small=[]
    y_tot=[]
    diff=[np.mean(roots_l[:,0])-np.mean(roots_r[:,0]),0] #we want to remove the translation made by the projection due to tilt
    roots_l=roots_l-diff

    if len(roots_l)>len(roots_r): #we check which one is the smallest list
        big_list=roots_l
        small_list=roots_r
        ind=1
    else:
        big_list=roots_r
        small_list=roots_l
        ind=2
    
    small_list=small_list[small_list[:, 1].argsort()] #we sort along the y axis
    for i in range(len(small_list)):
        common=np.array(np.where(np.isin(big_list[:,1],small_list[i,1])==True))[0] #we check if there is a common y for the i th point of small list
        if len(common)==1: #on regarde si la liste ne contient qu'une valeur de y correspondante
            x_tot_big=np.append(x_tot_big,big_list[common[0]][0])
            x_tot_small=np.append(x_tot_small,small_list[i,0])
            y_tot=np.append(y_tot,small_list[i,1])
        elif len(common)>1: # if multiple values we check for the closest one
            ref=np.zeros((len(common),2))+small_list[i]
            to_test=big_list[common]
            dist = np.linalg.norm(ref-to_test,axis=1)
            min_dist=np.where(dist==min(dist))[0]
            x_tot_big=np.append(x_tot_big,big_list[common[min_dist]][0][0])
            x_tot_small=np.append(x_tot_small,small_list[i,0])
            y_tot=np.append(y_tot,small_list[i,1])

    if ind==1:
        roots_l_tot=np.array(list(zip(x_tot_big,y_tot)))+diff
        roots_r_tot=np.array(list(zip(x_tot_small,y_tot)))
    else:
        roots_r_tot=np.array(list(zip(x_tot_big,y_tot)))
        roots_l_tot=np.array(list(zip(x_tot_small,y_tot)))+diff
        
    return roots_l_tot,roots_r_tot
    
def get_3D(data_l,data_r,origin,smooth,old_start_l,old_end_l,old_start_r,old_end_r,alpha1,alpha2):

    points_l=data_l-origin
    points_l=np.array([points_l[:,1],points_l[:,0]]).T
    points_r=data_r-origin
    points_r=np.array([points_r[:,1],points_r[:,0]]).T
    
    #alpha1=20*math.pi/180
    #alpha2=1*math.pi/180

    start_l=np.flip(np.array(old_start_l)-origin)
    end_l=np.flip(np.array(old_end_l)-origin)
    start_r=np.flip(np.array(old_start_r)-origin)
    end_r=np.flip(np.array(old_end_r)-origin)

    Ys=np.linspace(min(min(points_l[:,1]),min(points_r[:,1])),max(max(points_l[:,1]),max(points_r[:,1])),500)
    tck_l,u=get_spline(points_l,start_l,end_l,smooth)
    t_l, c_l, k_l = tck_l
    tck_r,u=get_spline(points_r,start_r,end_r,smooth)
    t_r, c_r, k_r = tck_r

    polys_l = [interpolate.PPoly.from_spline((t_l, cj, k_l)) for cj in c_l]
    polys_r = [interpolate.PPoly.from_spline((t_r, cj, k_r)) for cj in c_r]

    roots_l=get_points_from_polys(Ys,polys_l,tck_l)
    roots_r=get_points_from_polys(Ys,polys_r,tck_r)
    
    roots_l_new,roots_r_new=make_same_size(roots_l,roots_r,start_l,start_r)
    xl=roots_l_new[:,0]
    y_tot=roots_l_new[:,1]
    xr=roots_r_new[:,0]
    x=(math.sin(alpha1)*xr-math.sin(alpha2)*xl)/(math.sin(alpha1-alpha2))
    z=(math.cos(alpha2)*xl-math.cos(alpha1)*xr)/(math.sin(alpha1-alpha2))


    return points_l,points_r,xl,xr,x,z,y_tot,tck_l,tck_r

def get_coordinates(data):
    X=data
    a=np.where(X > 0)
    xi=a[1]
    xi = xi[..., None]
    
    yi=a[0]
    yi = yi[..., None]
#zi=X[xi,yi]
#zi = zi[..., None]
    x_shaped=np.hstack((xi,yi))
    return x_shaped
def compute_delaunay_tetra_circumcenters(dt):

#Compute the centers of the circumscribing circle of each tetrahedron in the Delaunay triangulation.
#:param dt: the Delaunay triangulation
#:return: array of xyz points

    simp_pts = dt.points[dt.simplices]
# (n, 4, 3) array of tetrahedra points where simp_pts[i, j, :] holds the j'th 3D point (of four) of the i'th tetrahedron
    assert simp_pts.shape[1] == 4 and simp_pts.shape[2] == 3

# finding the circumcenter (x, y, z) of a simplex defined by four points:
# (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x1)**2 + (y-y1)**2 + (z-z1)**2
# (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x2)**2 + (y-y2)**2 + (z-z2)**2
# (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x3)**2 + (y-y3)**2 + (z-z3)**2
# becomes three linear equations (squares are canceled):
# 2(x1-x0)*x + 2(y1-y0)*y + 2(z1-z0)*y = (x1**2 + y1**2 + z1**2) - (x0**2 + y0**2 + z0**2)
# 2(x2-x0)*x + 2(y2-y0)*y + 2(z2-z0)*y = (x2**2 + y2**2 + z2**2) - (x0**2 + y0**2 + z0**2)
# 2(x3-x0)*x + 2(y3-y0)*y + 2(z3-z0)*y = (x3**2 + y3**2 + z3**2) - (x0**2 + y0**2 + z0**2)

# building the 3x3 matrix of the linear equations
    a = 2 * (simp_pts[:, 1, 0] - simp_pts[:, 0, 0])
    b = 2 * (simp_pts[:, 1, 1] - simp_pts[:, 0, 1])
    c = 2 * (simp_pts[:, 1, 2] - simp_pts[:, 0, 2])
    d = 2 * (simp_pts[:, 2, 0] - simp_pts[:, 0, 0])
    e = 2 * (simp_pts[:, 2, 1] - simp_pts[:, 0, 1])
    f = 2 * (simp_pts[:, 2, 2] - simp_pts[:, 0, 2])
    g = 2 * (simp_pts[:, 3, 0] - simp_pts[:, 0, 0])
    h = 2 * (simp_pts[:, 3, 1] - simp_pts[:, 0, 1])
    i = 2 * (simp_pts[:, 3, 2] - simp_pts[:, 0, 2])

    v1 = (simp_pts[:, 1, 0] ** 2 + simp_pts[:, 1, 1] ** 2 + simp_pts[:, 1, 2] ** 2) - (simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2)
    v2 = (simp_pts[:, 2, 0] ** 2 + simp_pts[:, 2, 1] ** 2 + simp_pts[:, 2, 2] ** 2) - (simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2)
    v3 = (simp_pts[:, 3, 0] ** 2 + simp_pts[:, 3, 1] ** 2 + simp_pts[:, 3, 2] ** 2) - (simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2)

# solve a 3x3 system by inversion (see https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices)
    A = e*i - f*h
    B = -(d*i - f*g)
    C = d*h - e*g
    D = -(b*i - c*h)
    E = a*i - c*g
    F = -(a*h - b*g)
    G = b*f - c*e
    H = -(a*f - c*d)
    I = a*e - b*d

    det = a*A + b*B + c*C

# multiplying inv*[v1, v2, v3] to get solution point (x, y, z)
    x = (A*v1 + D*v2 + G*v3) / det
    y = (B*v1 + E*v2 + H*v3) / det
    z = (C*v1 + F*v2 + I*v3) / det

   # print(x.shape)
    xyz=np.vstack((x, y, z))

    #xyz=np.delete(xyz,np.isnan(xyz).any(axis=0),1)
    #xyz=xyz[~np.isnan(xyz).any(axis=1)]
   # print(xyz.shape)
    

    return (xyz).T

# ici on essaye d'enlever les ligne qui sont dans la courbure 

def compute_voronoi_vertices_and_edges(points, r_thresh):

#Compute (finite) Voronoi edges and vertices of a set of points.
#:param points: input points.
#:param r_thresh: radius value for filtering out vertices corresponding to
#Delaunay tetrahedrons with large radii of circumscribing sphere (alpha-shape condition).
#:return: array of xyz Voronoi vertex points and an edge list.

    dt = Delaunay(points)
    xyz_centers = compute_delaunay_tetra_circumcenters(dt)
    tetra_in=[]

# filtering tetrahedrons that have radius > thresh
    simp_pts_0 = dt.points[dt.simplices[:, 0]]
    radii = np.linalg.norm(xyz_centers - simp_pts_0, axis=1)
    is_in = radii < r_thresh

# build an edge list from (filtered) tetrahedrons neighbor relations
    edge_lst = []
    for i in range(len(dt.neighbors)):
        if not is_in[i]:
            continue  # i is an outside tetra
        tetra_in=np.append(tetra_in,i)   # ici on stock les tetrahèdre  qui correspondent au critère
        for j in dt.neighbors[i]:
            if j != -1 and is_in[j]:
                edge_lst.append((i, j))

    return xyz_centers, edge_lst, tetra_in

#superpose les centres de tetrahedres avec les tétraèdres seuillés et créer une liste de edges des tetrahedres
def plot_vern(ax, centers,points, tri,tetra_in):
    inds=np.ndarray((0,2))
    for tr in tetra_in:
        cents = centers[int(tr), :]
        #ax.plot3D(pts[0], pts[1], pts[2], color='g', lw='0.1')
        #ax.scatter(cents[0], cents[1], cents[2], color='b')
        pts =points[[tri.simplices[:][[int(tr)]]]][:]
        pts=pts.reshape(4,3)
        #ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
        #ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
        #ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
        #ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
        #ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
        #ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')
        ind0=np.where((points==pts[0]).all(axis=1))
        ind0=np.array(ind0)[0][0]
        ind1=np.where((points==pts[1]).all(axis=1))
        ind1=np.array(ind1)[0][0]
        ind2=np.where((points==pts[2]).all(axis=1))
        ind2=np.array(ind2)[0][0]
        ind3=np.where((points==pts[3]).all(axis=1))
        ind3=np.array(ind3)[0][0]
        inds=np.vstack([inds,[ind0,ind1]])
        inds=np.vstack([inds,[ind0,ind2]])
        inds=np.vstack([inds,[ind0,ind3]])
        inds=np.vstack([inds,[ind1,ind2]])
        inds=np.vstack([inds,[ind1,ind3]])
        inds=np.vstack([inds,[ind2,ind3]])
        inds=inds.astype(int)
    return inds# inds are all the edges
    #ax.scatter(points[:,0], points[:,1], points[:,2], color='b')

def get_ordered_points(x_stereo,y_stereo,z_stereo,start,end):

    points=np.vstack((max(y_stereo)-y_stereo,x_stereo,z_stereo)).T

    new_points=points
    for point in points:
        new_points=np.append(new_points,[[point[0]+0.1,point[1],point[2]]],axis=0)
        new_points=np.append(new_points,[[point[0],point[1]+0.1,point[2]]],axis=0)
        new_points=np.append(new_points,[[point[0],point[1],point[2]+0.1]],axis=0)

    tri = Delaunay(new_points)

    vern=compute_voronoi_vertices_and_edges(new_points,r_thresh=20)
    #liste des points reliés entre eux
    vernedge=np.array(vern[1])
    vernedge2=vernedge[:,1]
    centers_vern = vern[0]
    tri_vern = vern[1]
    tetra_in=vern[2]

    
    inds=plot_vern(ax, centers_vern,new_points, tri,tetra_in)

    points=new_points

    kdt = cKDTree(points)
    dist0, idx0 = kdt.query(np.array(start))
    dist1, idx1 = kdt.query(np.array(end))
    edge_lengths = [np.linalg.norm(points[e[0], :] - points[e[1], :]) for e in inds]
    g = nx.Graph((i, j, {'weight': dist}) for (i, j), dist in zip(inds, edge_lengths))
    path_s = nx.shortest_path(g,source=idx0,target=idx1, weight='weight')
    
    return points[path_s],new_points

#-----------------------------------------------------------------------------------------------

