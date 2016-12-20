#!/usr/bin/python

from skimage import data, io, filters, feature
import matplotlib.pyplot as plt
import sys
from PIL import Image
import numpy as np
import osgeo
import gdal
import scipy.ndimage as nd
from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.externals import joblib
from sklearn.cluster import KMeans

# Get inputs for the path to the geotif and the average tree radius:

path_to_file  = raw_input("Please enter path to tif image: ")
im            = Image.open(path_to_file)
treeSize      = raw_input("Please enter average tree radius in orchard: ")
treeSize      = float(treeSize)

# Get Lat on Lon from geotiffs:

gtif          = gdal.Open(path_to_file)
geoprojection = gtif.GetGeoTransform()

# Convert image to numpy array:

im = np.array(im)
nx,ny,nbands=im.shape
im = im[:,:,0:3].reshape(-1, 3)     # discard 4th band (non colour) in tiff
                                    # and reshape image as a list of pixels

# Convert image using PCA:

print "converting using PCA..."
pca_model = joblib.load('./aero_pca_model.pkl')
im        = pca_model.transform(im)

# Segement using k-means on the color

print "segmenting using kmeans clustering on colour..."                                 # The model training code is in the attached notebook
clt_model = joblib.load('./aero_kmeans_model30_pca.pkl.gz')  # load a previously trained k-means model
im        = clt_model.predict(im)
im = im.reshape(nx,ny)
cluster_list = [4,6,8,16,17,19,21,24,29]     # These are the cluster labels with the best correlation to trees
c = (im==cluster_list[0] ).astype(int)
for i in cluster_list[1:]:
    c=c+ (im==i).astype(int)
c=np.uint8(c)
c *= 255                                     # Converts to black and white image

# Write intermediate result to file:

c_image=Image.fromarray(c)
c_image.save("./clustered_segments.tif")

# Fill holes in binary image:
 
large_objs=nd.binary_fill_holes(c)
large_objs=np.uint8(large_objs)
large_objs*= 255

# Smoothe image for noise reduction:

large_objs = nd.gaussian_filter(large_objs, sigma=0.5)
large_objs = (large_objs>10 ).astype(int)
large_objs=np.uint8(large_objs)
large_objs *= 255

# Find physical area covered by segments:

pixelsPerSquareMetre = 400                           # Found by inspection (should use osgeo lib for this)
avgPixelsPerTree = pixelsPerSquareMetre*3.141*(treeSize**2)
label_objects, nb_labels = nd.label(large_objs)      # labels every segment and counts number of segments
sizes = np.bincount(label_objects.ravel())           # measures the size of every segmentation
mask_sizes = sizes > (avgPixelsPerTree)/6            # filters out small segmentations
mask_sizes[0] = 0
large_objs = mask_sizes[label_objects]               # filters out small segmentations
large_objs=np.uint8(large_objs)
large_objs *= 255

label_objects, nb_labels = nd.label(large_objs)      # relabel after filtering

# Define depth as the number of nested lists:

depth = lambda L: isinstance(L, list) and max(map(depth, L))+1

# Get important geo info:

lat1      = geoprojection[3]
lon1      = geoprojection[0]
delta_lat = geoprojection[5]
delta_lon = geoprojection[1]

# Further segment the segments for overlapping trees using kmeans on the pixel position"

print "further segmenting using k-means clustering on pixel position"

def KMeansOnSegment(cluster,size,im):
    nClusters = int(size/avgPixelsPerTree)
    if nClusters==0:
        nClusters=1

    x=im
    x*=255

    clt = KMeans(n_clusters =nClusters)
    (a,b) = np.where(x==255)
    inputList=[list(x) for x in zip(a,b)]
    clt_model = clt.fit(inputList)
    
    answer = clt_model.predict(inputList)
    answer=zip(a,b,answer)
    answer=np.array([list(x) for x in answer])

    clusterCentres=[]
    for j in range(0,nClusters):
        avgy=0
        avgx=0
        count=0
        for i in range(0,len(answer)):
            if(answer[i][2]==j):
                avgx += answer[i][0]
                avgy += answer[i][1]
                count +=1
        avgx = avgx/count
        avgy = avgy/count
        
        avg_lat = ((avgx)*delta_lat)+lat1
        avg_lon = ((avgy)*delta_lon)+lon1

        rad = ((count/pixelsPerSquareMetre)/3.141)**0.5
        
        clusterCentres.append([avg_lat,avg_lon,rad])
    return clusterCentres

def GetAvgPosAndSize(i):
    cluster=np.where(label_objects == i)
    avg_lat = (np.mean(cluster[0])*delta_lat)+lat1
    avg_lon = (np.mean(cluster[1])*delta_lon)+lon1
    size    = np.bincount(label_objects.flatten())[i]
    if (size>2*avgPixelsPerTree):
        answer = KMeansOnSegment(cluster,size,np.uint8(label_objects==i))
        return answer
    else:
        rad = ((size/pixelsPerSquareMetre)/3.141)**0.5
        return [avg_lat,avg_lon,rad]

result=joblib.Parallel(n_jobs=7)\
  (joblib.delayed(GetAvgPosAndSize)(i)\
    for i in range(1,nb_labels))                  # mulithreaded, uses 7 processes. Increase for better cpu 

flatList=[]    
for item in result:                              # The previous result had to be flattened.
    if (depth(item)==1):                         # there is probably a better way of doing this..
        flatList.append(item)
    else:
        for item2 in item:
            flatList.append(item2)

# Write result to file:

np.savetxt("./tree_locations_and_sizes.csv", flatList, delimiter=",", header="lat,lon,radius",comments='')
