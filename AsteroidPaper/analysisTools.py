import numpy as np
import pandas as pd
import sqlite3 as sql
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import hstack
import matplotlib.pyplot as plt 
from astroML.plotting import hist
import scipy 
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
 

### selection tools and numerical analysis 
# robust standard deviation
def sigG(arr):
    return 0.741*(np.quantile(arr, 0.75)-np.quantile(arr, 0.25))
  
# hack for plotting with Joachim's old code
def calcStdDev(x, robust=True):
    return sigG(x)

def printStats(arr):
    print('           ', np.min(arr), np.mean(arr), np.median(arr), np.max(arr), np.size(arr)) 
    return 

def randomizePhotometry(df, sigErr, bands = ('u', 'g', 'r', 'i', 'z')):
    for b in bands:
        df[b] += norm(0, sigErr).rvs(np.size(df[b]))
    return 

def addErrors(grIn, riIn, izIn, err = 0.02):
    grOut = grIn + np.random.normal(0, err, np.size(grIn))
    riOut = riIn + np.random.normal(0, err, np.size(riIn))
    izOut = izIn + np.random.normal(0, err, np.size(izIn))
    acolorOut = 0.89 * grOut + 0.45 * riOut - 0.57
    return grOut, riOut, izOut, acolorOut


# return standard and robust stats 
def robustStats(arr):
    Andata = np.size(arr)
    Amin=np.min(arr)
    Amax=np.max(arr)
    Amean = np.mean(arr)
    Amedian = np.median(arr)
    Astd = np.std(arr)
    AsigG = sigG(arr)
    # the fraction of sample within 2*sigG from the median (95% for Gauss)
    xx = arr[(arr>Amedian-2*AsigG)&(arr<Amedian+2*AsigG)]
    frac = np.size(xx)/Andata
    print('size, range:', Andata, Amin, Amax)
    print('mean, median:', Amean, Amedian)
    print('std, sigG:', Astd, AsigG)
    print('fraction in median+-2*sigG:', frac)
    return

# asteroid specific code

# assign colors for each asteroid 
def assignColors(moc):
    moc["u-g"] = moc["u"] - moc["g"]
    moc["g-r"] = moc["g"] - moc["r"]
    moc["r-i"] = moc["r"] - moc["i"]
    moc["i-z"] = moc["i"] - moc["z"]
    moc["a_color"] = 0.89 * moc["g-r"] + 0.45 * moc["r-i"] - 0.57
    return

# assign a region depending on where falls on axis
# the version from the MMI paper
def assignRegionMMI(a_color, i_z):
    region = np.ones_like(a_color)
    region = np.where((a_color > 0) & (i_z > (a_color - 0.05)), 2, region)
    region = np.where((a_color > 0) & (i_z < (a_color - 0.05)), 3, region)
    return region

# assign a region depending on where falls on axis
def assignRegion(a_color, i_z):
    region = np.ones_like(a_color)
    region = np.where((a_color > 0) & (i_z > (a_color - 0.05)), 2, region)
    region = np.where((a_color > 0) & (i_z <= (a_color - 0.05)), 3, region)
    region = np.where(((a_color > 0) & (i_z > (a_color - 0.05))) & (i_z < 0.08), 4, region)
    return region
 
# assign a region depending on where falls on axis
# an extension of assignRegion to more regions
def assignRegionOverfit(a_color, i_z):
    region = np.ones_like(a_color)
    region = np.where((a_color > 0) & (i_z > (a_color - 0.05)), 2, region)
    region = np.where((a_color > 0) & (i_z <= (a_color - 0.05)), 3, region)
    region = np.where(((a_color > 0) & (i_z > (a_color - 0.05)) & (i_z < 0.08)), 4, region)
    region = np.where(((a_color > -0.05) & (a_color <= 0) & (np.abs(i_z) < 0.3)), 5, region)
    return region

def assignConstAlbedo(df, regionFlag, nReg, pVarray):
    pV = np.ones_like(regionFlag)
    for i in range(0, nReg):
        print(pVarray[i])
        pV = np.where(regionFlag == i+1, pVarray[i], pV)  
    return pV

def assignModelAlbedo(df, modelType = "constRegAlbedo", colName = "pVmodel1"):
    # as published in the MMI paper
    if (modelType == "constRegAlbedoMMI"):
        regionFlag = assignRegionMMI(df["a_color"], df["i-z"])
        df['region'] = regionFlag
        pVarray = 1.0*np.arange(3)
        for i in range(0, 3):
            dataReg = df[regionFlag == i+1]
            pVarray[i] =  np.median(dataReg['pV'])
        df[colName] = assignConstAlbedo(df, regionFlag, 3, pVarray)

    # improved version from the II paper
    if (modelType == "constRegAlbedo"):
        regionFlag = assignRegion(df["a_color"], df["i-z"])
        df['region'] = regionFlag
        pVarray = 1.0*np.arange(4)
        for i in range(0, 4):
            dataReg = df[regionFlag == i+1]
            pVarray[i] =  np.median(dataReg['pV'])
        df[colName] = assignConstAlbedo(df, regionFlag, 4, pVarray)

    # an attempt to further improve the version from the II paper
    if (modelType == "constRegAlbedoOverfit"):
        regionFlag = assignRegionOverfit(df["a_color"], df["i-z"])
        df['region'] = regionFlag
        pVarray = 1.0*np.arange(5)
        for i in range(0, 5):
            dataReg = df[regionFlag == i+1]
            pVarray[i] =  np.median(dataReg['pV'])
        df[colName] = assignConstAlbedo(df, regionFlag, 5, pVarray)

    return 


# given a list of coordinates for a sample of points, and 
# coordinates of a particular point in the same space,
# return distances and indices to Nnn nearest neighbors
def findNnn(coords, point, Nnn):
    dSquare = 0*coords[0]
    indices = np.linspace(0,np.size(dSquare))
    for coordAll, coordPoint in zip(coords, point):
        dSquare += (coordAll-coordPoint)*(coordAll-coordPoint)
    # sort in ascending order
    indices = np.argsort(dSquare)
    return np.sqrt(dSquare[indices[0:Nnn+1]]), indices[0:Nnn]

# given a list of coordinates for a sample of points,  
# assign a model value for pV to each point as 
# the median value of pV for Nnn nearest neighbors
# return vector of model values and sigG for sqrt(pV/pVmodel)
def getpVmodel(coords, pV, Nnn, statType):
    pVmodel = 0*pV
    for j in range(0, np.size(pV)):
        myPoint = (coords[0][j], coords[1][j], coords[2][j])
        # because myPoint is included in myCoords, ask for
        # Nnn+1 and later discard the first value
        distances, indices = findNnn(coords, myPoint, Nnn+1)
        if (statType==0):
            pVmodel[j] = np.median(pV[indices[1:]])
        elif (statType==1):
            pVmodel[j] = np.mean(pV[indices[1:]])
        else: 
            pVmodel[j] = np.average(pV[indices[1:]])
    Drat = np.sqrt(pV/pVmodel)
    return pVmodel, sigG(Drat)

 
# given a list of coordinates for a sample of points,  
# assign a model value for pV to each point as 
# the median value of pV for Nnn nearest neighbors
# return vector of model values and sigG for sqrt(pV/pVmodel)
def getpVmodelSample(coords, pV, coordsSample, pVSample, Nnn, statType):
    pVmodel = 0*pV
    for j in range(0, np.size(pV)):
        myPoint = (coords[0][j], coords[1][j], coords[2][j])
        distances, indices = findNnn(coordsSample, myPoint, Nnn)
        if (statType==0):
            pVmodel[j] = np.median(pVSample[indices])
        elif (statType==1):
            pVmodel[j] = np.mean(pVSample[indices])
        else:
            pVmodel[j] = np.average(pVSample[indices])
    Drat = np.sqrt(pV/pVmodel)
    return pVmodel, sigG(Drat)
 

def doGMM(df, attributes, components, classVecName, nClassMemberMin=3):

    # setup of GMM's data structures 
    Xarrays = []
    for attr in attributes:
        Xarrays.append(np.vstack([df[a] for a in attr]).T)
        
    # call the workhorse, clfs is a list with all results 
    clfs = []
    for attr, X in zip(attributes, Xarrays):
        clfs_i = []
        for comp in components:
            print("  - {0} component fit".format(comp))
            clf = GaussianMixture(comp, covariance_type='full',
                      random_state=0, max_iter=500)
            clf.fit(X)
            clfs_i.append(clf)

            if not clf.converged_:
                print("           NOT CONVERGED!")
        clfs.append(clfs_i)
 
    # Grab the best classifier, based on the BIC
    i = 0
    X = Xarrays[i]
    BIC = [c.bic(X) for c in clfs[i]]
    i_best = np.argmin(BIC)
    # clf is the GMM data structure 
    clf = clfs[i][i_best]
    n_components = clf.n_components
    class_labels = []
    
    # predict class for each data point
    GMMc = clf.predict(X)
    df[classVecName] = GMMc
    # assign 
    classes = np.unique(GMMc)
    class_labels.append(GMMc)
    
    # compute class stats
    counts = np.sum(GMMc == classes[:, None], 1)
    size = np.array([np.linalg.det(C) for C in clf.covariances_])
    weights = clf.weights_
    density = counts / size
    density[counts < nClassMemberMin] = 0
    isort = np.argsort(density)[::-1]
    Cmeans = []
    Cstdevs = []
    Ccounts = []
    attributes0 = attributes[0]
    nGMMclass = 0 
    for j in range(n_components):
        if (counts[isort[j]]>=nClassMemberMin):
            nGMMclass += 1
            flag = (GMMc == isort[j])
            Ccounts.append(np.sum(flag))
            for n in attributes0:
                vec = df[n]
                classvec = vec[flag]
                Cmeans.append(np.mean(classvec))
                Cstdevs.append(np.std(classvec))
    
    # return the number of components and their stats
    return nGMMclass, Ccounts, Cmeans, Cstdevs 
