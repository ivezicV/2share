import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
import analysisTools as vat 



### plots 
# quick plot 
def qp(d, Xstr, Xmin, Xmax, Ystr, Ymin, Ymax):
    ax = plt.axes()
    ax.set_xlabel(Xstr)
    ax.set_ylabel(Ystr)
    ax.scatter(d[Xstr], d[Ystr], s=0.01, c='blue')  
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    plt.show()
    return

# quick plot - compare three subsamples
def qp3(d1, d2, d3, Xstr, Xmin, Xmax, Ystr, Ymin, Ymax):
    ax = plt.axes()
    ax.set_xlabel(Xstr)
    ax.set_ylabel(Ystr)
    ax.scatter(d1[Xstr], d1[Ystr], s=0.01, c='green') 
    ax.scatter(d2[Xstr], d2[Ystr], s=0.01, c='red') 
    ax.scatter(d3[Xstr], d3[Ystr], s=0.01, c='blue') 
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    plt.show()
    return

# quick plot - binned median
def qpBM(d, Xstr, Xmin, Xmax, Ystr, Ymin, Ymax, nBin, Nsigma=3, offset=0.01):
         
    print('medianAll:', np.median(d[Ystr]), 'std.dev.All:', sigG(d[Ystr]))
    print('N=', np.size(d[Ystr]), 'min=', np.min(d[Ystr]), 'max=', np.max(d[Ystr]))

    ax = plt.axes()
    ax.scatter(d[Xstr], d[Ystr], s=0.01, c='black') 
    # binning
    xBinM, nPtsM, medianBinM, sigGbinM = fitMedians(d[Xstr], d[Ystr], Xmin, Xmax, nBin, 1)
    # plotting
    ax.scatter(xBinM, medianBinM, s=30.0, c='black', alpha=0.8)
    ax.scatter(xBinM, medianBinM, s=15.0, c='yellow', alpha=0.3)
    #
    TwoSigP = medianBinM + Nsigma*sigGbinM
    TwoSigM = medianBinM - Nsigma*sigGbinM 
    ax.plot(xBinM, TwoSigP, c='yellow')
    ax.plot(xBinM, TwoSigM, c='yellow')
    #
    rmsBin = np.sqrt(nPtsM) / np.sqrt(np.pi/2) * sigGbinM
    rmsP = medianBinM + rmsBin
    rmsM = medianBinM - rmsBin
    ax.plot(xBinM, rmsP, c='cyan')
    ax.plot(xBinM, rmsM, c='cyan')
    # 
    xL = np.linspace(-100,100)
    ax.plot(xL, 0*xL, c='red')
    ax.plot(xL, 0*xL+offset, '--', c='red')
    ax.plot(xL, 0*xL-offset, '--', c='red')
    # 
    ax.set_xlabel(Xstr)
    ax.set_ylabel(Ystr)
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    plt.show()
    return
 
def qphist(arr, xMin, xMax, xLabel, verbose = False):
    ax = plt.axes()
    hist(arr, bins='knuth', ax=ax, histtype='stepfilled', ec='k', fc='#AAAAAA')
    ax.set_xlabel(xLabel)
    ax.set_ylabel('n')
    ax.set_xlim(xMin, xMax)
    plt.show()
    if (verbose):
        print('Min, max: ', np.min(arr),np.max(arr)) 
        print('Mean, median: ', np.mean(arr),np.median(arr)) 
        print('sigG, st.dev.: ', sigG(arr),np.std(arr)) 
    return 

def qpH0(arr, xMin, xMax, xLabel, nBins=0, verbose = False):
    ax = plt.axes()
    if (nBins>0):
        hist, bins = np.histogram(arr, bins=nBins)
        center = (bins[:-1]+bins[1:])/2
        ax.plot(center, hist, drawstyle='steps', c='blue')   
    else:
        plt.hist(arr, bins='auto', histtype='stepfilled', ec='k', fc='red') 
 
    ax.set_xlabel(xLabel)
    ax.set_ylabel('n')
    ax.set_xlim(xMin, xMax)
    ax.plot([-1000, 1000], [0, 0], '--k')
    plt.show()
    if (verbose):
        print('Min, max: ', np.min(arr),np.max(arr)) 
        print('Mean, median: ', np.mean(arr),np.median(arr)) 
        print('sigG, st.dev.: ', sigG(arr),np.std(arr))  
    return 

def qpHdm(d, band, dmMax, xMin, xMax, nBins=50, verbose=False):
    str = 'd' + band
    dm = 1000*d[str]
    dmOK = dm[np.abs(dm)<(1000*dmMax)]
    xLabel = str + ' (milimag)'
    qpH0(dmOK, 1000*xMin, 1000*xMax, xLabel, nBins, verbose)
    return np.mean(dmOK), np.median(dmOK), sigG(dmOK)


def qp2hist(arr1, arr2, xMin, xMax, xLabel, nBins=0, verbose = False):
    ax = plt.axes()
    if (nBins>0):
        hist, bins = np.histogram(arr1, bins=nBins)
        center = (bins[:-1]+bins[1:])/2
        ax.plot(center, hist, drawstyle='steps', c='red')   
        hist2, bins2 = np.histogram(arr2, bins=nBins)
        center2 = (bins2[:-1]+bins2[1:])/2
        ax.plot(center2, hist2, drawstyle='steps', c='blue')   
    else:
        plt.hist(arr1, bins='auto', histtype='stepfilled', ec='k', fc='yellow')
        plt.hist(arr2, bins='auto', histtype='stepfilled', ec='red', fc='blue')
 
    ax.set_xlabel(xLabel)
    ax.set_ylabel('n')
    ax.set_xlim(xMin, xMax)
    plt.show()
    if (verbose):
        print('Min: ', np.min(arr1),np.min(arr2)) 
        print('Median: ', np.median(arr1),np.median(arr2)) 
        print('sigG: ', sigG(arr1),sigG(arr2)) 
        print('Max: ', np.max(arr1),np.max(arr2)) 
    return 


# In[334]:


# copied from ATM

def plotHist(ax, xValues, xRange, 
             bins=10, 
             numGauss=1, 
             robust=True, 
             swapAxes=False,
             useMedian=True,
             statRange=None,
             histKwargs={
                "histtype" : "stepfilled",
                "alpha" : 0.5, 
                "normed" : True}, 
             plotKwargs={
                "ls" : "-",
                "c" : "red"}, 
             plotKwargsComponents={
                "ls" : ":",
                "c" : "red",
                "alpha" : 0.5}):
    
    xValues_cut = xValues[(xValues > xRange[0]) & (xValues < xRange[1])]
    if statRange is not None:
        xValues_stat = xValues_cut[(xValues_cut > statRange[0]) & (xValues_cut < statRange[1])]
    else:
        xValues_stat = xValues_cut
    print("{} values are outside the defined minimum and maximum.".format(len(xValues) - len(xValues_cut)))
    
    if statRange is not None:
        print("Statistics will be calculated between {} and {}".format(statRange[0], statRange[1]))
    
    ax.hist(xValues, bins=np.linspace(xRange[0], xRange[1], bins), **histKwargs)
    
    stats = []
    if numGauss > 0:
        if numGauss == 1:
            stats = np.zeros(3)
            w = 1
            if useMedian is True:
                mu = np.median(xValues_stat)
            else:
                mu = np.mean(xValues_stat)
                
            sigma = calcStdDev(xValues_stat, robust=robust)
            grid = np.linspace(*xRange, 1000)
            gauss = norm(mu, sigma).pdf(grid)
            if swapAxes is True:
                ax.plot(gauss, grid, **plotKwargs)
            else:
                ax.plot(grid, gauss, **plotKwargs)

            stats[0] = mu
            stats[1] = sigma
            stats[2] = w

        else:
            # Fit GMM
            gmm = GaussianMixture(n_components=numGauss, covariance_type="full", tol=0.00000001)
            gmm = gmm.fit(X=np.expand_dims(xValues_stat, 1))

            # Evaluate GMM
            gmm_x = np.linspace(*xRange, 1000)
            gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1))) 
            
            if swapAxes is True:
                ax.plot(gmm_y, gmm_x, **plotKwargs)
            else:
                ax.plot(gmm_x, gmm_y, **plotKwargs)

            mu = gmm.means_.ravel()
            sigma = np.sqrt(gmm.covariances_.ravel())
            w = gmm.weights_.ravel()

            for i in range(numGauss):
                grid = np.linspace(*xRange, 1000)
                gauss = norm(mu[i], sigma[i]).pdf(grid)
                ax.plot(grid, w[i] * gauss, **plotKwargsComponents)

            stats = np.zeros((numGauss, 3))
            stats[:, 0] = mu
            stats[:, 1] = sigma
            stats[:, 2] = w
        
    return ax, stats

# ax[1,0] histogram function
def plotRegHist2(ax, data_pVreg, color, histKwargs, numGauss=1):
    ax, stats = plotHist(ax,
                         data_pVreg, 
                          [0, 2.5], 
                          numGauss=numGauss,
                          bins=50,
                          histKwargs=histKwargs,
                          plotKwargs={'ls': '-', 'c': color},
                          plotKwargsComponents={'ls': ':', 'c': color, 'alpha': 0.5})
    ax.set_xlabel("$p_V/p_V^{MMI}$")
    ax.set_ylabel("n")
    ax.set_xlim(0.0, 2.5)
    if (numGauss>1):
        for gauss in stats:
            print("mu : {:.3f}".format(gauss[0]))
            print("sigma : {:.3f}".format(gauss[1]))
            print("weight : {:.3f}".format(gauss[2]))
            print("")
    else:
        print("mu : {:.3f}".format(stats[0]))
        print("sigma : {:.3f}".format(stats[1]))
        print("weight : {:.3f}".format(stats[2]))
         
    return stats



# robust standard deviation
def sigG(arr):
    return 0.741*(np.quantile(arr, 0.75)-np.quantile(arr, 0.25))
  
# hack for plotting with Joachim's old code
def calcStdDev(x, robust=True):
    return sigG(x)





def quadPlotsOld(reg1, reg2, reg3, reg4, data, Drat):
    DPI = 300
    SAVE_DIR = "../plots/"
    FORMAT = "png"
    SAVE_FIGS = False


    histKwargs = {
        "histtype" : "stepfilled",
        "color" : "#718CA1",
        "alpha" : 0.8, 
        "normed" : True,
    }
    plotKwargs = {
        "ls" : "-",
        "lw" : 1,
        "c" : "red"
    }

    plotKwargsComponents = {
        "ls" : ":",
        "lw" : 1,
        "c" : "red",
        "alpha" : 0.8
    }

    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }

    pVreg1 = np.median(reg1['pV'])
    pVreg2 = np.median(reg2['pV'])
    pVreg3 = np.median(reg3['pV'])
    pVreg4 = np.median(reg4['pV'])



    fig, ax = plt.subplots(2, 2, dpi=DPI, figsize=(5.5, 5.5))
    fig.subplots_adjust(
        # the left side of the subplots of the figure
        left=0.05,  
        # the right side of the subplots of the figure
        right=0.95,
        # the bottom of the subplots of the figure
        bottom=0.05,
        # the top of the subplots of the figure
        top=0.95,
        # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        wspace=0.25,   
        # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        hspace=0.55)   

    cbar_ax = fig.add_axes([0.05, 0.49, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,0].scatter(data["a_color"],
                         data["i-z"],
                         c=data["pV"],
                         s=1,
                         vmin=0.0,
                         vmax=0.5,
                         cmap="Spectral")
    ax[0,0].vlines([0], -0.2, 0.3, **verticalLines)
    xgrid = np.linspace(-0, 0.15, 20)
    modelo = 0.1 + 0.0*xgrid
    ax[0,0].plot(xgrid, modelo, c='k', lw=1, ls="--")
    a_grid = np.linspace(-0, 0.3, 20)
    iz_model = -0.05 + 1.0*a_grid
    ax[0,0].plot(a_grid,iz_model, c='k', lw=1, ls="--")
    ax[0,0].set_ylabel("$i-z$")
    ax[0,0].set_xlabel("$a$")
    ax[0,0].set_xlim(-0.3, 0.3)
    ax[0,0].set_ylim(-0.2, 0.3)
    ax[0,0].text(-0.28, -0.18, pVreg1)
    ax[0,0].text(0.10, 0.26, pVreg2)
    ax[0,0].text(0.10, -0.18, pVreg3)
    ax[0,0].text(0.10, -0.18, pVreg4)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_V$", 
                 orientation="horizontal")

    cbar_ax = fig.add_axes([0.55, 0.49, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,1].scatter(data["a_color"],
                         data["i-z"],
                         c=data['pVrat'],
                         s=0.1,
                         vmin=0.2,
                         vmax=1.8,
                         cmap="Spectral")
    ax[0,1].vlines([0], -0.2, 0.3, **verticalLines)
    xgrid = np.linspace(-0, 0.15, 20)
    modelo = 0.1 + 0.0*xgrid
    ax[0,1].plot(xgrid, modelo, c='k', lw=1, ls="--")
    a_grid = np.linspace(-0, 0.3, 20)
    iz_model = -0.05 + 1.0*a_grid
    ax[0,1].plot(a_grid,iz_model, c='k', lw=1, ls="--")
    ax[0,1].set_ylabel("$i-z$")
    ax[0,1].set_xlabel("$a$")
    ax[0,1].set_xlim(-0.3, 0.3)
    ax[0,1].set_ylim(-0.2, 0.3)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_{V}/p_V^{model}$", 
                 orientation="horizontal")


    ### histogram panels

    # ax[1,0] histogram
    stats1 = plotRegHist2(ax[1,0], reg1['pVrat'], 'red', histKwargs)
    stats2 = plotRegHist2(ax[1,0], reg2['pVrat'], 'green', histKwargs )
    stats3 = plotRegHist2(ax[1,0], reg3['pVrat'], 'blue', histKwargs)
    stats4 = plotRegHist2(ax[1,0], reg4['pVrat'], 'purple', histKwargs)


    ax[1,1], stats = plotHist(ax[1,1], 
                              Drat, 
                              [0.4, 1.6], 
                              numGauss=1,
                              bins=50,
                              histKwargs=histKwargs,
                              plotKwargs=plotKwargs,
                              plotKwargsComponents=plotKwargsComponents)
    ax[1,1].set_xlabel("$D^{SDSS} / D^{ATM}$")
    ax[1,1].set_ylabel("n")#
    ax[1,1].set_xlim(0.4, 1.6)
    ax[1,1].set_ylim(0, 3)
    ax[1,1].set_xticks(np.arange(0.4, 1.8, 0.2))
    ax[1,1].set_yticks(np.arange(0, 3.5, 0.5))
    for gauss in [stats]:
        print("mu : {:.3f}".format(gauss[0]))
        print("sigma : {:.3f}".format(gauss[1]))
        print("weight : {:.3f}".format(gauss[2]))
        print("")
        ax[1,1].text(1.22, 2.75, r"$\mu$: {:.3f}".format(stats[0]))
        ax[1,1].text(1.22, 2.50, r"$\sigma_G$: {:.3f}".format(stats[1]))
    
    plt.show()


# given vectors x and y, fit medians in bins from xMin to xMax, with Nbin steps,
# and return xBin, medianBin, medianErrBin 
def fitMedians(x, y, xMin, xMax, Nbin, verbose=1): 

    # first generate bins
    xEdge = np.linspace(xMin, xMax, (Nbin+1)) 
    xBin = np.linspace(0, 1, Nbin)
    nPts = 0*np.linspace(0, 1, Nbin)
    medianBin = 0*np.linspace(0, 1, Nbin)
    sigGbin = -1+0*np.linspace(0, 1, Nbin) 
    for i in range(0, Nbin): 
        xBin[i] = 0.5*(xEdge[i]+xEdge[i+1]) 
        yAux = y[(x>xEdge[i])&(x<=xEdge[i+1])]
        if (yAux.size > 0):
            nPts[i] = yAux.size
            medianBin[i] = np.median(yAux)
            # robust estimate of standard deviation: 0.741*(q75-q25)
            sigmaG = 0.741*(np.percentile(yAux,75)-np.percentile(yAux,25))
            # uncertainty of the median: sqrt(pi/2)*st.dev/sqrt(N)
            sigGbin[i] = np.sqrt(np.pi/2)*sigmaG/np.sqrt(nPts[i])
        else:
            nPts[i] = 0
            medianBin[i] = 0
            sigGbin[i] = 0
        
    if (verbose):
        print('median:', np.median(medianBin[nPts>0]), 'std.dev:', np.std(medianBin[nPts>0]))

    return xBin, nPts, medianBin, sigGbin




def drawXDellipse(clf, ax):
    for i in range(clf.n_components):
        draw_ellipse(clf.mu[i], clf.V[i], scales=[2], ax=ax, ec='k', fc='gray', alpha=0.2)


def plotRegions002(ax):
    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }    
 
    ax.vlines([0.02], -0.2, 0.3, **verticalLines)
    ax.vlines([0.02], -0.03, 0.08, **verticalLines)

    xgrid = np.linspace(0.02, 0.13, 20)
    modelo = 0.08 + 0.0*xgrid
    ax.plot(xgrid, modelo, c='k', lw=1, ls="--")

    a_grid = np.linspace(0.02, 0.3, 20)
    iz_model = -0.05 + 1.0*a_grid
    ax.plot(a_grid,iz_model, c='k', lw=1, ls="--")
    return


def plotRegions(ax):
    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }    
 
    ax.vlines([0.0], -0.8, 0.8, **verticalLines)
    # ax.vlines([0.0], -0.03, 0.08, **verticalLines)

    xgrid = np.linspace(0.0, 0.13, 20)
    modelo = 0.08 + 0.0*xgrid
    ax.plot(xgrid, modelo, c='k', lw=1, ls="--")

    a_grid = np.linspace(0.0, 0.3, 20)
    iz_model = -0.05 + 1.0*a_grid
    ax.plot(a_grid,iz_model, c='k', lw=1, ls="--")
    return



def quadPlotsOrigColors(data, pVrat, drawXD = False, clf = 0):
    DPI = 300
    SAVE_DIR = "../plots/"
    FORMAT = "png"
    SAVE_FIGS = True 


    histKwargs = {
        "histtype" : "stepfilled",
        "color" : "#718CA1",
        "alpha" : 0.8, 
        "normed" : True,
    }
    plotKwargs = {
        "ls" : "-",
        "lw" : 1,
        "c" : "red"
    }

    plotKwargsComponents = {
        "ls" : ":",
        "lw" : 1,
        "c" : "red",
        "alpha" : 0.8
    }

    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }

    fig, ax = plt.subplots(2, 2, dpi=DPI, figsize=(5.5, 5.5))
    fig.subplots_adjust(
        # the left side of the subplots of the figure
        left=0.13,  
        # the right side of the subplots of the figure
        right=0.95,
        # the bottom of the subplots of the figure
        bottom=0.18,
        # the top of the subplots of the figure
        top=0.95,
        # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        wspace=0.38,   
        # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        hspace=0.58)   

    cbar_ax = fig.add_axes([0.13, 0.56, 0.34, 0.015])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,0].scatter(data["g-r"],
                         data["r-i"],
                         c=data["pV"],
                         s=0.5,
                         vmin=0.0,
                         vmax=0.3,
                         cmap="Spectral")
    ax[0,0].set_ylabel("$g-r$")
    ax[0,0].set_xlabel("$r-i$")
    ax[0,0].set_xlim(0.3, 0.9)
    ax[0,0].set_ylim(-0.1, 0.4)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_V$", 
                 orientation="horizontal")


    cbar_ax = fig.add_axes([0.61, 0.56, 0.34, 0.015])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,1].scatter(data["a_color"],
                         data["i-z"],
                         c=data["pV"],
                         s=0.5,
                         vmin=0.0,
                         vmax=0.3,
                         cmap="Spectral")

    plotRegions(ax[0,1])
    ax[0,1].set_ylabel("$i-z$")
    ax[0,1].set_xlabel("$a$")
    ax[0,1].set_xlim(-0.3, 0.3)
    ax[0,1].set_ylim(-0.2, 0.3)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_{V}$", 
                 orientation="horizontal")
    if (drawXD):
        drawXDellipse(clf, ax[0,1])


    cbar_ax = fig.add_axes([0.13, 0.09, 0.34, 0.015]) 
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[1,0].scatter(data["a_color"],
                         data["i-z"],
                         c=pVrat,
                         s=1.0,
                         vmin=0.5,
                         vmax=1.5,
                         cmap="Spectral")
    plotRegions(ax[1,0])
    ax[1,0].set_ylabel("$i-z$")
    ax[1,0].set_xlabel("$a$")
    ax[1,0].set_xlim(-0.3, 0.3)
    ax[1,0].set_ylim(-0.2, 0.3)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_{V}/p_V^{model}$", 
                 orientation="horizontal")


    ax[1,1], stats = plotHist(ax[1,1], 
                              np.sqrt(pVrat), 
                              [0.4, 1.6], 
                              numGauss=1,
                              bins=50,
                              histKwargs=histKwargs,
                              plotKwargs=plotKwargs,
                              plotKwargsComponents=plotKwargsComponents)
    ax[1,1].set_xlabel("$D^{SDSS} / D^{ATM}$")
    ax[1,1].set_ylabel("n")#
    ax[1,1].set_xlim(0.4, 1.6)
    ax[1,1].set_ylim(0, 3)
    ax[1,1].set_xticks(np.arange(0.4, 1.8, 0.2))
    ax[1,1].set_yticks(np.arange(0, 3.5, 0.5))
    for gauss in [stats]:
        print("mu : {:.3f}".format(gauss[0]))
        print("sigma : {:.3f}".format(gauss[1]))
        print("weight : {:.3f}".format(gauss[2]))
        print("")
        vv = np.sqrt(pVrat)
        median = np.median(vv)
        sG = sigG(vv)
        xx = vv[(vv>median-2*sG)&(vv<median+2*sG)]
        f95 = 100*np.size(xx)/np.size(vv)
        ax[1,1].text(1.18, 2.75, r"$\mu$: {:.3f}".format(median))
        ax[1,1].text(1.18, 2.50, r"$\sigma_G$: {:.3f}".format(sG))
        ax[1,1].text(1.18, 2.23, r"$f95$: {:.1f}".format(f95))
    
    plt.savefig('quadPlotsOrigColors.png')
    plt.show()




## analogous to quadPlotsOrigColors but instead of data frame as the first input
## this version takes four vectors: gr, ri, iz, pV
def quadPlotsOrigColors2(gr, ri, iz, pV, pVrat, drawXD = False, clf = 0):
    DPI = 300
    SAVE_DIR = "../plots/"
    FORMAT = "png"
    SAVE_FIGS = False

    acolor = 0.89 * gr + 0.45 * ri - 0.57
    Drat = np.sqrt(pVrat)

    histKwargs = {
        "histtype" : "stepfilled",
        "color" : "#718CA1",
        "alpha" : 0.8, 
        "normed" : True,
    }
    plotKwargs = {
        "ls" : "-",
        "lw" : 1,
        "c" : "red"
    }

    plotKwargsComponents = {
        "ls" : ":",
        "lw" : 1,
        "c" : "red",
        "alpha" : 0.8
    }

    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }

    fig, ax = plt.subplots(2, 2, dpi=DPI, figsize=(5.5, 5.5))
    fig.subplots_adjust(
        # the left side of the subplots of the figure
        left=0.05,  
        # the right side of the subplots of the figure
        right=0.95,
        # the bottom of the subplots of the figure
        bottom=0.05,
        # the top of the subplots of the figure
        top=0.95,
        # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        wspace=0.35,   
        # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        hspace=0.55)   

    cbar_ax = fig.add_axes([0.05, 0.49, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,0].scatter(gr,
                         ri,
                         c=pV,
                         s=0.5,
                         vmin=0.0,
                         vmax=0.3,
                         cmap="Spectral")
    ax[0,0].set_ylabel("$g-r$")
    ax[0,0].set_xlabel("$r-i$")
    ax[0,0].set_xlim(0.3, 0.9)
    ax[0,0].set_ylim(-0.1, 0.4)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_V$", 
                 orientation="horizontal")


    cbar_ax = fig.add_axes([0.55, 0.49, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,1].scatter(acolor,
                         iz,
                         c=pV,
                         s=0.5,
                         vmin=0.0,
                         vmax=0.3,
                         cmap="Spectral")

    plotRegions(ax[0,1])
    ax[0,1].set_ylabel("$i-z$")
    ax[0,1].set_xlabel("$a$")
    ax[0,1].set_xlim(-0.3, 0.3)
    ax[0,1].set_ylim(-0.2, 0.3)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_{V}$", 
                 orientation="horizontal")
    if (drawXD):
        drawXDellipse(clf, ax[0,1])


    cbar_ax = fig.add_axes([0.05, -0.06, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[1,0].scatter(acolor,
                         iz,
                         c=pVrat,
                         s=0.3,
                         vmin=0.5,
                         vmax=1.5,
                         cmap="Spectral")
    plotRegions(ax[1,0])
    ax[1,0].set_ylabel("$i-z$")
    ax[1,0].set_xlabel("$a$")
    ax[1,0].set_xlim(-0.3, 0.3)
    ax[1,0].set_ylim(-0.2, 0.3)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_{V}/p_V^{model}$", 
                 orientation="horizontal")


    ax[1,1], stats = plotHist(ax[1,1], 
                              Drat, 
                              [0.4, 1.6], 
                              numGauss=1,
                              bins=50,
                              histKwargs=histKwargs,
                              plotKwargs=plotKwargs,
                              plotKwargsComponents=plotKwargsComponents)
    ax[1,1].set_xlabel("$D^{SDSS} / D^{ATM}$")
    ax[1,1].set_ylabel("n")#
    ax[1,1].set_xlim(0.4, 1.6)
    ax[1,1].set_ylim(0, 3)
    ax[1,1].set_xticks(np.arange(0.4, 1.8, 0.2))
    ax[1,1].set_yticks(np.arange(0, 3.5, 0.5))
    for gauss in [stats]:
        print("mu : {:.3f}".format(gauss[0]))
        print("sigma : {:.3f}".format(gauss[1]))
        print("weight : {:.3f}".format(gauss[2]))
        print("")
        ax[1,1].text(1.22, 2.75, r"$\mu$: {:.3f}".format(stats[0]))
        ax[1,1].text(1.22, 2.50, r"$\sigma_G$: {:.3f}".format(stats[1]))
    
    plt.show()
    
    


## similar to quadPlotsOrigColors2 but instead of histogram, it has two 
## rows of color diagrams - with the data and data/model 
def quadPlotsOrigColors3(gr, ri, iz, pV, pVrat, drawXD = False, clf = 0):
    DPI = 300
    SAVE_DIR = "../plots/"
    FORMAT = "png"
    SAVE_FIGS = False

    acolor = 0.89 * gr + 0.45 * ri - 0.57
 
    histKwargs = {
        "histtype" : "stepfilled",
        "color" : "#718CA1",
        "alpha" : 0.8, 
        "normed" : True,
    }
    plotKwargs = {
        "ls" : "-",
        "lw" : 1,
        "c" : "red"
    }

    plotKwargsComponents = {
        "ls" : ":",
        "lw" : 1,
        "c" : "red",
        "alpha" : 0.8
    }

    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }

    fig, ax = plt.subplots(2, 2, dpi=DPI, figsize=(5.5, 5.5))
    fig.subplots_adjust(
        # the left side of the subplots of the figure
        left=0.05,  
        # the right side of the subplots of the figure
        right=0.95,
        # the bottom of the subplots of the figure
        bottom=0.05,
        # the top of the subplots of the figure
        top=0.95,
        # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        wspace=0.35,   
        # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        hspace=0.55)   

    cbar_ax = fig.add_axes([0.05, 0.49, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,0].scatter(gr,
                         ri,
                         c=pV,
                         s=0.5,
                         vmin=0.0,
                         vmax=0.3,
                         cmap="Spectral")
    ax[0,0].set_ylabel("$g-r$")
    ax[0,0].set_xlabel("$r-i$")
    ax[0,0].set_xlim(0.3, 0.9)
    ax[0,0].set_ylim(-0.1, 0.4)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_V$", 
                 orientation="horizontal")


    cbar_ax = fig.add_axes([0.55, 0.49, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,1].scatter(acolor,
                         iz,
                         c=pV,
                         s=0.5,
                         vmin=0.0,
                         vmax=0.3,
                         cmap="Spectral")

    plotRegions(ax[0,1])
    ax[0,1].set_ylabel("$i-z$")
    ax[0,1].set_xlabel("$a$")
    ax[0,1].set_xlim(-0.3, 0.3)
    ax[0,1].set_ylim(-0.2, 0.3)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_{V}$", 
                 orientation="horizontal")
    if (drawXD):
        drawXDellipse(clf, ax[0,1])

        
    cbar_ax = fig.add_axes([0.05, -0.06, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[1,0].scatter(gr,
                         ri,
                         c=pVrat,
                         s=0.5,
                         vmin=0.5,
                         vmax=1.5,
                         cmap="Spectral")
    ax[1,0].set_ylabel("$g-r$")
    ax[1,0].set_xlabel("$r-i$")
    ax[1,0].set_xlim(0.3, 0.9)
    ax[1,0].set_ylim(-0.1, 0.4)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_V/p_V^{model}$", 
                 orientation="horizontal")


    

    cbar_ax = fig.add_axes([0.55, -0.06, 0.40, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[1,1].scatter(acolor,
                         iz,
                         c=pVrat,
                         s=0.3,
                         vmin=0.5,
                         vmax=1.5,
                         cmap="Spectral")
    plotRegions(ax[1,1])
    ax[1,1].set_ylabel("$i-z$")
    ax[1,1].set_xlabel("$a$")
    ax[1,1].set_xlim(-0.3, 0.3)
    ax[1,1].set_ylim(-0.2, 0.3)
    plotRegions(ax[0,1])

    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_{V}/p_V^{model}$", 
                 orientation="horizontal")


     
    plt.show()





## plotting code
def twoPanelsPlot(df, kw): 

    xVec1 = df[kw['Xstr1']]      
    yVec1 = df[kw['Ystr1']]
    cVec1 = df[kw['Cstr1']]
    cLabel1 = kw['Clabel1']

    xVec2 = df[kw['Xstr2']]      
    yVec2 = df[kw['Ystr2']]
    cVec2 = df[kw['Cstr2']]
    cLabel2 = kw['Clabel2']

    DPI = 300
    SAVE_DIR = "../plots/"
    FORMAT = "png"
    SAVE_FIGS = False

    histKwargs = {
        "histtype" : "stepfilled",
        "color" : "#718CA1",
        "alpha" : 0.8, 
        "normed" : True,
    }
    plotKwargs = {
        "ls" : "-",
        "lw" : 1,
        "c" : "red"
    }

    plotKwargsComponents = {
        "ls" : ":",
        "lw" : 1,
        "c" : "red",
        "alpha" : 0.8
    }

    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }

    
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=DPI, figsize=(9.5, 4.8))
    fig.subplots_adjust(
        # the left side of the subplots of the figure
        left=0.09,  
        # the right side of the subplots of the figure
        right=0.95,
        # the bottom of the subplots of the figure
        bottom=0.25,
        # the top of the subplots of the figure
        top=0.95,
        # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        wspace=0.35,   
        # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        hspace=0.55)   
    
    # left subplot           
    cbar_ax = fig.add_axes([0.09, 0.12, 0.37, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax1.scatter(xVec1,
                         yVec1,
                         c=cVec1,
                         s=kw['symbSize'],
                         vmin=kw['Cmin1'],
                         vmax=kw['Cmax1'],
                         cmap="Spectral")
    xlabel = kw['Xstr1']
    if (kw['Xstr1']=='a_color'):
        xlabel = 'a'
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_ylabel(kw["Ystr1"], fontsize=14)
    ax1.set_xlim(kw["Xmin1"], kw["Xmax1"])
    ax1.set_ylim(kw["Ymin1"], kw["Ymax1"])
    cbar = fig.colorbar(cm,
                 cax=cbar_ax,
                 orientation="horizontal")
    cbar.set_label(cLabel1, fontsize = 14)
    if ((kw['Xstr1']=='a')&(kw['Ystr1']=='i-z')):
        plotRegions(ax1)
   
    # right subplot
    cbar_ax = fig.add_axes([0.58, 0.12, 0.37, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax2.scatter(xVec2,
                         yVec2,
                         c=cVec2,
                         s=kw['symbSize'],
                         vmin=kw['Cmin2'],
                         vmax=kw['Cmax2'],
                         cmap="Spectral")

    plotRegions(ax2)
    xlabel = kw['Xstr2']
    if (kw['Xstr2']=='a_color'):
        xlabel = 'a'
    ax2.set_xlabel(xlabel, fontsize=14)
    ax2.set_ylabel(kw["Ystr2"], fontsize=14)
    ax2.set_xlim(kw["Xmin2"], kw["Xmax2"])
    ax2.set_ylim(kw["Ymin2"], kw["Ymax2"])
    cbar = fig.colorbar(cm,
                 cax=cbar_ax,
                 orientation="horizontal")
    cbar.set_label(cLabel2, fontsize = 14)
                 
    plt.savefig(kw['plotName'], dpi=600)
    print('saved plot as:', kw['plotName']) 
    plt.show()
    return





## plotting code
def threePanelsPlot(df, kw): 

    xVec1 = df[kw['Xstr1']]      
    yVec1 = df[kw['Ystr1']]
    cVec1 = df[kw['Cstr1']]
    
    xVec2 = df[kw['Xstr2']]      
    yVec2 = df[kw['Ystr2']]
    cVec2 = df[kw['Cstr2']]

    xVec3 = df[kw['Xstr3']]      
    yVec3 = df[kw['Ystr3']]
    cVec3 = df[kw['Cstr3']]

    DPI = 300
    SAVE_DIR = "../plots/"
    FORMAT = "png"
    SAVE_FIGS = False

    histKwargs = {
        "histtype" : "stepfilled",
        "color" : "#718CA1",
        "alpha" : 0.8, 
        "normed" : True,
    }
    plotKwargs = {
        "ls" : "-",
        "lw" : 1,
        "c" : "red"
    }

    plotKwargsComponents = {
        "ls" : ":",
        "lw" : 1,
        "c" : "red",
        "alpha" : 0.8
    }

    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }

    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=DPI, figsize=(9.5, 3.5))
    fig.subplots_adjust(
        # the left side of the subplots of the figure
        left=0.05,  
        # the right side of the subplots of the figure
        right=1.2,
        # the bottom of the subplots of the figure
        bottom=0.05,
        # the top of the subplots of the figure
        top=0.95,
        # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        wspace=0.35,   
        # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        hspace=0.55)   
    
    # left subplot           
    cbar_ax = fig.add_axes([0.04, -0.12, 0.32, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax1.scatter(xVec1,
                         yVec1,
                         c=cVec1,
                         s=kw['symbSize'],
                         vmin=kw['Cmin1'],
                         vmax=kw['Cmax2'],
                         cmap="Spectral")
    ax1.set_xlabel(kw["Xstr1"], fontsize=14)
    ax1.set_ylabel(kw["Ystr1"], fontsize=14)
    ax1.set_xlim(kw["Xmin1"], kw["Xmax1"])
    ax1.set_ylim(kw["Ymin1"], kw["Ymax1"])
    cbar = fig.colorbar(cm,
                 cax=cbar_ax,
                 orientation="horizontal")
    cbar.set_label(r"$p_{V}$", fontsize = 14)
    
    # middle subplot
    cbar_ax = fig.add_axes([0.46, -0.12, 0.32, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax2.scatter(xVec2,
                         yVec2,
                         c=cVec2,
                         s=kw['symbSize'],
                         vmin=kw['Cmin2'],
                         vmax=kw['Cmax2'],
                         cmap="Spectral")

    #plotRegions(ax2)
    ax2.set_xlabel(kw["Xstr2"], fontsize=14)
    ax2.set_ylabel(kw["Ystr2"], fontsize=14)
    ax2.set_xlim(kw["Xmin2"], kw["Xmax2"])
    ax2.set_ylim(kw["Ymin2"], kw["Ymax2"])
    cbar = fig.colorbar(cm,
                 cax=cbar_ax,
                 orientation="horizontal")
    cbar.set_label(r"$p_{V}$", fontsize = 14)

     # right subplot           
    cbar_ax = fig.add_axes([0.88, -0.12, 0.32, 0.02])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax3.scatter(xVec3,
                         yVec3,
                         c=cVec3,
                         s=kw['symbSize'],
                         vmin=kw['Cmin3'],
                         vmax=kw['Cmax3'],
                         cmap="Spectral")
    ax3.set_xlabel(kw["Xstr3"], fontsize=14)
    ax3.set_ylabel(kw["Ystr3"], fontsize=14)
    ax3.set_xlim(kw["Xmin3"], kw["Xmax3"])
    ax3.set_ylim(kw["Ymin3"], kw["Ymax3"])
    cbar = fig.colorbar(cm,
                 cax=cbar_ax,
                 orientation="horizontal")
    cbar.set_label(r"$p_{V}$", fontsize = 14)
                 
    plt.savefig(kw['plotName'], dpi=600)
    print('saved plot as:', kw['plotName']) 
    plt.show()
    return







## plotting code
def discrete3PanelsPlot(df, classCol, kw): 

    DPI = 300
    SAVE_DIR = "../plots/"
    FORMAT = "png"
    SAVE_FIGS = False

    histKwargs = {
        "histtype" : "stepfilled",
        "color" : "#718CA1",
        "alpha" : 0.8, 
        "normed" : True,
    }
    plotKwargs = {
        "ls" : "-",
        "lw" : 1,
        "c" : "red"
    }

    plotKwargsComponents = {
        "ls" : ":",
        "lw" : 1,
        "c" : "red",
        "alpha" : 0.8
    }

    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }


    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=DPI, figsize=(9.0, 2.5))
    fig.subplots_adjust(
        # the left side of the subplots of the figure
        left=0.08,  
        # the right side of the subplots of the figure
        right=0.99,
        # the bottom of the subplots of the figure
        bottom=0.2,
        # the top of the subplots of the figure
        top=0.95,
        # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        wspace=0.3,   
        # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        hspace=0.45)   
    
    # left subplot           
    for i in range(0, kw['classCount']):
        flag = (df[kw['classVec']] == i)
        dfClass = df[flag]
        if (classCol[i] != 'noplot'):
            ax1.scatter(dfClass[kw['Xstr1']],
                         dfClass[kw['Ystr1']],
                         c=classCol[i],
                         s=kw['symbSize'])
         
    ax1.set_xlabel(kw["Xstr1"], fontsize=14)
    ax1.set_ylabel(kw["Ystr1"], fontsize=14)
    ax1.set_xlim(kw["Xmin1"], kw["Xmax1"])
    ax1.set_ylim(kw["Ymin1"], kw["Ymax1"])
    
    # middle subplot
    for i in range(0, kw['classCount']):
        flag = (df[kw['classVec']] == i)
        dfClass = df[flag]
        if (classCol[i] != 'noplot'):
            ax2.scatter(dfClass[kw['Xstr2']],
                         dfClass[kw['Ystr2']],
                         c=classCol[i],
                         s=kw['symbSize'])

    plotRegions(ax2)
    ax2.set_xlabel(kw["Xstr2"], fontsize=14)
    ax2.set_ylabel(kw["Ystr2"], fontsize=14)
    ax2.set_xlim(kw["Xmin2"], kw["Xmax2"])
    ax2.set_ylim(kw["Ymin2"], kw["Ymax2"])

    # right subplot           
    for i in range(0, kw['classCount']):
        flag = (df[kw['classVec']] == i)
        dfClass = df[flag]
        if (classCol[i] != 'noplot'):
            ax3.scatter(dfClass[kw['Xstr3']],
                         dfClass[kw['Ystr3']],
                         c=classCol[i],
                         s=kw['symbSize'])
    ax3.set_xlabel(kw["Xstr3"], fontsize=14)
    ax3.set_ylabel(kw["Ystr3"], fontsize=14)
    ax3.set_xlim(kw["Xmin3"], kw["Xmax3"])
    ax3.set_ylim(kw["Ymin3"], kw["Ymax3"])

                 
    plt.savefig(kw['plotName'], dpi=600)
    print('saved plot as:', kw['plotName']) 
    plt.show()
    return




def quadPlots(data, Drat):
    DPI = 300
    SAVE_DIR = "../plots/"
    FORMAT = "png"
    SAVE_FIGS = False


    histKwargs = {
        "histtype" : "stepfilled",
        "color" : "#718CA1",
        "alpha" : 0.8, 
        "normed" : True,
    }
    plotKwargs = {
        "ls" : "-",
        "lw" : 1,
        "c" : "red"
    }

    plotKwargsComponents = {
        "ls" : ":",
        "lw" : 1,
        "c" : "red",
        "alpha" : 0.8
    }

    verticalLines = {
        "lw" : 1,
        "linestyle" : "--",
    }

    fig, ax = plt.subplots(2, 2, dpi=DPI, figsize=(5.5, 5.5))
    fig.subplots_adjust(
        # the left side of the subplots of the figure
        left=0.12,  
        # the right side of the subplots of the figure
        right=0.95,
        # the bottom of the subplots of the figure
        bottom=0.1,
        # the top of the subplots of the figure
        top=0.95,
        # the amount of width reserved for space between subplots,
        # expressed as a fraction of the average axis width
        wspace=0.35,   
        # the amount of height reserved for space between subplots,
        # expressed as a fraction of the average axis height
        hspace=0.55)   

    data['region'] =  vat.assignRegionMMI(data["a_color"], data["i-z"])
    oldreg1 = data[data['region']==1]
    oldreg2 = data[data['region']==2]
    oldreg3 = data[data['region']==3]

    pVreg1 = np.median(oldreg1['pV'])
    pVreg2 = np.median(oldreg2['pV'])
    pVreg3 = np.median(oldreg3['pV'])
    
    cbar_ax = fig.add_axes([0.11, 0.52, 0.37, 0.017])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,0].scatter(data["a_color"],
                         data["i-z"],
                         c=data["pV"],
                         s=1,
                         vmin=0.0,
                         vmax=0.5,
                         cmap="Spectral")
    ax[0,0].vlines([0], -0.2, 0.3, **verticalLines)
    xgrid = np.linspace(-0, 0.15, 20)
    splitOGreg2 = 0.08 + 0.0*xgrid
    ax[0,0].plot(xgrid, splitOGreg2, c='k', lw=1, ls="--")
    a_grid = np.linspace(-0, 0.3, 20)
    iz_model = -0.05 + 1.0*a_grid
    ax[0,0].plot(a_grid,iz_model, c='k', lw=1, ls="--")
    ax[0,0].set_ylabel("$i-z$")
    ax[0,0].set_xlabel("$a$")
    ax[0,0].set_xlim(-0.3, 0.3)
    ax[0,0].set_ylim(-0.2, 0.3)
    
    ax[0,0].text(-0.28, -0.18, "%.3f" % pVreg1)
    ax[0,0].text(0.15, 0.26, "%.3f" % pVreg2)
    ax[0,0].text(0.15, -0.18, "%.3f" % pVreg3)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_V$", 
                 orientation="horizontal")

    cbar_ax = fig.add_axes([0.59, 0.52, 0.37, 0.017])
    cbar_ax.xaxis.set_label_position('bottom')
    cbar_ax.xaxis.set_ticks_position('top')
    cm = ax[0,1].scatter(data["a_color"],
                         data["i-z"],
                         c=data['pVrat'],
                         s=1,
                         vmin=0.5,
                         vmax=1.5,
                         cmap="Spectral")
    ax[0,1].vlines([0], -0.2, 0.3, **verticalLines)
    xgrid = np.linspace(-0, 0.15, 20)
    splitOGreg2 = 0.08 + 0.0*xgrid
    ax[0,1].plot(xgrid, splitOGreg2, c='k', lw=1, ls="--")
    a_grid = np.linspace(-0, 0.3, 20)
    iz_model = -0.05 + 1.0*a_grid
    ax[0,1].plot(a_grid,iz_model, c='k', lw=1, ls="--")
    ax[0,1].set_ylabel("$i-z$")
    ax[0,1].set_xlabel("$a$")
    ax[0,1].set_xlim(-0.3, 0.3)
    ax[0,1].set_ylim(-0.2, 0.3)
    fig.colorbar(cm,
                 cax=cbar_ax,
                 label=r"$p_{V}/p_V^{MMI}$", 
                 orientation="horizontal")


    ### histogram panels

    # ax[1,0] histogram
    stats = plotRegHist2(ax[1,0], oldreg2['pVrat'], 'green', histKwargs )

    ax[1,1], stats = plotHist(ax[1,1], 
                              Drat, 
                              [0.4, 1.6], 
                              numGauss=1,
                              bins=50,
                              histKwargs=histKwargs,
                              plotKwargs=plotKwargs,
                              plotKwargsComponents=plotKwargsComponents)
    ax[1,1].set_xlabel("$D^{SDSS} / D^{ATM}$")
    ax[1,1].set_ylabel("n")#
    ax[1,1].set_xlim(0.4, 1.6)
    ax[1,1].set_ylim(0, 3)
    ax[1,1].set_xticks(np.arange(0.4, 1.8, 0.2))
    ax[1,1].set_yticks(np.arange(0, 3.5, 0.5))
    for gauss in [stats]:
        print("mu : {:.3f}".format(gauss[0]))
        print("sigma : {:.3f}".format(gauss[1]))
        print("weight : {:.3f}".format(gauss[2]))
        print("")
        vv = Drat
        median = np.median(vv)
        sG = sigG(vv)
        xx = vv[(vv>median-2*sG)&(vv<median+2*sG)]
        f95 = 100*np.size(xx)/np.size(vv)
        ax[1,1].text(1.18, 2.75, r"$\mu$: {:.3f}".format(median))
        ax[1,1].text(1.18, 2.50, r"$\sigma_G$: {:.3f}".format(sG))
        ax[1,1].text(1.18, 2.23, r"$f95$: {:.1f}".format(f95))

    plt.savefig("quadPlot.png", dpi=600)
    print('saved plot as:', "quadPlot.png")
    plt.show()



def XDcontourplot(Teff, logg, FeH):
    # cannibalized from https://www.astroml.org/examples/datasets/plot_SDSS_SSPP.html
    #------------------------------------------------------------
    # Plot the results using the binned_statistic function
    from astroML.stats import binned_statistic_2d
    N, xedges, yedges = binned_statistic_2d(Teff, logg, FeH,
                                        'count', bins=100)
    FeH_mean, xedges, yedges = binned_statistic_2d(Teff, logg, FeH,
                                               'mean', bins=100)
    
    FeH_sigG, xedges, yedges = binned_statistic_2d(Teff, logg, FeH,
                                               vat.sigG, bins=100)

    # Define custom colormaps: Set pixels with no sources to white
    cmap = plt.cm.jet
    cmap.set_bad('w', 1.)

    cmap_multicolor = plt.cm.jet
    cmap_multicolor.set_bad('w', 1.)

    # Create figure and subplots
    fig= plt.figure(figsize=(16, 5))
    fig.subplots_adjust(wspace=0.25, left=0.1, right=0.95,
                    bottom=0.07, top=0.95)

    #--------------------
    # First axes:
    ax1 = plt.subplot(131)
    plt.imshow(np.log10(N.T), origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap)
    plt.xlim(xedges[0], xedges[-1])
    plt.ylim(yedges[0], yedges[-1])
    plt.xlabel('a', fontsize = 14)
    plt.ylabel('i-z', fontsize = 14)

    cb = plt.colorbar(ticks=[0, 1, 2, 3],
                  format=r'$10^{%i}$', orientation='horizontal')
    cb.set_label(r'$\mathrm{number\ in\ pixel}$', fontsize = 14)
    plt.clim(0, 3)

    #--------------------
    # Second axes:
    ax2 = plt.subplot(132)
    plt.imshow(FeH_mean.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    plt.xlim(xedges[0], xedges[-1])
    plt.ylim(yedges[0], yedges[-1])
    plt.xlabel('a', fontsize = 14)
    plt.ylabel('i-z', fontsize = 14)

    cb = plt.colorbar(ticks=np.arange(0, 0.301, 0.1),
                  format=r'$%.1f$', orientation='horizontal')
    cb.set_label(r'$\mathrm{mean\ p_V \ in\ pixel}$', fontsize = 14)
    plt.clim(0, 0.3)

    # Draw density contours over the colors
    levels = np.linspace(0, np.log10(N.max()), 7)[2:]
    plt.contour(np.log10(N.T), levels, colors='k', linewidths=1,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    #--------------------
    # Third axes:
    ax3 = plt.subplot(133)
    plt.imshow(FeH_sigG.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto', interpolation='nearest', cmap=cmap_multicolor)
    plt.xlim(xedges[0], xedges[-1])
    plt.ylim(yedges[0], yedges[-1])
    plt.xlabel('a', fontsize = 14)
    plt.ylabel('i-z', fontsize = 14)

    cb = plt.colorbar(ticks=np.arange(0, 0.101, 0.02),
                  format=r'$%.2f$', orientation='horizontal')
    cb.set_label(r'$\mathrm{\sigma_G(p_V) \ in\ pixel}$', fontsize = 14)
    plt.clim(0.0, 0.1)
    
    # Draw density contours over the colors
    levels = np.linspace(0, np.log10(N.max()), 7)[2:]
    plt.contour(np.log10(N.T), levels, colors='k', linewidths=1,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    plotRegions(ax1)
    plotRegions(ax2)
    plotRegions(ax3)

    plt.savefig("XDcountourplot.png", dpi=600)
    print('saved plot as:', "XDcountourplot.png")
    plt.show()
