import numpy as np 
import os 
# importing local tools: 
import plottingTools as pt

# read ATM SED file
def getSED(SEDfile):
    # 1) read wavelength (in m), flux (F_lambda in W/m2/m), emissivity and albedo
    # 2) return wavelength in micron, flux (F_lambda in W/m2/micron), emissivity and albedo
    wavelength, flux, epsilon, albedo = np.loadtxt(SEDfile)
    # translate flux to AB mags
    Dast = 10
    magAB = getABmag(wavelength, flux*Dast**2)
    return 1.0e6*wavelength, magAB, epsilon, albedo    


# flux to AB mag
def getABmag(wave, Flambda):
    # given wavelength and Flambda in SI units, return AB magnitudes
    c = 2.998e8
    Fnu = wave**2 / c * Flambda
    return -2.5*np.log10(Fnu*1.0e26/3631)


def getSEDwithNoise(dataDIR, ATMDIR, Dast, rAU, fName, tax):

    # standard wavelengths and expected m5 depths 
    wavSPH, m5static = getSPHERExSensitivity(dataDIR)
    # get noiseless magnitudes for ATM model corrected with chosen Bus-DeMeo class
    SEDfile = ATMDIR + fName + str(rAU) + '.dat' 
    magTrue = getATMBusSED(SEDfile, wavSPH, tax, Dast, rAU, dataDIR)
    
    ### add photometric noise 
    # since asteroids move, we need to shift m5 by 0.75 mag
    # because we don't have 4 surveys as for static sources
    m5SPH = m5static - 0.75     ## IMPORTANT !!! 
    # corresponding photometric errors 
    magErr = getPhotomErrors(magTrue, m5SPH)

    ## obs magnitudes: true mags with added photometric noise and var. offset
    # draw from a Gaussian distribution
    magObs = magTrue + np.random.normal(0, magErr)   
    return wavSPH, magTrue, magObs, magErr 


def getPhotomErrors(mag, m5):
    ## generate photometric errors (use eq.5 from ls.st/lop with gamma=0.039)
    rgamma = 0.039
    xval = np.power(10, 0.4*(mag-m5))
    # random photometric error 
    err_rand = np.sqrt((0.04-rgamma)*xval + rgamma*xval*xval)
    # add systematic error for SPHEREx photometric extractions (in mag)
    sysErr = 0.01    ## somewhat arbitrary, but realistic 
    return np.sqrt(err_rand**2 + sysErr**2)
    


# similar to getSPHERExSED, but already has noise-free static model
def getObsSED(wavSPH, magTrue, mjd0, Amp, Per, sysErr, dataDIR, BrendanFile): 

    # Brendan's file with SPHEREx observations
    mjdObs, waveObs, m5Obs = getBrendanSpectrum(dataDIR, BrendanFile)
    # interpolate true mags to observed wavelengths
    magTrueObs = np.interp(waveObs, wavSPH, magTrue)

    ## photometric errors 
    magErr = getPhotomErrors(magTrueObs, m5Obs) 
    # draw from a Gaussian distribution
    dmNoise = np.random.normal(0, magErr)

    ## generate light curve offsets
    dmOff = Amp*np.sin(2*np.pi*(mjdObs-mjd0)/Per) 

    # raw magnitudes: true mags with added variability and photometric noise
    magRaw = magTrueObs + dmNoise + dmOff

    return waveObs, mjdObs, magRaw, magErr, dmOff, dmNoise 


def getSPHERExSensitivity(dataDIR):
    # read data from Olivier Dore's Point_Source_Sensitivity_v28_base_cbe.txt file
    # wavelengthSPHEREx (in micron) is the first column
    # m5SPHEREx (AB magnitude) is the second column
    dataFile = dataDIR + 'Point_Source_Sensitivity_v28_base_cbe.txt'
    SPHERExSensitivity = np.loadtxt(dataFile)
    return SPHERExSensitivity.T[0], SPHERExSensitivity.T[1]


def getATMBusSED(SEDfile, waveSPH, BusTaxi, Dast, rAU, BusDIR):
  
    # given ATM model, scale by Dast and interpolate to waveSPH
    magTrue, eps, alb = getATMmodelMag(SEDfile, Dast, waveSPH)
   
    # if requested, correct with Bus-DeMeo reflectivity curve 
    if (BusTaxi!=''):
        # read reflectivity curve
        file = BusDIR + "/" + "reflectivity" + BusTaxi + ".dat"
        refldata = np.loadtxt(file, skiprows=1) 
        waveSPHrefl, reflectivity = refldata.T[0], refldata.T[1] 
        print('read in', file)
        if (waveSPHrefl.size != waveSPH.size):
            print('ERROR: different standard SPHEREx wavelength grids!')
        # assumption is that emission is negligible below this wavelength
        wavMinEm = 2.0  # micron, OK outside Earth's orbit     
        # compute and apply correction
        magTrue += getBusAKARIMagCorr(waveSPH, reflectivity, magTrue, wavMinEm)

    return magTrue


# no-noise version
def getATMmodelMag(SEDfile, Dast, waveSPH):
    # 1) read wavelength (in m), flux (F_lambda in W/m2/m), emissivity and albedo
    # 2) correct flux from the fiducial D=1km to Dast 
    # 3) given input wavelength array, compute AB magnitudes 
    # 4) return true AB magnitudes, epsilon, and albedo interpolated to waveSPH values

    # 1) read data
    wavelength, flux, epsilon, albedo = np.loadtxt(SEDfile)
    # 2) correct for Dast and translate flux to AB mags
    magAB = getABmag(wavelength, flux*Dast**2)
    # 3) interpolate magAB, epsilon and albedo to waveSPH
    SPHmagAB = np.interp(waveSPH, 1.0e6*wavelength, magAB)
    SPHeps = np.interp(waveSPH, 1.0e6*wavelength, epsilon)
    SPHalb = np.interp(waveSPH, 1.0e6*wavelength, albedo)
 
    return SPHmagAB, SPHeps, SPHalb


def getBusAKARIMagCorr(wave, refl, magTot, wavMax): 
    ## compute additive correction to magTot because of a different
    ## reflectivity curve affecting the scattered flux; the correction
    ## vanishes at the first wavelength in the SPHEREx standard grid
    refl0 = refl/refl[0]
    # part 1: emission negligible at short wavelengths
    dmag1 = -2.5*np.log10(refl0)
    # part 2: extrapolate the scattered component from short wavelengths
    # compute fraction of the total flux due to scattered component
    magTotATwavMax = np.interp(wavMax, wave, magTot)
    ftotATwavMax = 10**(0.4*(magTot-magTotATwavMax)) 
    # and extrapolate as Rayleigh-Jeans tail 
    fCorr = 1 - (1-refl0) * ftotATwavMax * (wavMax/wave)**2
    dmag2 = -2.5*np.log10(fCorr)
    return np.where(wave < wavMax, dmag1, dmag2)
  

# read MJD and wavelength from Brendan's file, and regrid wavelengths
# to the standard wavelength; return MJD and corresponding standard 
# wavelength and 5-sigma SPHEREx depth
def getBrendanSpectrum(dataDIR, dataFile, singleSurvey=True):
    b = np.loadtxt(dataFile, skiprows=1)
    mjdBrendan = b.T[0]
    wavBrendan = b.T[1]
    waveSPH, m5SPH = getSPHERExSensitivity(dataDIR) 
    wavBrendanSPH = getStandardLambda(wavBrendan, waveSPH)
    m5BrendanSPH = np.interp(wavBrendanSPH, waveSPH, m5SPH)
    if (singleSurvey):
        # since asteroids move, we need to increase errors by sqrt(4)
        # because we don't have 4 surveys as for static sources, or
        # the 5-sigma limiting depth is shallower by ~0.75 mag
        m5BrendanSPH -= 1.25*np.log10(4)
 
    return mjdBrendan, wavBrendanSPH, m5BrendanSPH
   

# wrapper around getOrbitInfoFromBrendanSpectrum since wav is not needed
def getOrbitInfoFromBrendansMJDs(mjd):
    return getOrbitInfoFromBrendanSpectrum(mjd, mjd) 


# given BrendanSpectrum (mjd, wavelength), for each SPHEREx season/survey 
# (separated by >100d), find for all its orbits (<0.05d) how many pairs of
# fluxes per orbit; return as (j=0..Nseason; k=0..Norbits)
# NoPairs[j,k], MJDmin[j,k], MJDmax[j,k]
def getOrbitInfoFromBrendanSpectrum(mjd,wav):
    Norbits = []
    NoPairs = []
    MJDmin = []
    MJDmax = []
    Mmin = []
    Mmax = [] 
    nps = []
    Mmin.append(mjd[0])
    k = 1 
    Nobs = 0 
    for i in range(0,len(mjd)):
        Nobs += 1
        dt = mjd[i] - mjd[i-1]
        if (dt>0.05):
            # new orbit...
            Mmax.append(mjd[i-1])
            nps.append(int(k/2))
            k = 1     
            if (dt>100):
                # and also a new season
                MJDmin.append(Mmin)
                MJDmax.append(Mmax)
                NoPairs.append(nps)
                Mmin = []
                Mmax = [] 
                nps = []
            Mmin.append(mjd[i])
        else:
            # not new orbit, simply count the point
            k += 1
            if (i == (len(mjd)-1)):
                # special case of the last point 
                Mmax.append(mjd[i])
                MJDmin.append(Mmin)
                MJDmax.append(Mmax)
                nps.append(int(k/2))
                NoPairs.append(nps) 
    return NoPairs, MJDmin, MJDmax


def getSPHERExSeasons(NoPairs, MJDmin, MJDmax, verbose=False):
    Nseasons = len(MJDmin)
    Nobs = 0
    for i in range(0, Nseasons):   
        Norbits=len(MJDmin[i])
        Nobs += 2*np.sum(NoPairs[i])
        dt = []
        if verbose:
            for j in range(0,len(NoPairs[i])):
                dMJD = int(60*24*(MJDmax[i][j] - MJDmin[i][j]))
                dt.append(dMJD)
            print('season', i, '  Norb:', Norbits, ' Nobs=', Nobs)
            print('    NoPairs=', NoPairs[i])
            print('         dt=', dt)
    print('No. of observations:', Nobs)
    return Nobs


## select observations from a single season (zero-indexed!) 
def selectSeasonSED(season, waveObs, mjdObs, magRaw, magErr, dmOff, dmNoise):

    NoPairs, MJDmin, MJDmax = getOrbitInfoFromBrendansMJDs(mjdObs) 
    Nseasons = len(MJDmin)
    if (season > Nseasons): 
        print('there are only', Nseasons,' seasons, not', season)
        return
    Norbits=len(MJDmin[season])
    mjdMinVal = MJDmin[season][0]
    mjdMaxVal = MJDmax[season][Norbits-1]
     
    wS = waveObs[(mjdObs>=mjdMinVal)&(mjdObs<=mjdMaxVal)]
    mjdS = mjdObs[(mjdObs>=mjdMinVal)&(mjdObs<=mjdMaxVal)]
    mRawS = magRaw[(mjdObs>=mjdMinVal)&(mjdObs<=mjdMaxVal)]
    mErrS = magErr[(mjdObs>=mjdMinVal)&(mjdObs<=mjdMaxVal)]
    dmOffS = dmOff[(mjdObs>=mjdMinVal)&(mjdObs<=mjdMaxVal)]
    dmNoiseS = dmNoise[(mjdObs>=mjdMinVal)&(mjdObs<=mjdMaxVal)]

    return wS, mjdS, mRawS, mErrS, dmOffS, dmNoiseS


# given lambda from Brendon's file, return lambdaGrid which are 
# the closest values in the standard SPHEREx wavelength grid
def getStandardLambda(waveBrandon, waveSPHEREx):
    lambdaGrid = 0*waveBrandon
    for i in range(0,len(waveBrandon)):
        delLambda = np.abs(waveSPHEREx - waveBrandon[i])
        lambdaGrid[i] = getClosest(delLambda, waveSPHEREx)[0][1] 
    return lambdaGrid 
    
def getClosest(list1, list2):
    zipped_pairs = zip(list1, list2)
    return sorted(zipped_pairs)

def dumpSPHERExSED(MJD, wavelength, mag, magErr, dmVarOff, randNoise, filename):
    np.savetxt(filename, (MJD, wavelength, mag, magErr, dmVarOff, randNoise)) 
    return


def simSPHERExSpec(Dast, rAU, SEDfile, dataDIR, BusTaxi, LC, obsFile, ABrange='', outfilerootname=''): 

    ## set defaults
    if (ABrange==''):
        ABrange = [15.0, 20.0]
    if (outfilerootname==''):
        outfilerootname = './simSPHERExSpecDefault'
    ABmin, ABmax = ABrange
    destfiles = []

    ## SPHEREx standard wavelengths and expected m5 depths for static sources
    wavSPH, m5static = getSPHERExSensitivity(dataDIR)
    ## noise-free SED computed by ATM and corrected for the Bus emissivity
    magTrue = getATMBusSED(SEDfile, wavSPH, BusTaxi[0], Dast, rAU, dataDIR)
 
    ## generate light-curve offsets and noise, and produce "raw" SPHEREx spectrum 
    # light curve parameters
    mjd0 = LC['mjd0']   # arbitrary mjd for phase=0 
    Amp = LC['Ampl']    # sysErr = LC[3]  
    Per = LC['Period']    # period in days 
    sysErr = LC['sysErr'] # additional systematic photometric error for SPHEREx moving objects
    # and now add photometric noise and variability offsets 
    wavObs, mjdObs, magRaw, magErr, dmOff, dmN = getObsSED(wavSPH, magTrue, mjd0, Amp, Per, sysErr, dataDIR, obsFile) 

    ## now analyze and plot all seasons separately
    # first get seasonal and orbital information
    NoPairs, MJDmin, MJDmax = getOrbitInfoFromBrendansMJDs(mjdObs) 
    Nobs = getSPHERExSeasons(NoPairs, MJDmin, MJDmax, verbose=0)
    for season in range(0, len(MJDmin)):
        print('Season', season)
        wS, mjdS, mRawS, mErrS, dmOffS, dmNoiseS = selectSeasonSED(season, wavObs, mjdObs, magRaw, magErr, dmOff, dmN)
        pt.plotSEDerrors(wS, mRawS, mErrS, ABmin, ABmax, True, wavSPH, magTrue) 
        ## save simulated raw SPHEREx SEDs 
        # first data
        outfile = outfilerootname + '_rawSED_Season' + str(season) + '.dat'
        dumpSPHERExSED(mjdS, wS, mRawS, mErrS, dmOffS, dmNoiseS, outfile) 
        # and then plots
        destfile = outfilerootname + '_rawSED_Season' + str(season) + '.png'
        cmdstr = 'mv oneSEDerrors.png ' + destfile
        destfiles.append(destfile)
        os.system(cmdstr) 

    print('produced plots (shown above):')
    for i in range(0,len(destfiles)):
        print('  ', destfiles[i])
    print(' and corresponding data files that have extension dat instead of png')
    print(' ')
    print(' Each data file lists, for a single SPHEREx season, the following quantities:')
    print(' MJD wavelength magSPHEREx magUncertainty varOffset randomNoise')
    print(' the last two entries are added for convenience:')
    print(' the input noiseless model can be obtained as magTrue = mag - varOffset - randomNoise')
    return 
