import numpy as np 
import os 
import matplotlib.pyplot as plt

# for Figure 1 from the paper 
def plotSEDs(wave, flux, labels, lineStyles, lineColors, lineWidths, plotName, yMin=-19, yMax=-14.5):
    ax = plt.axes()
    ax.set_xlabel('$\lambda$ ($\mu$m)')
    ax.set_ylabel('magnitude (AB)')
    for plot in range(0,len(wave)):
        ax.plot(wave[plot], flux[plot], lw = lineWidths[plot], 
                ls = lineStyles[plot], c = lineColors[plot], label = labels[plot])
    ax.legend(title = '        r      $\epsilon$     $p_{v}$    D    T$_1$')
    ax.set_xlim(0.5, 5.3)
    ax.set_ylim(yMin, yMax)
    
    plt.savefig(plotName, dpi=600)
    print('saved plot as:', plotName)   
    plt.show()


def plotSEDerrors(wav, mag, magErr, magMin, magMax, model=False, wM="", mM="", model2=False, wM2="", mM2="", wavMin=0.5, wavMax=5.3):   
    ax = plt.axes()
    ax.set_xlabel('$\lambda$ ($\mu$m)')
    ax.set_ylabel('AB magnitude')
    ax.scatter(wav, mag, s=2.0)
    ax.errorbar(wav, mag, magErr, fmt='.k', lw=2, ecolor='gray')
    ax.set_xlim(wavMin, wavMax)
    ax.set_ylim(magMax, magMin)
    if (model):
        ax.plot(wM, mM, lw=1, c='r')
    if (model2):
        ax.plot(wM2, mM2, lw=1, c='b')
    plt.savefig('oneSEDerrors.png')
    plt.show()
 

def plotCorrectedObs(waveSPH, model, lambdaGridS, noise, magSPHobs, magSPHcorr, yMin, yMax, modelCorr='', ZeroCorr=True):

    lambdaDichroic = 2.42
    if (modelCorr==''):
        modelCorr = model

    fig = plt.figure(figsize=(12, 4))
    fig.subplots_adjust(bottom=0.12, left=0.07, right=0.95)

    ax = fig.add_subplot(1,2,1)
    ax.set(xlabel='$\lambda$ ($\mu$m)',
                ylabel='AB magnitude',
                xlim=(0.5, 5.3),
                ylim=(yMax, yMin))                     
    ax.errorbar(lambdaGridS, magSPHobs, noise, fmt='.k', lw=2, ecolor='gray')
    # rescale the model to agree at the blue edge
    # correct so that the offset is 0 across the blue dichroic
    if (ZeroCorr):
        # true model
        modelObs = np.interp(lambdaGridS, waveSPH, model)
        dmag = magSPHobs - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        model += np.median(dmagBlue)
        # model for deriving correction
        modelObs = np.interp(lambdaGridS, waveSPH, modelCorr)
        dmag = magSPHobs - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        modelCorr += np.median(dmagBlue)
    ax.plot(waveSPH, model, lw=1, c='b')
      
    ax = fig.add_subplot(1,2,2)
    ax.set(xlabel='$\lambda$ ($\mu$m)',
                ylabel='AB magnitude',
                xlim=(0.5, 5.3),
                ylim=(yMax, yMin))                     
    ax.errorbar(lambdaGridS, magSPHcorr, noise, fmt='.b', lw=2, ecolor='blue')
    if (ZeroCorr):
        # true model
        modelObs = np.interp(lambdaGridS, waveSPH, model)
        dmag = magSPHcorr - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        model += np.median(dmagBlue)
        # model for deriving correction
        modelObs = np.interp(lambdaGridS, waveSPH, modelCorr)
        dmag = magSPHcorr - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        modelCorr += np.median(dmagBlue)

    ax.plot(waveSPH, model, lw=1, c='b')
    ax.plot(waveSPH, modelCorr, lw=1, c='r')

    if (1):
        modelObs = np.interp(lambdaGridS, waveSPH, model)
        dmagObs = magSPHobs - modelObs
        dmagObsBlue = dmagObs[lambdaGridS<2.42]
        dmagCorr = magSPHcorr - modelObs
        dmagCorrBlue = dmagCorr[lambdaGridS<2.42]
    plt.savefig('correctedSPHERExSED.png')
    plt.show()
    return 





def plotCorrectedObs4(waveSPH, model, lambdaGridS, noise, magSPHobs, magSPHcorr, yMin, yMax, m1, m2, c1, c2):

    lambdaDichroic = 2.42
    ZeroCorr=True 
    # rescale models to agree at the blue edge
    # correct so that the offset is 0 across the blue dichroic
    if (ZeroCorr):
        # true model
        modelObs = np.interp(lambdaGridS, waveSPH, model)
        dmag = magSPHobs - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        model += np.median(dmagBlue)
        # model 1 for deriving correction
        modelObs = np.interp(lambdaGridS, waveSPH, m1)
        dmag = magSPHobs - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        m1 += np.median(dmagBlue)
        # model 2 for deriving correction
        modelObs = np.interp(lambdaGridS, waveSPH, m2)
        dmag = magSPHobs - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        m2 += np.median(dmagBlue)

    ## plot 
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(bottom=0.12, left=0.07, right=0.95)

    ax = fig.add_subplot(2,2,1)
    ax.set(xlabel='$\lambda$ ($\mu$m)',
                ylabel='AB magnitude',
                xlim=(0.5, 5.3),
                ylim=(yMax, yMin))                     
    ax.errorbar(lambdaGridS, magSPHobs, noise, fmt='.k', lw=2, ecolor='gray')
    ax.plot(waveSPH, model, lw=1, c='b')
      
      
    ax = fig.add_subplot(2,2,2)
    ax.set(xlabel='$\lambda$ ($\mu$m)',
                ylabel='AB magnitude',
                xlim=(0.5, 5.3),
                ylim=(yMax, yMin))                     
    ax.errorbar(lambdaGridS, magSPHcorr, noise, fmt='.b', lw=2, ecolor='blue')
    if (ZeroCorr):
        modelObs = np.interp(lambdaGridS, waveSPH, model)
        dmag = magSPHcorr - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        model += np.median(dmagBlue)
    ax.plot(waveSPH, model, lw=1, c='r')
    # ax.plot(waveSPH, m1, lw=1, c='r')

    ax = fig.add_subplot(2,2,3)
    ax.set(xlabel='$\lambda$ ($\mu$m)',
                ylabel='AB magnitude',
                xlim=(0.5, 5.3),
                ylim=(yMax, yMin))                     
    ax.errorbar(lambdaGridS, c1, noise, fmt='.b', lw=2, ecolor='blue')
    if (ZeroCorr):
        modelObs = np.interp(lambdaGridS, waveSPH, m1)
        dmag = c1 - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        m1 += np.median(dmagBlue)
    ax.plot(waveSPH, model, lw=1, c='b')
    ax.plot(waveSPH, m1, lw=1, c='r')
  
    ax = fig.add_subplot(2,2,4)
    ax.set(xlabel='$\lambda$ ($\mu$m)',
                ylabel='AB magnitude',
                xlim=(0.5, 5.3),
                ylim=(yMax, yMin))                     
    ax.errorbar(lambdaGridS, c2, noise, fmt='.b', lw=2, ecolor='blue')
    if (ZeroCorr):
        modelObs = np.interp(lambdaGridS, waveSPH, m2)
        dmag = c2 - modelObs
        dmagBlue = dmag[lambdaGridS<lambdaDichroic]
        m2 += np.median(dmagBlue)
    ax.plot(waveSPH, model, lw=1, c='b')
    ax.plot(waveSPH, m2, lw=1, c='r')

    plt.savefig('correctedSPHERExSED4.png')
    plt.show()
    return 




