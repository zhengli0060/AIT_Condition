import sys
sys.path.append("./indTest")
from kerpy.GaussianKernel import GaussianKernel
# from HSICTestObject import HSICTestObject
from numpy import shape,savetxt,loadtxt,transpose,shape,reshape,concatenate
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
from independence_testing.HSICBlockTestObject import HSICBlockTestObject
import numpy as np
import pandas as pd


def test(x,y,alpha=0.05,**params):
    verbose = params.get('verbose', False)
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()
##    kernelY = GaussianKernel(float(0.3))
##    kernelX=GaussianKernel(float(0.3))



    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=False, kernelY_use_median=False,
                                          rff=True, num_rfx=40, num_rfy=40, num_nullsims=1000)




    pvalue = myspectralobject.compute_pvalue(x, y)

    #print(pvalue)
    # if pvalue >alpha:
    #     return True, pvalue
    # else:
    #     return False, pvalue
    return pvalue

def test2(x,y,alph=0.08):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()
##    kernelY = GaussianKernel(float(0.45))
##    kernelX=GaussianKernel(float(0.45))
    num_samples = lens

    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                    kernelX_use_median=False, kernelY_use_median=False,
                                    blocksize=80, nullvarmethod='permutation')

    pvalue = myblockobject.compute_pvalue(x, y)

    if pvalue >alph:
        return True
    else:
        return False



def INtest(x,y,alph=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()
##    kernelY = GaussianKernel(float(0.4))
##    kernelX=GaussianKernel(float(0.4))
    num_samples = lens

    myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                          kernelX_use_median=True, kernelY_use_median=True,
                                          rff=True, num_rfx=20, num_rfy=20, num_nullsims=1000)

    pvalue = myspectralobject.compute_pvalue(x, y)

    return pvalue

def INtest2(x,y,alph=0.01):
    lens = len(x)
    x=x.reshape(lens,1)
    y=y.reshape(lens,1)
    kernelX=GaussianKernel()
    kernelY=GaussianKernel()
    num_samples = lens

    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                    kernelX_use_median=True, kernelY_use_median=True,
                                    blocksize=200, nullvarmethod='permutation')

    pvalue = myblockobject.compute_pvalue(x, y)

    return pvalue



def main():
    x=np.random.uniform(size=10000)
    y=np.random.uniform(size=10000)+x
    print(test(x,y))

if __name__ == '__main__':
    main()
