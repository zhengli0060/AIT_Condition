from kerpy.GaussianKernel import GaussianKernel
from utils.independence_testing.HSICBlockTestObject import HSICBlockTestObject

def hsic(x,y, **params):


    lens = len(x)
    if x.ndim == 1:
        x=x.reshape(lens,1)
    if y.ndim == 1:
        y=y.reshape(lens,1)

    blocksize = params.get('blocksize', int(lens/15))
    nullvarmethod = params.get('nullvarmethod', 'direct')
    width = params.get('width', 1.0)
    kernelX_use_median = params.get('kernelX_use_median', False)
    kernelY_use_median = params.get('kernelY_use_median', False)
    width_X = params.get('width_x', None)
    width_Y = params.get('width_y', None)
    if width_X is None:
        width_X = width
    if width_Y is None:
        width_Y = width
    kernelX=GaussianKernel(width_X)
    kernelY=GaussianKernel(width_Y)

    num_samples = lens
    myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY,
                                    kernelX_use_median=kernelX_use_median, kernelY_use_median=kernelY_use_median,
                                    blocksize=blocksize, nullvarmethod=nullvarmethod)

    pvalue = myblockobject.compute_pvalue(x, y)

    return pvalue


def main():
    import numpy as np
    np.random.seed(42)
    x=np.random.uniform(size=1000)
    y=np.random.uniform(size=1000)+x
    print(hsic(x,y))

    x=np.random.uniform(size=1000)
    y=np.random.uniform(size=1000)
    print(hsic(x,y))

if __name__ == '__main__':
    main()