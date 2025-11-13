
import numpy as np
import asgard

import matplotlib.pyplot as plt

# put testing code here
if __name__ == '__main__':

    full = asgard.pde_snapshot('full8.h5')
    adapt = asgard.pde_snapshot('adapt.h5')

    fdata = np.sort( np.abs(full.state) )[::-1]

    adata = np.sort( np.abs(adapt.state) )[::-1]

    ax = plt.subplot2grid((1, 2), (0, 0))

    ax.loglog(fdata, 'b')
    ax.loglog(adata, 'r')

    ax = plt.subplot2grid((1, 2), (0, 1))

    fdata = np.abs(full.state[0:9])
    adata = np.abs(adapt.state[0:9])

    ax.semilogy(fdata, 'b')
    ax.semilogy(adata, 'r')

    plt.show()
