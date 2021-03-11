# lib
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List, Dict, Optional
from icecream import ic
import colorsys

# custom lib
import jx_lib

class SOM:
    def __init__(
        self,
        training_data, # Assumed: normalized
        space: int      = 100, # 100 x 100 grid of neurons
        alpha_0: float  = 0.8,
        verbose: bool   = True,
        path: str       = "output",
    ):
        # Initialize the system
        self.training_data = training_data
        self.space = space
        self.alpha_0 = alpha_0
        self.verbose = verbose
        self.path = path

        # Initialize random weights
        self.w = np.random.random((space,space,3))

        # init output
        jx_lib.create_all_folders(path)
        if verbose:
            self.imshow(data=self.w, name="w_0")
            self.imshow(data=np.reshape(self.training_data,(1,self.training_data.shape[0],3)), name="color_bar")

    def imshow(self, data, name:str, save:bool=True):
        fig = plt.figure()
        plt.imshow(data, interpolation='none')
        if save:
            fig.savefig("{}/{}.png".format(self.path, name), bbox_inches = 'tight')
        if self.verbose:
            plt.title(name)
            # plt.show(block=False)
            # plt.pause(0.5)
            plt.close(fig)

    def learn(
        self,
        sigma_0: int = 10, # [10,40,70]
        tot_training_epochs: int = 10 #600
    ):
        self.tot_training_epochs = tot_training_epochs
        # learn:
        epoch = 1
        T = tot_training_epochs
        N = self.space
        while epoch <= tot_training_epochs:
            k = epoch
                
            # Alpha(k) & s(k):
            alpha_k = self.alpha_0 * np.exp(- k / T)
            s_k = sigma_0 * np.exp(- k / T)
            s_k_2_2_division = 1 / (2 * (s_k ** 2)) # pre-optimization
            # w_ij:
            for x in self.training_data:
                # calculate performance index
                diff = np.linalg.norm(x - self.w, axis = 2)
                # find index of winning node
                ind = np.unravel_index(np.argmin(diff, axis=None), diff.shape) # y,x
                # Update weights for neighbourhood
                xx = np.arange(0, N, 1)
                yy = np.arange(0, N, 1)

                ### matrix form (optimization):
                Mj = np.meshgrid(xx, yy)
                Dx = (Mj[0] - ind[1]) ** 2
                Dy = (Mj[1] - ind[0]) ** 2
                Dij2 = Dx+Dy
                Nij = np.exp(- Dij2 * s_k_2_2_division )
                dxw = np.subtract(x, self.w)
                Nw = np.stack([Nij, Nij, Nij], axis=2) # depth stacking
                self.w = self.w + alpha_k * np.multiply(Nw, dxw)

            plot_ind = [1, 20, 40, 100, 600, 1000, 1500, 2000]
            if epoch in plot_ind:
                print("Epoch Number: {}".format(epoch))
                self.imshow(data=self.w, 
                    name="[s={}]_w_{}".format(sigma_0, epoch))
            
            epoch += 1

def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

def main():
    ### IMPORT DATA ###
    # manual pick:
    # inputRGB = np.array([
    #     [255,0,0],
    #     [0,255,0],
    #     [0,0,255],
    #     [255,255,0],
    #     [255,0,255],
    #     [0,255,255],
    #     [128,128,0],
    #     [128,0,128],
    #     [0,128,128],
    #     [255,128,0],
    #     [255,0,128],
    #     [128,255,0],
    #     [0,255,128],
    #     [128,0,255],
    #     [0,128,255],
    #     [255,20,147],
    #     [220,20,60],
    #     [255,51,51],
    #     [255,153,51],
    #     [255,255,51],
    #     [51,255,51],
    #     [153,255,51],
    #     [51,255,153],
    #     [51,255,255]])
    # inputRGB = np.array([
    #     [0,0,0],
    #     [255,255,255],
    #     [255,0,0],
    #     [0,255,0],
    #     [0,0,255],
    #     [255,255,0],
    #     [0,255,255],
    #     [255,0,255],
    #     [192,192,192],
    #     [128,128,128],
    #     [128,0,0],
    #     [128,128,0],
    #     [0,128,0],
    #     [128,0,128],
    #     [0,128,128],
    #     [0,0,128],
    #     [188,143,143],
    #     [210,105,30],
    #     [147,112,219],
    #     [127,255,212],
    #     [144,238,144],
    #     [255,160,122],
    #     [178,34,34],
    #     [72,61,139],
    # ])
    
    # randomly generate 24 Colors
    inputRGB = []
    for i in np.random.uniform(size=24):
        inputRGB.append(list(hsv2rgb(i,1.0,1.0)))
    inputRGB = np.array(inputRGB)

    # normalization
    normRGB = inputRGB/255.0
    ic(normRGB.shape)

    # SOM:
    for s in [10, 40, 70]:
        som = SOM(
            training_data = normRGB,
            # DEFAULT:
            space   = 100, # 100 x 100 grid of neurons
            alpha_0 = 0.8,
            verbose = True,
            path    = "output/p2",
        )
        som.learn(
            sigma_0 = s, # [10,40,70]
            tot_training_epochs= 2000
        )
    


if __name__ == "__main__":
    main()
