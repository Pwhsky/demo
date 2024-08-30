'''
Alex Lech 2024

'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
import os
import time 
import subprocess
import sys
from cython_functions import histogram2d_cython, gradient_cython

font = 19
pi = 3.14159

imageBounds 	   = float(sys.argv[3])*1e-6
spatialPeriodicity = float(sys.argv[4])*1e-9
periodicity		   = float(sys.argv[4])/1000
def dfToNumpy(column):
    return column.to_numpy()

def parseArgs():
	generateData    		= str(sys.argv[2])
	nDeposits               = str(sys.argv[1])
	return nDeposits,generateData
	
def generateNewData():
	print("Generating new data...\n")
	subprocess.run(["g++","functions.cpp","particle.cpp","compute_temperature.cpp","-o","sim","-Ofast", "-fopenmp" , "-funroll-all-loops"])
	subprocess.run(["./sim",nDeposits,sys.argv[3], sys.argv[4]])

def loadData():
	df = pd.read_csv("temperature.csv",engine="pyarrow")
	depositDF = pd.read_csv('deposits.csv')
	df.sort_values(by=['z'])

	with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    		#Parallel data loading
    		futures = [
        	executor.submit(dfToNumpy, df['x']),
        	executor.submit(dfToNumpy, df['y']),
      		executor.submit(dfToNumpy, df['z']),
       		executor.submit(dfToNumpy, df['temperature'])
       		]
	x    = futures[0].result()
	y    = futures[1].result()
	z    = futures[2].result()
	grad = futures[3].result()
	return x,y,z,grad,depositDF
	
def generateLaserProfile(spatialPeriodicity): #Generates gaussian laser profile
	x   = np.linspace(-imageBounds*spatialPeriodicity,imageBounds*spatialPeriodicity, 200)
	y   = np.linspace(-imageBounds*spatialPeriodicity,imageBounds*spatialPeriodicity, 200)
	X,Y = np.meshgrid(x,y)	
	Z = (1+ (np.cos(2*pi*X/spatialPeriodicity)))/2
	return X,Y,Z
	
def generateFigure(imageBounds):
	fig, ax = plt.subplots(1, 2, figsize=(19, 7))
	axisTitles = [f"$\Delta$T for {nDeposits} deposits",f"Position of {nDeposits} deposits",f"Laser intensity for $\Lambda$ = {periodicity} μm"] 
	axisLabelsX = ['X ($\mu m$)','X ($\mu m$)','X ($\mu m$)']
	axisLabelsY = ['Z ($\mu m$)','Z ($\mu m$)','Y ($\mu m$)']

	index = 0
	for axis in ax:
		axis.set_title(axisTitles[index],fontsize=font)
		axis.axis('equal')
		axis.set_xlim(-imageBounds,imageBounds)
		axis.set_ylim(-imageBounds,imageBounds)
		
		axis.set_xlabel(axisLabelsX[index],fontsize=font-1)
		axis.set_ylabel(axisLabelsY[index],fontsize=font-1)
		
		axis.set_yticks([min(z),min(z)/2,0,max(z)/2,max(z)])
		axis.set_xticks([min(x),min(x)/2,0,max(x)/2,max(x)])
		
		axis.set_xticklabels([round(min(x)*1e6,1),round(min(x)/2*1e6,1),0,round(max(x)/2*1e6,1), round(max(x)*1e6,1)],fontsize=font)
		axis.set_yticklabels([round(min(z)*1e6,1),round(min(z)/2*1e6,1),0,round(max(z)/2*1e6,1), round(max(z)*1e6,1)],fontsize=font)
		if index == 0:
			
			im = ax[0].imshow(H.T, origin='lower',  cmap='plasma',
           			 extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
			cbar = plt.colorbar(im,ax=ax[0])
			cbar.set_label(f"$\Delta$T [K]",fontsize=font-5)
			
			cbar.ax.tick_params(labelsize=15)
			
		if index == 1:
			ax[1].grid(True)	
			ax[1].scatter(depositDF['x'], depositDF['z'], label='_nolegend_',s=10)
		index+=1


#Main()
os.chdir("src")
nDeposits,generateData = parseArgs()

if (generateData == "true"):
	generateNewData()
else:
	print("No new data will be generated, starting plotting... \n")
	
tic = time.time()
print("Starting plotting...")
x,y,z,grad,depositDF = loadData()

#Generate histogram image
x_bins = np.linspace(-imageBounds,imageBounds,200)
y_bins = np.linspace(-imageBounds,imageBounds,200)
H, xedges, yedges = histogram2d_cython(x, z, grad, x_bins, y_bins)
generateFigure(imageBounds)

os.chdir("..")
os.chdir("results")
plt.savefig("temperatureIncrease.png")


toc = time.time()
print("Plotting finished after " + str(round(toc-tic)) + " s")
