# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:19:47 2020

@author: user
"""

import numpy as np
from numpy import genfromtxt
import math
import copy
import cv2 as cv
import sys, os
import multiprocessing as mp
import tensorflow as tf
import matplotlib.pyplot as plt


def bayer_reverse(img):
    height,width,c = img.shape
    tmp = np.zeros([height,width])
    for i in range( height ):
        for j in range( width ):
            if i % 2 == 0 :
                if j % 2 == 0:
                    tmp[i][j] = img[i][j][1] #G
                else:
                    tmp[i][j] = img[i][j][2] #R
            else :
                if j % 2 == 0:
                    tmp[i][j] = img[i][j][0] #B
                else:
                    tmp[i][j] = img[i][j][1] #G
    
    return tmp

def ACPIgreenH(CFA):
    # Read the image size
    nRow, nCol = CFA.shape

    # Initialize the output green plane
    green = copy.copy(CFA)

    # Interpolate the missing green samples at blue sampling positions
    for i in range(3,nRow-1,2):
        for j in range(2,nCol-3,2):
            a = CFA[i, j-1]+CFA[i, j+1]
            b = 2*CFA[i, j]-CFA[i, j-2]-CFA[i, j+2]
            green[i, j] = a/2+b/2       

    # Interpolate the missing green samples at red sampling positions
    for i in range(2,nRow-3,2):
        for j in range(3,nCol-1,2):
            a = CFA[i, j-1]+CFA[i, j+1]
            b = 2*CFA[i, j]-CFA[i, j-2]-CFA[i, j+2]
            green[i, j] = a/2+b/2
    
    return green


def ACPIgreenV(CFA):
    # Read the image size
    nRow, nCol = CFA.shape

    # Initialize the output green plane
    green = copy.copy(CFA)

    # Interpolate the missing green samples at blue sampling positions
    for i in range(3,nRow-1,2):
        for j in range(2,nCol-3,2):
            a = CFA[i-1, j]+CFA[i+1, j]
            b = 2*CFA[i, j]-CFA[i-2, j]-CFA[i+2, j]
            green[i, j] = a/2+b/2


    # Interpolate the missing green samples at red sampling positions
    for i in range(2,nRow-3,2):
        for j in range(3,nCol-1,2):	
            a = CFA[i-1, j]+CFA[i+1, j]
            b = 2*CFA[i, j]-CFA[i-2, j]-CFA[i+2, j]
            green[i, j] = a/2+b/2
    return green

def jointDDFWgreen(CFA,gh,gv):
    # Read the image size
    nRow, nCol = CFA.shape

    # Initialize the output green image
    G = copy.copy(CFA)
    DM = np.zeros([nRow, nCol])

    # Compute the differences of chrominance values
    CH = CFA-gh
    CV = CFA-gv
    
    # Compute the horizontal gradients of chrominances
    DH = np.zeros([nRow, nCol])
    DH[0:-2:2,1:-3:2] = abs(CH[0:-2:2,1:-3:2]-CH[0:-2:2,3:-1:2]) # positions of red pixels
    DH[1:-1:2,0:-4:2] = abs(CH[1:-1:2,0:-4:2]-CH[1:-1:2,2:-2:2]) # positions of blue pixels

    
    # Compute the vertical gradients of chrominances
    DV = np.zeros([nRow, nCol])
    DV[0:-4:2,1:-1:2] = abs(CV[0:-4:2,1:-1:2]-CV[2:-2:2,1:-1:2]) # positions of red pixels
    DV[1:-3:2,0:-2:2] = abs(CV[1:-3:2,0:-2:2]-CV[3:-1:2,0:-2:2]) # positions of blue pixels

    

    # Compute DeltaH and DeltaV
    DeltaH = np.zeros([nRow, nCol])
    DeltaV = np.zeros([nRow, nCol])

    # positions of red pixels
    for i in range(2,nRow-3,2):
        for j in range(3,nCol-1,2): 
            DeltaH[i,j] = DH[i-2,j-2]+DH[i-2,j]+DH[i,j-2]+DH[i,j]+DH[i+2,j-2]+DH[i+2,j]+DH[i-1,j-1]+DH[i+1,j-1]   
            DeltaV[i,j] = DV[i-2,j-2]+DV[i-2,j]+DV[i-2,j+2]+DV[i,j-2]+DV[i,j]+DV[i,j+2]+DV[i-1,j-1]+DV[i-1,j+1]
    
    # positions of blue pixels
    for i in range(3,nRow-1,2):
        for j in range(2,nCol-3,2): 	
            DeltaH[i,j] = DH[i-2,j-2]+DH[i-2,j]+DH[i,j-2]+DH[i,j]+DH[i+2,j-2]+DH[i+2,j]+DH[i-1,j-1]+DH[i+1,j-1]   
            DeltaV[i,j] = DV[i-2,j-2]+DV[i-2,j]+DV[i-2,j+2]+DV[i,j-2]+DV[i,j]+DV[i,j+2]+DV[i-1,j-1]+DV[i-1,j+1]


    
    # Decide between the horizontal and vertical interpolations
    T = 1.5

    for i in range(2,nRow-3,2):
        for j in range(3,nCol-1,2):
            if (1+DeltaH[i,j])/(1+DeltaV[i,j])>T:
                G[i,j] = gv[i,j] # vertical strong edge
                DM[i,j] = 1
            elif (1+DeltaV[i,j])/(1+DeltaH[i,j])>T:
                G[i,j] = gh[i,j] # horizontal strong edge
                DM[i,j] = 2
            else:
                h1 = CFA[i, j+1]-CFA[i, j-1]
                h2 = 2*CFA[i, j]-CFA[i, j-2]-CFA[i, j+2]
                v1 = CFA[i+1, j]-CFA[i-1, j]
                v2 = 2*CFA[i, j]-CFA[i-2, j]-CFA[i+2, j]
                HG = abs(h1)+abs(h2)
                VG = abs(v1)+abs(v2)
                w1 = 1/(1+HG)
                w2 = 1/(1+VG)
                G[i,j] = (w1*gh[i,j]+w2*gv[i,j])/(w1+w2)
                DM[i,j] = 3
       

    for i in range(3,nRow-1,2):
        for j in range(2,nCol-3,2):
            if (1+DeltaH[i,j])/(1+DeltaV[i,j])>T:
                G[i,j] = gv[i,j] # vertical strong edge
                DM[i,j] = 1
            elif (1+DeltaV[i,j])/(1+DeltaH[i,j])>T:
                G[i,j] = gh[i,j] # horizontal strong edge
                DM[i,j] = 2
            else:
                h1 = CFA[i, j-1]-CFA[i, j+1]
                h2 = 2*CFA[i, j]-CFA[i, j-2]-CFA[i, j+2]
                v1 = CFA[i-1, j]-CFA[i+1, j]
                v2 = 2*CFA[i, j]-CFA[i-2, j]-CFA[i+2, j]
                HG = abs(h1)+abs(h2)
                VG = abs(v1)+abs(v2)
                w1 = 1/(1+HG)
                w2 = 1/(1+VG)
                G[i,j] = (w1*gh[i,j]+w2*gv[i,j])/(w1+w2)
                DM[i,j] = 3
       
    return G, DM


def DDFWweights(CFA,i,j,type):
    cf = 2
    weights = np.zeros(4)
    if type == 1 :
            h = abs( CFA[i,j-1]-CFA[i,j+1])
            v = abs( CFA[i-1,j]-CFA[i+1,j])
            weights[0] = 1/(1+h+cf*abs(CFA[i,j-3]-CFA[i,j-1])) # left
            weights[1] = 1/(1+v+cf*abs(CFA[i+3,j]-CFA[i+1,j])) # bottom
            weights[2] = 1/(1+h+cf*abs(CFA[i,j+3]-CFA[i,j+1])) # right
            weights[3] = 1/(1+v+cf*abs(CFA[i-3,j]-CFA[i-1,j])) # top
            
    elif  type == 2:
            h = abs(CFA[i-1,j-1]-CFA[i+1,j+1])
            v = abs(CFA[i-1,j+1]-CFA[i+1,j-1])
            weights[0] = 1/(1+h+cf*abs(CFA[i-3,j-3]-CFA[i-1,j-1])) # top left corner
            weights[1] = 1/(1+v+cf*abs(CFA[i+3,j-3]-CFA[i+1,j-1])) # bottom left corner
            weights[2] = 1/(1+h+cf*abs(CFA[i+3,j+3]-CFA[i+1,j+1])) # bottom right corner
            weights[3] = 1/(1+v+cf*abs(CFA[i-3,j+3]-CFA[i-1,j+1])) # top right corner        
         
    else:
            print('Unknown type.')
    return weights

def DDFW_RG_diff(CFA,green):
    nRow, nCol = CFA.shape

    # Interpolate red/green color difference values at blue sampling positions
    KR = np.zeros([nRow,nCol])
    KR = CFA-green

    for i in range(3,nRow-3,2):
        for j in range(4,nCol-3,2):
            # Compute the weights of two diagonal directions
            w = DDFWweights(KR,i,j,2)
            
            # Compute red/green color difference values
            a = KR[i-1,j-1]*w[0]+KR[i+1,j-1]*w[1]+KR[i+1,j+1]*w[2]+KR[i-1,j+1]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KR[i,j] = a/b


    # Interpolate red/green color difference values at green sampling positions

    for i in range(4,nRow-3,2):
        for j in range(4,nCol-3,2):
            
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KR,i,j,1)
            
            # Compute red/green color difference values
            a = KR[i,j-1]*w[0]+KR[i+1,j]*w[1]+KR[i,j+1]*w[2]+KR[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KR[i,j] = a/b


    for i in range(3,nRow-3,2):
        for j in range(3,nCol-3,2):
            
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KR,i,j,1)	    
            # Compute red/green color difference values
            a = KR[i,j-1]*w[0]+KR[i+1,j]*w[1]+KR[i,j+1]*w[2]+KR[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KR[i,j] = a/b
          
    return KR

def DDFW_BG_diff(CFA,green):
    # Read the image size
    nRow, nCol = CFA.shape

    # Interpolate blue/green color difference values at red sampling positions
    KB = np.zeros([nRow,nCol])
    KB = CFA-green

    for i in range(4,nRow-3,2):
        for j in range(3,nCol-3,2):	    
        # Compute the weights of two diagonal directions
            w = DDFWweights(KB,i,j,2)
            
            # Compute blue/green color difference values
            a = KB[i-1,j-1]*w[0]+KB[i+1,j-1]*w[1]+KB[i+1,j+1]*w[2]+KB[i-1,j+1]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KB[i,j] = a/b
            


    # Interpolate blue/green color difference values at green sampling positions

    for i in range(4,nRow-3,2):
        for j in range(4,nCol-3,2):
            
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KB,i,j,1)
            
            # Compute blue/green color difference values
            a = KB[i,j-1]*w[0]+KB[i+1,j]*w[1]+KB[i,j+1]*w[2]+KB[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KB[i,j] = a/b
            

    for i in range(3,nRow-3,2):
        for j in range(3,nCol-3,2):
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KB,i,j,1)
            
            # Compute blue/green color difference values
            a = KB[i,j-1]*w[0]+KB[i+1,j]*w[1]+KB[i,j+1]*w[2]+KB[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KB[i,j] = a/b
        
    
    return KB


def DDFW_refine_green(CFA,KR,KB, DM):

    # Read the image size
    nRow, nCol = CFA.shape

    # Initialize the output G image
    G = CFA

    # Refine green samples at red sampling positions

    for i in range(4,nRow-3,2):
        for j in range(3,nCol-3,2):        
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KR,i,j,1)
            
            # Refine the green samples
            a = KR[i,j-1]*w[0]+KR[i+1,j]*w[1]+KR[i,j+1]*w[2]+KR[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            G[i,j] = CFA[i,j] - a/b
            
            #DM[i, j] = (w[0] + w[2]) / b
            
    
    # Refine green samples at blue sampling positions

    for i in range(3,nRow-3,2):
        for j in range(4,nCol-3,2):    
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KB,i,j,1)
            
            # Refine the green samples
            a = KB[i,j-1]*w[0]+KB[i+1,j]*w[1]+KB[i,j+1]*w[2]+KB[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            G[i,j] = CFA[i,j] - a/b
            
            DM[i, j] = (w[0] + w[2]) / b
    return G, DM


def DDFW_refine_RG_diff(KR, DM):
    # Refine the red/green color difference plane.

    nRow, nCol = KR.shape

    # Refine red/green color difference values at blue sampling positions

    for i in range(3,nRow-3,2):
        for j in range(4,nCol-3,2):        
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KR,i,j,1)
            
            # Refine red/green color difference values
            a = KR[i,j-1]*w[0]+KR[i+1,j]*w[1]+KR[i,j+1]*w[2]+KR[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KR[i,j] = a/b
            

    # Refine red/green color difference values at green sampling positions

    for i in range(4,nRow-3,2):
        for j in range(4,nCol-3,2):            
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KR,i,j,1)
            
            # Refine red/green color difference values
            a = KR[i,j-1]*w[0]+KR[i+1,j]*w[1]+KR[i,j+1]*w[2]+KR[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KR[i,j] = a/b
            

    for i in range(3,nRow-3,2):
        for j in range(3,nCol-3,2):            
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KR,i,j,1)
            
            # Refine red/green color difference values
            a = KR[i,j-1]*w[0]+KR[i+1,j]*w[1]+KR[i,j+1]*w[2]+KR[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KR[i,j] = a/b
    return KR, DM



def DDFW_refine_BG_diff(KB, DM):

    nRow, nCol = KB.shape

    # Refine blue/green color difference values at red sampling positions

    for i in range(4,nRow-3,2):
        for j in range(3,nCol-3,2):        
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KB,i,j,1)
            
            # blue/green color difference values
            a = KB[i,j-1]*w[0]+KB[i+1,j]*w[1]+KB[i,j+1]*w[2]+KB[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KB[i,j] = a/b
            
            DM[i, j] = (w[0] + w[2]) / b

    # Refine blue/green color difference values at green sampling positions
 
    for i in range(4,nRow-3,2):
        for j in range(4,nCol-3,2):            
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KB,i,j,1)
            
            # Refine red/green color difference values
            a = KB[i,j-1]*w[0]+KB[i+1,j]*w[1]+KB[i,j+1]*w[2]+KB[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KB[i,j] = a/b
            
            DM[i, j] = (w[0] + w[2]) / b
            
     
    for i in range(3,nRow-3,2):
        for j in range(3,nCol-3,2):            
            # Compute the weights of horizontal and vertical directions
            w = DDFWweights(KB,i,j,1)
            
            # Refine red/green color difference values
            a = KB[i,j-1]*w[0]+KB[i+1,j]*w[1]+KB[i,j+1]*w[2]+KB[i-1,j]*w[3]
            b = w[0]+w[1]+w[2]+w[3]
            KB[i,j] = a/b
            
            DM[i, j] = (w[0] + w[2]) / b
    return KB, DM

def RoundImage(img):
    nRow, nCol,c = img.shape
    out = np.zeros([nRow,nCol])
    for i in range(nRow):
        for j in range(nCol):
            img[i][j][0] = round(img[i][j][0])
            if img[i][j][0] > 255:
                img[i][j][0] = 255
            if img[i][j][0] < 0:
                img[i][j][0] = 0
            
            img[i][j][1] = round(img[i][j][1])
            if img[i][j][1] > 255:
                img[i][j][1] = 255
            if img[i][j][1] < 0:
                img[i][j][1] = 0
            
            img[i][j][2] = round(img[i][j][2])
            if img[i][j][2] > 255:
                img[i][j][2] = 255
            if img[i][j][2] < 0:
                img[i][j][2] = 0


    cv.imwrite('output.png',img)
    return img

def is_green(i, j):
    return (i%2) == (j%2)

def _DeltaH(CFA, i, j):
    if is_green(i, j):
        return (CFA[i, j] / 2) + (CFA[i, j-2] / 4) +   \
            (CFA[i, j+2] / 4) - (CFA[i, j-1] / 2) - \
            (CFA[i, j+1] / 2)
    else:
        return (CFA[i, j-1] / 2) + (CFA[i, j+1] / 2) - \
            (CFA[i, j-2] / 4) - (CFA[i, j+2] / 4) - \
            (CFA[i, j] / 2)

def _DeltaV(CFA, i, j):        
    if is_green(i, j):
        return (CFA[i, j] / 2) + (CFA[i-2, j] / 4) +   \
            (CFA[i+2, j] / 4) - (CFA[i-1, j] / 2) - \
            (CFA[i+1, j] / 2)
    else:
        return (CFA[i-1, j] / 2) + (CFA[i+1, j] / 2) - \
            (CFA[i-2, j] / 4) - (CFA[i+2, j] / 4) - \
            (CFA[i, j] / 2)

def getDelta(dh, dv, gh, gv):        
    if gv <= gh:
        if gv * 4 <= gh:
            return dv
        elif gv * 2 <= gh:
            return (3 * dv + dh) / 4
        else:
            return (dv + dh) / 2
    else:
        if gh * 4 <= gv:
            return dh
        elif gh * 2 <= gv:
            return (3 * dh + dv) / 4
        else:
            return (dh + dv) / 2

def MGBI_green(CFA):
    nRow, nCol = CFA.shape
    out = np.copy(CFA)
    
    # Compute DeltaH and DeltaV
    DeltaH = np.zeros([nRow, nCol])
    DeltaV = np.zeros([nRow, nCol])
    
    for i in range(2, nRow-3):
        for j in range(2, nCol-3):
            DeltaH[i, j] = _DeltaH(CFA, i, j)
            DeltaV[i, j] = _DeltaV(CFA, i, j)

    # Interpolation G
    for i in range(3, nRow-3):
        for j in range(3, nCol-3):
            if not is_green(i, j):
                Delta_H = np.zeros((3, 3))
                Delta_V = np.zeros((3, 3))
                
                for m in range(0, 3):
                    for n in range(0, 3):
                        Delta_H[m, n] = DeltaH[i+2*(m-1), j+(n-1)]
                        Delta_V[m, n] = DeltaV[i+(m-1), j+2*(n-1)]
                
                Gh = np.dot(np.abs(Delta_H[:, 0] - Delta_H[:, 1]) + 
                            np.abs(Delta_H[:, 1] - Delta_H[:, 2]), 
                            np.array([0.25, 0.5, 0.25]))                
                Gv = np.dot(np.abs(Delta_V[0, :] - Delta_V[1, :]) + 
                            np.abs(Delta_V[1, :] - Delta_V[2, :]), 
                            np.array([0.25, 0.5, 0.25]))

                tmp_H = np.dot(np.array([0.05, 0.9, 0.05]), Delta_H)
                Dh = np.dot(tmp_H, np.array([0.05, 0.9, 0.05]))      
                tmp_V = np.dot(np.array([0.05, 0.9, 0.05]), Delta_V)
                Dv = np.dot(tmp_V, np.array([0.05, 0.9, 0.05]))  
                
                D = getDelta(Dh, Dv, Gh, Gv)
                
                # 插補出G #
                temp = CFA[i, j] + D
                out[i, j] = 255 if temp > 255 else 0 if temp < 0 else temp

    return out


def jointDDFW(CFA, name):
    # Interpolate the green plane
    #gh = ACPIgreenH(CFA) # horizontally interpolated green image
    #gv = ACPIgreenV(CFA) # vertically interpolated green image
    DM = np.zeros(CFA.shape)
    
    #G0, DM = jointDDFWgreen(CFA,gh,gv)
    G0 = MGBI_green(CFA)
    #print('../result/'+ name.split('.')[0] +'_2x.png')
    #MGBI = cv.imread('../result/'+ name.split('.')[0] +'_2x.png')
    #G0 = MGBI[:, :, 1]
    
    #plt.imshow(np.abs(G1-G0), cmap='hsv', vmin=0, vmax=30)
    #plt.show()
    #cv.imwrite(name.split('.')[0] +'_2x.png', MGBI)

    # Interpolate color difference planes

    KR = DDFW_RG_diff(CFA,G0)
    KB = DDFW_BG_diff(CFA,G0)
    
    # Refine the estimates

    G, DM = DDFW_refine_green(CFA,KR,KB, DM)
    KR, DM = DDFW_refine_RG_diff(KR+G0-G, DM)
    KB, DM = DDFW_refine_BG_diff(KB+G0-G, DM)

    
    # Generate the demosaicked result
    w,h = G.shape
    d_out = np.zeros([w,h,3])
    d_out[:,:,0] = G+KB
    d_out[:,:,1] = G
    d_out[:,:,2] = G+KR
    #print(d_out.shape)
    
    d_out = RoundImage(d_out)
    return d_out, DM


def zoomGreen(green,DM,k,T):
    # Read the green image size
    m,n = green.shape
    nRow = 2*m
    nCol = 2*n

    # Initialize the output green image and the directional matrix DM
    A = np.zeros([nRow,nCol]) 
    temp = np.zeros([nRow,nCol])
    
    A[0:-1:2,0:-1:2] = green
    temp[0:-1:2,0:-1:2] = DM
    np.savetxt("A.csv",A, delimiter=",")
    # Do the interpolation of the green plane
    
    for i in range(3,nRow-3,2):
        for j in range(3,nCol-3,2):    
            # Compute weights and interpolation directions
            w1,w2,n = WghtDir(A[i-3:i+4,j-3:j+4],k,T)
            
            
            

            # Compute green sample values
            A[i,j] = greenValue(A[i-3:i+4,j-3:j+4],1,w1,w2,n,i,j)
            
            temp[i,j] = n # store the new estimated edge direction

    #np.savetxt("A1.csv",A, delimiter=",")

    for i in range(4,nRow-4,2):
        for j in range(3,nCol-3,2):
        
            # Compute weights
            w1,w2 = CalcWeights(A[i-2:i+3,j-2:j+3],k)
            
            # Compute green sample values
            n = temp[i,j-1]+temp[i,j+1]
            A[i,j] = greenValue(A[i-3:i+4,j-3:j+4],2,w1,w2,n,i,j)
        


    for i in range(3,nRow-3,2):
        for j in range(4,nCol-4,2):
        
            # Compute weights
            w1,w2 = CalcWeights(A[i-2:i+3,j-2:j+3],k)
            
            # Compute green sample values
            n = temp[i-1,j]+temp[i+1,j]
            
            A[i,j] = greenValue(A[i-3:i+4,j-3:j+4],3,w1,w2,n,i,j)
           
    #np.savetxt("A.csv",A, delimiter=",")
    DM = temp
    return A, DM
    
def  WghtDir(A,k,T):
    # 45 degree diagonal direction
    t1 = abs(A[2,0]-A[0,2])   
    t2 = abs(A[4,0]-A[2,2])+abs(A[2,2]-A[0,4])     
    t3 = abs(A[6,0]-A[4,2])+abs(A[4,2]-A[2,4])+abs(A[2,4]-A[0,6]) 
    t4 = abs(A[6,2]-A[4,4])+abs(A[4,4]-A[2,6]) 
    t5 = abs(A[6,4]-A[4,6]) 
    d1 = t1+t2+t3+t4+t5
    
    # 135 degree diagonal direction
    t1 = abs(A[0,4]-A[2,6])   
    t2 = abs(A[0,2]-A[2,4])+abs(A[2,4]-A[4,6])   
    t3 = abs(A[0,0]-A[2,2])+abs(A[2,2]-A[4,4])+abs(A[4,4]-A[6,6]) 
    t4 = abs(A[2,0]-A[4,2])+abs(A[4,2]-A[6,4])
    t5 = abs(A[4,0]-A[6,2])
    d2 = t1+t2+t3+t4+t5

    # Compute the weight vector
    w1 = 1/(1+math.pow(d1,k) ) 
    w2 = 1/(1+math.pow(d2,k) )

    # Compute the directional index
    
    if (1+d1)/(1+d2) > T:
       n = 1 # 135 degree strong edge
    elif (1+d2)/(1+d1) > T:
       n = 2 # 45 degree strong edge
    else:
       n = 3
    return (w1,w2,n)

def CalcWeights(A,k):
    

    # horizontal direction   
    t1 = abs(A[0,1]-A[0,3])+abs(A[2,1]-A[2,3])+abs(A[4,1]-A[4,3])
    t2 = abs(A[1,0]-A[1,2])+abs(A[1,2]-A[1,4])
    t3 = abs(A[3,0]-A[3,2])+abs(A[3,2]-A[3,4])
    d1 = t1+t2+t3

    # vertical direction   
    t1 = abs(A[1,0]-A[3,0])+abs(A[1,2]-A[3,2])+abs(A[1,4]-A[3,4])
    t2 = abs(A[0,1]-A[2,1])+abs(A[2,1]-A[4,1])
    t3 = abs(A[0,3]-A[2,3])+abs(A[2,3]-A[4,3])
    d2 = t1+t2+t3

    # Compute the weight vector
    w1 = 1/(1+math.pow(d1,k)) 
    w2 = 1/(1+math.pow(d2,k))
    
    return w1,w2
    
def greenValue(A,type,w1,w2,n,i,j):
    

    f = np.array([-1,9,9,-1])/16
    if type == 1 :
        v1 = np.array([ A[6,0],A[4,2],A[2,4],A[0,6] ])
        v2 = np.array([ A[0,0],A[2,2],A[4,4],A[6,6] ])
    else :
        v1 = np.array([ A[3,1],A[3,2],A[3,4],A[3,6] ])
        v2 = np.array([ A[1,3],A[2,3],A[4,3],A[6,3] ])	
    
    if n == 1:
       p = v2[0]*f[0] + v2[1]*f[1] + v2[2]*f[2] + v2[3]*f[3] 
    elif n == 2:
       p = v1[0]*f[0] + v1[1]*f[1] + v1[2]*f[2] + v1[3]*f[3] 
    else:
       p1 = v1[0]*f[0] + v1[1]*f[1] + v1[2]*f[2] + v1[3]*f[3]
       p2 = v2[0]*f[0] + v2[1]*f[1] + v2[2]*f[2] + v2[3]*f[3]
       p = (w1*p1+w2*p2)/(w1+w2)


    return p
    
    
def zoomColorDiff(color_diff,DM):
    # Read the color difference image size
    m,n = color_diff.shape
    nRow = 2*m
    nCol = 2*n

    # Initialize the output image
    A = np.zeros([nRow,nCol])
    
    A[0:nRow:2,0:nCol:2] = color_diff


    # Do the interpolation of the color difference plane
    
    for i in range(3,nRow-3,2):
        for j in range(3,nCol-3,2):	
            if DM[i,j]==1:
               A[i,j] = (A[i-1,j-1]+A[i+1,j+1])/2
            elif DM[i,j]==2:
               A[i,j] = (A[i-1,j+1]+A[i+1,j-1])/2
            else:
               A[i,j] = (A[i-1,j-1]+A[i-1,j+1]+A[i+1,j-1]+A[i+1,j+1])/4
        

    
    for i in range(4,nRow-4,2):
        for j in range(3,nCol-3,2):
    
            if DM[i,j-1]+DM[i,j+1]==1:
               A[i,j] = (A[i-1,j]+A[i+1,j])/2
            elif DM[i,j-1]+DM[i,j+1]==2:
               A[i,j] = (A[i,j-1]+A[i,j+1])/2
            else: 
               A[i,j] = (A[i-1,j]+A[i+1,j]+A[i,j-1]+A[i,j+1])/4
      

    
    for i in range(3,nRow-3,2):
        for j in range(4,nCol-4,2):
    
            if DM[i-1,j]+DM[i+1,j]==1:
               A[i,j] = (A[i-1,j]+A[i+1,j])/2
            elif DM[i-1,j]+DM[i+1,j]==2:
               A[i,j] = (A[i,j-1]+A[i,j+1])/2
            else:
               A[i,j] = (A[i-1,j]+A[i+1,j]+A[i,j-1]+A[i,j+1])/4  
        
    return A

def jointZoom(img,DM,k,T):
    # Read every color plane

    G = img[:,:,1]
    grDiff = img[:,:,2]-G
    gbDiff = img[:,:,0]-G

    # Perform the enlargement of every color plane
    zoomedG, DM = zoomGreen(G,DM,k,T) # Zoom green plane
    zoomedR = zoomedG+zoomColorDiff(grDiff,DM) # Zoom red/green color difference
    zoomedB = zoomedG+zoomColorDiff(gbDiff,DM) # Zoom blue/green color difference
    
    np.savetxt("zoomedG.csv", zoomedG, delimiter=",")
    np.savetxt("zoomedR.csv", zoomedR, delimiter=",")
    np.savetxt("zoomedB.csv", zoomedB, delimiter=",")


    out = np.zeros([G.shape[0]*2,G.shape[1]*2,3])
    # Obtain the output image
    out[:,:,2] = zoomedR
    out[:,:,1] = zoomedG
    out[:,:,0] = zoomedB
    
    return out

def epi5Zoom(img, DM):
    h, w, c = img.shape
    out = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    window = np.zeros((5, 5, 3), dtype=np.uint8)
    
    gamma = np.array([0.15, -0.75, 1, -0.28, -0.12])
    base_linear = np.array([0.10151382, -0.30232316, 1.0731945, 0.15881203, -0.031327657])
    base_cubic  = np.array([0.13411383, -0.46307590, 1.2877634, 0.08989102, -0.048093690])
    base = base_linear - 0.1*gamma
    
    for oy in range(4, (h*2)-4):
        for ox in range(4, (w*2)-4):
            
            j = int((ox)/2)
            i = int((oy)/2)
            
            for m in range(5):
                    for n in range(5):
                        window[m, n, :] = img[i-2+m, j-2+n, :]
            
            b = base if (ox % 2) == 1 else np.flip(base)
            a = base if (oy % 2) == 1 else np.flip(base)
            
            for color in range(3):
                ori = window[:, :, color].astype(np.float64)
                temp = round(np.dot(np.dot(a, ori), b))
                    
                temp *= (temp>0)
                if temp > 255 :
                    temp = 255

                out[oy, ox, color] = temp
                
    return out

def epiWeight(vs, gamma=2):
        x_gamma = gamma/2
        
        tmp = (1-vs)
        vc3 = (tmp*vs*vs)*x_gamma
        vc0 = (tmp*tmp*vs)*x_gamma
        vc1 = tmp + 2*vc0 - 1*vc3
        vc2 = vs + 2*vc3 - 1*vc0
        vc0 = (-1)*vc0
        vc3 = (-1)*vc3

        return np.array([vc0, vc1, vc2, vc3])

def epiZoom(img, DM):
    h, w, c = img.shape
    out = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    window = np.zeros((4, 4, 3), dtype=np.uint8)
    g = 1.5
    r = 1
    
    g1 = 1
    r1 = 0.5
    
    for oy in range(5, (h*2)-5):
        for ox in range(5, (w*2)-5):
            x = (ox + 0.5) * (0.5) - 0.5
            y = (oy + 0.5) * (0.5) - 0.5
            
            j = int(x)	#i = floor(x) 
            i = int(y) #j = floor(y)
            dx = x - float(j)
            dy = y - float(i)
            
            nj = int((ox)/2)
            ni = int((oy)/2)
            
            for m in range(4):
                    for n in range(4):
                        window[m, n, :] = img[i-1+m, j-1+n, :]

            if not is_green(ni, nj):
                b = epiWeight(dx, g1 + r1*(1-DM[ni, nj]))
                a = epiWeight(dy, g1 + r1*DM[ni, nj])
            else:
                b = epiWeight(dx, g + r*(1-DM[ni, nj]))
                a = epiWeight(dy, g + r*DM[ni, nj])
            
            for color in range(3):
                ori = window[:, :, color].astype(np.float64)
                temp = round(np.dot(np.dot(a, ori), b))
                    
                temp *= (temp>0)
                if temp > 255 :
                    temp = 255

                out[oy, ox, color] = temp
                
    return out

def DZ(CFA, name):
    #2.1Interpolation of Green Plane
    #Directional interpolation step
    

    k = 5
    T = 1.15

    d_out, DM = jointDDFW(CFA, name)
    cv.imwrite('DDFW_' + name + '.png', d_out)
    plt.imshow(DM, cmap='hsv', vmin=0, vmax=1)
    plt.show()
    w,h = CFA.shape
    
    #test = cv.resize(d_out,(2*h,2*w),interpolation=cv.INTER_CUBIC)
    #test = cv.resize(d_out,(2*h,2*w),interpolation=cv.INTER_LINEAR)
    
    #cv.imwrite('good.png',test)
    
    #OUT = jointZoom(d_out/255, DM, k, T)
    #OUT = RoundImage(OUT*255)
    #OUT = jointZoom(d_out, DM, k, T)
    #OUT = epiZoom(d_out, DM)
    OUT = d_out
    #OUT = cv.resize(d_out,(2*h,2*w),interpolation=cv.INTER_CUBIC)
    
    return OUT



def cpsnr_calc(RGB1,RGB2,b):

    RGB1 = RGB1.astype('double') 
    RGB2 = RGB2.astype('double')
    diff = RGB1[b:-1-b,b:-1-b,:]-RGB2[b:-1-b,b:-1-b,:]
    num = np.size(diff[:,:,1])
    MSE_R = np.sum( np.power(diff[:,:,2],2) )/num
    MSE_G = np.sum( np.power(diff[:,:,1],2) )/num
    MSE_B = np.sum( np.power(diff[:,:,0],2) )/num
    CMSE = (MSE_R + MSE_G + MSE_B)/(3)
    CPSNR = 10*math.log(255*255/CMSE,10)
    PSNR_R = 10*math.log(255*255/MSE_R,10)
    PSNR_G = 10*math.log(255*255/MSE_G,10)
    PSNR_B = 10*math.log(255*255/MSE_B,10)
    return MSE_R, MSE_G, MSE_B, CPSNR, PSNR_R, PSNR_G, PSNR_B

def get_cpsnr(RGB1,RGB2,b):
    RGB1 = RGB1.astype('double') 
    RGB2 = RGB2.astype('double')
    diff = RGB1[b:-1-b,b:-1-b,:]-RGB2[b:-1-b,b:-1-b,:]
    num = np.size(diff[:,:,1])
    MSE_R = np.sum( np.power(diff[:,:,2],2) )
    MSE_G = np.sum( np.power(diff[:,:,1],2) )
    MSE_B = np.sum( np.power(diff[:,:,0],2) )
    CMSE = (MSE_R + MSE_G + MSE_B)/(3*num)
    CPSNR = 10*math.log(255*255/CMSE,10)

    return CPSNR

def read_img(img):
	return tf.convert_to_tensor(img[:, :, 0:3], dtype=np.uint8)

def do_psnr(tf_img1, tf_img2):
	return tf.image.psnr(tf_img1, tf_img2, max_val=255)

def run(path):
    img = cv.imread(path)
    name = os.path.basename(path).split('.')[0]
    rows,cols,h = img.shape
    
    #zoom = np.zeros([math.floor(cols/2),math.floor(rows/2)])
    
    print(img.shape)
    print((math.floor(rows/2),math.floor(cols/2)))
    
    if (rows % 2) == 1:
        rows = rows - 1
    if (cols % 2) == 1:
        cols = cols - 1
    
    #####################################
    zoom = cv.resize(img[0:rows, 0:cols, :],(math.floor(cols/2),math.floor(rows/2)),interpolation=cv.INTER_LINEAR)
    #zoom = cv.resize(img,(math.floor(rows/2),math.floor(cols/2)),interpolation=cv.INTER_CUBIC)
    #zoom = img[0:cols-1:2,0:rows-1:2] 
    #zoom = img
    img = zoom
    #####################################
    
    tmp = bayer_reverse(zoom)
    cv.imwrite('tmp.bmp',tmp)
    
    out = DZ(tmp, name)
    cv.imwrite('output_' + name + '.png',out)
    #out = test
    b=14
    
    MSE_R, MSE_G, MSE_B, CPSNR, PSNR_R, PSNR_G, PSNR_B = cpsnr_calc(img,out,b)
    #print('MSE_R = %lf ' %MSE_R)
    #print('MSE_G = %lf ' %MSE_G)
    #print('MSE_B = %lf ' %MSE_B)
    #print('CPSNR = %lf ' %CPSNR)
    #print('PSNR_R = %lf ' %PSNR_R)
    #print('PSNR_G = %lf ' %PSNR_G)
    #print('PSNR_B = %lf ' %PSNR_B)
    
    CPSNR = get_cpsnr(img,out,b)

    #p = float(do_psnr(read_img(img[b:-1-b,b:-1-b,:]), read_img(out[b:-1-b,b:-1-b,:])))

    print('CPSNR = %lf ' %CPSNR)
    #print(p)
    

    '''
    img = cv.imread('test.bmp')
    DM = genfromtxt('DM1.csv', delimiter=',')
    k = 5 T = 1.15
    img = img.astype('float') 
    
    OUT = jointZoom(img/255.0, DM, k, T)
    OUT = RoundImage(OUT*255)
    #OUT = jointZoom(img, DM, k, T)
    cv.imwrite('output3.png',OUT)
    img = cv.imread('kodim04.png',)
    CPSNR = get_cpsnr(img,OUT,12)
    print('CPSNR = %lf ' %CPSNR)
    '''
    
    return CPSNR
    
def main():
    arg = sys.argv[1]

    if os.path.isfile(arg):
        files = [arg]
    elif os.path.isdir(arg):
        files = [os.path.join(arg, f) for f in os.listdir(arg)]
    else:
        print("[ARGS] File '" + arg + "' does not exist!")
        return -1
    
    files.sort()
    
    print("Files to be run:")
    print(*files, sep="\n")
    print("============================")
    
    if len(files) != 1:    
        num_cores = mp.cpu_count()
        
        pool = mp.Pool(processes=(num_cores if len(files) > num_cores else len(files)))
        ss = np.array(pool.map(run, files)) 
        pool.close()  
        pool.join()
        
        for s in ss:
            print(s)
        print(np.average(ss))
    
    else:
        print(run(files[0]))

if __name__ == '__main__':
    main()