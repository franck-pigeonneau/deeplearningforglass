#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri. Nov 20 2026

This script python trains a model to fit the Poisson's ratio. As explained in
ref. [1], the model fit the unpredicted coefficient 'alpha' in the Makishima
and Mackenzie's model of the Poisson's ratio [2].

To fit the alpha factor, the atomic packing factor is required. It is taken
from the Young's modulus.

Databases for both nu and E come from Interglad V8 anf Sciglass. Only 20 oxides
are involved as pinpointed out in [1].

References:

[1] F. Pigeonneau, M. Rondet, O. de Lataulade, E. Hachem (2025). 
    Physical-informed deep learning prediction of solid and fluid mechanical 
    properties of oxide glasses J. Non-Cryst. Solids, 657:123476.
[2] A. Makishima & J. D. Mackenzie (). Calculation of bulk modulus, 
    shear modulus and Poisson's ratio of glass J. Non-Cryst. Solids, 17:147-157.

@author: fpigeonneau, CEMEF/CFL
"""

# Import python modules
# ---------------------

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd

# Import our own modules
#c----------------------

from glassdata import GlassData
from network import NeuralNetwork

# PATH for the saving of figures
# ------------------------------

PATH='/home/fpigeonneau/ownCloud/Figures/MachineLearning/'

# ----------------------------------------
# Loading of the database of Young modulus
# ----------------------------------------

filedbE='DataBase/E20oxides.csv'
dbE=GlassData(filedbE)

# Constitution of the glass composed with the oxide
oxideE=dbE.oxide
NsampleE,NoxideE=dbE.shape()
dbE.bounds()

# ------------------------------
# Loading of dissociation energy
# ------------------------------

datadisso=pd.read_csv('dissociationenergy.csv')
G=np.zeros(NsampleE)
for i in range(NsampleE):
    G[i]=np.sum(datadisso['G'].values*dbE.x[i,:])
#end for

# Determination of E/2G unité SI corresponding to the atomic packing factor

dbE.y=dbE.y/(2.*G)
dbE.normalize_y()

# -------------------------
# Loading of neural network
# -------------------------

arch=[20,20,20]
nnmodelEsG=NeuralNetwork(NoxideE,arch,'gelu','linear')
nnmodelEsG.ArchName(arch)
nnmodelEsG.compile(1.e-4)
nnmodelEsG.load('Models/nnEsG'+nnmodelEsG.namearch,True)
nnmodelEsG.info()

# ----------------------------------------------
# Loading of the database of Poisson coefficient
# ----------------------------------------------

filedbnu='DataBase/nu20oxides.csv'
dbnu=GlassData(filedbnu)
dbnu.bounds()

# Determination of the compaction volume on the dataset of nu
# -----------------------------------------------------------

xdbE=dbE.xglasstoxdb(dbnu.oxide,dbnu.x)
Vt=dbE.physicaly(nnmodelEsG.model.predict(xdbE).transpose()[0,:])

# --------------------
# Computation of alpha
# --------------------

dbnu.y=1./(3.*(1.-2.*dbnu.y)*Vt)
dbnu.normalize_y()

# -------------------------------------------------
# Determination of the training and validation sets
# -------------------------------------------------

dbnu.split(0.7,0.2)

# ----------------------------
# Generation of neural network
# ----------------------------

reload=True
savefig=True
arch=[20,20,20]
Nepoch=5000
batchsize=256
Nfold=0
Savedata=False
errormax=0.1

# --------------------------------
# Definition of the neural network
# --------------------------------

nnmodel=NeuralNetwork(dbnu.noxide,arch,'gelu','linear')
nnmodel.compile(3.e-4)
nnmodel.ArchName(arch)
nnmodel.info()

# ---------------------
# Training of the model
# ---------------------

modelfile='Models/nn'+dbnu.nameproperty+nnmodel.namearch
nnmodel.load(modelfile,reload)

if (Nfold>0):
    # ----------------------------
    # Training by cross validation
    # ----------------------------
    
    kfold=KFold(n_splits=Nfold,shuffle=True)
    
    ifold=0
    for train_index,val_index in kfold.split(dbnu.x[dbnu.x_train:dbnu.x_test]):
        # -------------------------------------------------
        # Determination of the training and validation sets
        # -------------------------------------------------
        
        x_train,x_val=dbnu.x[train_index],dbnu.x[val_index]
        y_train,y_val=dbnu.y[train_index],dbnu.y[val_index]
        
        # ---------------------
        # Training of the model
        # ---------------------
        
        nnmodel.fit(x_train,y_train,x_val,y_val,epochs=Nepoch,batch_size=batchsize)
        
        # -------------------
        # Saving of the model
        # -------------------
        
        nnmodel.save(modelfile)
        
        # ----------------------
        # Incrementaion of ifold
        # ----------------------
        
        ifold+=1
    # end for
else:
    
    # -------------------------------------------------
    # Determination of the training and validation sets
    # -------------------------------------------------
    
    x_train,x_val=dbnu.x[dbnu.x_train:dbnu.x_valid],dbnu.x[dbnu.x_valid:dbnu.x_test]
    y_train,y_val=dbnu.y[dbnu.x_train:dbnu.x_valid],dbnu.y[dbnu.x_valid:dbnu.x_test]
    
    # ---------------------
    # Training of the model
    # ---------------------
    
    nnmodel.fit(x_train,y_train,x_val,y_val,epochs=Nepoch,batch_size=batchsize)
    
    # -------------------
    # Saving of the model
    # -------------------
    
    nnmodel.save(modelfile)
#endif

# ------------------
# Plot training data
# ------------------
    
lossfig='Figures/loss'+dbnu.nameproperty+nnmodel.namearch+'.pdf'
nnmodel.plot(lossfig,savefig)
    
# -----------------------
# Prediction of the model
# ----------------------
    
nuerr_nn_train=nnmodel.model.predict(x_train)
nuerr_nn_val=nnmodel.model.predict(x_val)
nuerr_nn_test=nnmodel.model.predict(dbnu.x[dbnu.x_test:dbnu.nsample-1])

# -------------------------------------------------
# Computation of the Vickers hardness from the data
# -------------------------------------------------

nuerr_actual_train=np.reshape(y_train,(-1,1))
nuerr_actual_val=np.reshape(y_val,(-1,1))
nuerr_actual_test=np.reshape(dbnu.y[dbnu.x_test:dbnu.nsample-1],(-1,1))
ymin=np.min(np.concatenate([nuerr_actual_train,nuerr_actual_val,nuerr_actual_test]))
ymax=np.max(np.concatenate([nuerr_actual_train,nuerr_actual_val,nuerr_actual_test]))

# ---------------------------------------------------------
# Determination of the R2 scores for the three sets of data
# ---------------------------------------------------------

r2_train=r2_score(nuerr_actual_train,nuerr_nn_train)
r2_val=r2_score(nuerr_actual_val,nuerr_nn_val)
r2_test=r2_score(nuerr_actual_test,nuerr_nn_test)

# ------------------
# Graphical plotting
# ------------------

plt.figure()
plt.plot(nuerr_actual_train,nuerr_nn_train,'ko')
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax]),'k',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.+errormax),'k-.',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.-errormax),'k-.',linewidth=2)
plt.xlabel(r'actual $\bar{\alpha}$',fontsize=12)
plt.ylabel(r'predicted $\bar{\alpha}$',fontsize=12)
plt.xlim((ymin,ymax))
plt.ylim((ymin,ymax))
plt.title(r'Training, $R^2$='+str(np.round(r2_train,decimals=3)),fontsize=12)
if (savefig):
   plt.savefig(PATH+'predictedvsactualalphanu-train.pdf',dpi=300,bbox_inches='tight')
#endif

plt.figure()
plt.plot(nuerr_actual_val,nuerr_nn_val,'bo')
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax]),'b',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.+errormax),'b-.',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.-errormax),'b-.',linewidth=2)
plt.xlabel(r'actual $\bar{\alpha}$',fontsize=12)
plt.ylabel(r'predicted $\bar{\alpha}$',fontsize=12)
plt.xlim((ymin,ymax))
plt.ylim((ymin,ymax))
plt.title(r'Validation, $R^2$='+str(np.round(r2_val,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactualalphanu-val.pdf',dpi=300,bbox_inches='tight')
#endif

plt.figure()
plt.plot(nuerr_actual_test,nuerr_nn_test,'go')
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax]),'g',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.+errormax),'g-.',linewidth=2)
plt.plot(np.array([ymin,ymax]),np.array([ymin,ymax])*(1.-errormax),'g-.',linewidth=2)
plt.xlabel(r'actual $\bar{\alpha}$',fontsize=12)
plt.ylabel(r'predicted $\bar{\alpha}$',fontsize=12)
plt.xlim((ymin,ymax))
plt.ylim((ymin,ymax))
plt.title(r'Test, $R^2$='+str(np.round(r2_test,decimals=3)),fontsize=12)
if (savefig):
    plt.savefig(PATH+'predictedvsactualalphanu-test.pdf',dpi=300,bbox_inches='tight')
#endif

# ----------------------------------------------------
# Prediction of the Poisson's ratio from the ANN model
# ----------------------------------------------------

nuANN=dbnu.PoissonRatio(dbE,nnmodelEsG,nnmodel,dbnu.oxide,dbnu.x)

# ---------------------
# Back to initial value
# ---------------------

dbnu.y=0.5-1./(6.*dbnu.physicaly(dbnu.y)*Vt)

plt.figure()
plt.plot(dbnu.y,nuANN,'ko')
plt.plot([0.9*np.min(dbnu.y),1.1*np.max(dbnu.y)],[0.9*np.min(dbnu.y),1.1*np.max(dbnu.y)],'k',linewidth=2)
plt.plot([0.9*np.min(dbnu.y),1.1*np.max(dbnu.y)],np.array([0.9*np.min(dbnu.y),1.1*np.max(dbnu.y)])*(1.+errormax),'k-.',linewidth=2)
plt.plot([0.9*np.min(dbnu.y),1.1*np.max(dbnu.y)],np.array([0.9*np.min(dbnu.y),1.1*np.max(dbnu.y)])*(1.-errormax),'k-.',linewidth=2)
plt.xlim((0.9*np.min(dbnu.y),1.1*np.max(dbnu.y)))
plt.ylim((0.9*np.min(dbnu.y),1.1*np.max(dbnu.y)))
plt.xlabel(r'Data-set $\nu$',fontsize=14)
plt.ylabel(r'predicted $\nu$',fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(PATH+'predictionvsdsetnualphaANN.pdf',dpi=300,bbox_inches='tight')
stdnu=np.std((nuANN-dbnu.y)/dbnu.y)
print('stdnu=',stdnu)
r2_nu=r2_score(dbnu.y,nuANN)
print('R2=',r2_nu)

