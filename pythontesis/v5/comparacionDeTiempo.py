# %%
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import time
from typing import List
import pickle
import os 

os.chdir("/home/UANDES/n.grandas/pythontesis/pythontesis/v5/RESULTADOS/comparacionDeTiempo") # Esta linea hace que el cuaderno corra sobre el directorio deseado
device = "cuda" # Esta linea hace que el cuaderno se ejecute con la gpu de envidia
dtype = torch.float64 # Esta linea hace que el tipo de dato sean floats de 64 bits


class NeuralNetworkPrueba(nn.Module): # Acá se declara el tipo de red que se va a usar
    def __init__(self):
        super(NeuralNetworkPrueba, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

redDinamica = NeuralNetworkPrueba().to(device) 


puntosAleatorios = torch.rand(40)*7
def actualizarPuntosAleatorios():
    global puntosAleatorios
    puntosAleatorios = torch.rand(40)*7

def solucionarPuntosAleatorios():
    pass

class ultimaComparacion():
    def __init__(self,tiempos,perdidas,promedios,nombresito):
        self.tiempos = tiempos
        self.registrosPerdidasGlobales = perdidas
        self.registroPromedios = promedios
        self.nombre = nombresito
        self.frecuenciaSampleo = 200 # Sampleo cada 200 datos
        pass
    def anadirDato(self,tiempo,perdida):
        self.tiempos.append(tiempo)
        self.registrosPerdidasGlobales.append(perdida)

metodoTradicional = []
metodoUsado = []

puntosPrueba = torch.linspace(0,7,400,requires_grad=True)
varPerdidaCondicionParada = 1e9


def perdidaParaRevisar():
    global varPerdidaCondicionParada
    with torch.no_grad():
        suma = 0
        for j in puntosPrueba:
            i = torch.tensor([j],device=device)
            y = redDinamica(i)
            suma += torch.abs(y-torch.sin(j))
    varPerdidaCondicionParada = suma
    return suma

varPerdidaUniforme = 1e2

def perdidaUniforme():
    global varPerdida,puntosAleatorios,varPerdidaUniforme
    suma = 0
    for j in puntosAleatorios:
        i = torch.tensor([j],device=device)
        y = redDinamica(i)
        suma += torch.abs(y-torch.sin(j))
    x0 =  torch.tensor([0.0],device=device,requires_grad =True) # Ubicación en X de la primera condición de frontera
    xPi = torch.linspace(3.14159/2,1,2,device = device) 
    xPi =  torch.tensor([xPi[0]],device=device,requires_grad =True) # Ubicación en X se la segunda condición de frontera
    suma+=100*(redDinamica(x0))**2
    suma+=100*(1-redDinamica(xPi))**2
    varPerdidaUniforme = suma.item()
    return suma



def revisador():
    return varPerdidaCondicionParada<0.01*len(puntosPrueba)

# %%
filename = "uniformes3.tar"
try:
    archivo = open(filename,"xb")
except:
    archivo = open(filename,"wb")
for run in range(5):
    optimizer = torch.optim.Adam(redDinamica.parameters(), lr=1e-3)
    registro_perdida=[]
    registro_promedio=[]
    registro_tiempo = []
    tiempoInicial = time.time()
    i = 0
    termino = False
    while  not termino and tiempoInicial+3600*10>time.time() :
        # Compute prediction and loss
        loss = perdidaUniforme()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            #print(loss.item()/len(puntosAleatorios))
            actualizarPuntosAleatorios()
        if i % 20 == 0:
            if True:
                registro_perdida.append(perdidaParaRevisar().item()/len(puntosAleatorios))
                registro_promedio.append(-1)
                registro_tiempo.append(time.time()-tiempoInicial)
            else:
                inutil = perdidaParaRevisar().item()/len(puntos)
            termino = revisador()
        if i % 300 == 0:
            if True:
                nombreParaGuardarRedIntermedia = f"estados/uniforme {i}, ronda {run}.tar"
                torch.save(redDinamica.state_dict(),nombreParaGuardarRedIntermedia)
        i+=1
    metodoTradicional.append(ultimaComparacion(
                                                registro_tiempo,
                                                registro_perdida,
                                                registro_promedio,
                                                f"uniforme {i}"
        ))
    
    print(f"TERMINO EN {time.time()-tiempoInicial} SEGUNDOS")
    nombreParaGuardarRedIntermedia = f"terminados/ronda {run}.tar"
    torch.save(redDinamica.state_dict(),nombreParaGuardarRedIntermedia)
    redDinamica = NeuralNetworkPrueba().to(device) 
pickle.dump(metodoTradicional,archivo)
archivo.close
        

# %%



