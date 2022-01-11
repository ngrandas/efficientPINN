import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import time
from typing import List
import timeit
import pickle
import os 
os.chdir("/home/UANDES/n.grandas/pythontesis/pythontesis/v3")
device = "cuda"
dtype = torch.float64
puntosCreados = []
class NeuralNetworkPrueba(nn.Module):
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

puntos = torch.linspace(0,7,20, device = device)
puntos.requires_grad = True
perdidavar = 1e9
promAct = 0 
xsup = 7
def limitador(x):
    xint = x
    while (xint>xsup):
        xint -= xsup
    return xint
    
def actualizarPuntos():
    # Esta función va actualizar los puntos. Para mantener un balance entre puntos nuevos y puntos anteriores
    #  tambien va a cubir el espacio recorrido por puntos anteriores. Un objetivo de esta función es correr 
    # con un numero bajo de puntos
    global perdidavar,puntos,puntosCreados,promAct
    coeficiente = 1e-6 #este coeficiente es para alentizar el avance con respecto a la función de perdiad
    nuevosPuntos = puntos.detach().tolist()
    promAct = limitador(promAct+coeficiente/perdidavar*len(puntos))  # acá se da el avance en promedio
    #print("promedio actual ", promAct)
    #print("perdida actual  ", perdidavar)
    #print("avance          ", coeficiente/perdidavar*len(puntos))
    nuevoPunto = max(min(rd.normalvariate(promAct,0.2),xsup),0) # acá se da el avance como una distribución normal
    nuevosPuntos.append(nuevoPunto)
    puntosCreados.append(nuevoPunto)
    
    tamTensor = 200
    borra = 60
    anade = 50
    if len(nuevosPuntos)>tamTensor:
        for i in range(borra):
            indice = rd.randint(0,tamTensor-borra-2)
            nuevosPuntos.pop(indice)
        for i in range(anade):
            nuevosPuntos.append(rd.uniform(0,promAct))
    puntos = torch.tensor(nuevosPuntos,device = device)
    puntos.requires_grad = True

def difeq(x):
    func = redDinamica
    x_in = torch.stack([x])
    y_temp = func(x_in)[0]
    #return y_temp
    der = torch.autograd.grad([y_temp,],[x_in,],create_graph=True)[0]
    if der is not None:
        segder = torch.autograd.grad([der,],[x_in,],create_graph=True)[0]
        if segder is not None:
            return (segder-y_temp)**2
        return x_in*0
    else:
        return x_in*0
puntosPrueba = torch.linspace(0,7,400,requires_grad=False,device = device)

def perdidaParaRevisar():
    x0 =  torch.tensor([puntos[0]],device=device,requires_grad =True)
    xPi = torch.linspace(3.14159/2,1,2,device = device)
    xPi =  torch.tensor([xPi[0]],device=device,requires_grad =True)
    suma = 0
    for j in puntosPrueba:
        i = torch.tensor([j],device=device,requires_grad =True)
        y = redDinamica(i)
        suma += (torch.sin(j)-y)**2
    suma+=100*(redDinamica(x0))**2
    suma+=100*(1-redDinamica(xPi))**2
    return suma


def revisador():
    return perdidaParaRevisar() <0.001*len(puntosPrueba)
def perdida2():
    global perdidavar
    x0 =  torch.tensor([puntos[0]],device=device,requires_grad =True)
    xPi = torch.linspace(3.14159/2,1,2,device = device)
    xPi =  torch.tensor([xPi[0]],device=device,requires_grad =True)
    suma = 0
    for j in puntos:
        i = torch.tensor([j],device=device,requires_grad =True)
        y = redDinamica(i)
        yprima=torch.autograd.grad(y,i,create_graph=True)[0]
        yprimaprima=torch.autograd.grad(yprima,i,create_graph=True)[0]
        suma+=(yprimaprima+y)**2
    suma+=100*(redDinamica(x0))**2
    suma+=100*(1-redDinamica(xPi))**2
    perdidavar = suma
    return suma

#@torch.jit.script
def miniPerdidaParalela(puntosLoc):
    futures : List[torch.jit.Future[torch.Tensor]] = []
    for punto in puntosLoc:
        futures.append(torch.jit.fork(difeq,punto))
    
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))

    return torch.sum(torch.stack(results))

def perdidaParalela():
    global perdidavar	
    suma = 0
    x0 =  torch.tensor([0.0],device=device,requires_grad =True)
    xPi = torch.linspace(3.14159/2,1,2,device = device)
    xPi =  torch.tensor([xPi[0]],device=device,requires_grad =True)
    suma += miniPerdidaParalela(puntos)
    suma+=1000*(redDinamica(x0)[0])**2
    suma+=1000*(1-redDinamica(xPi)[0])**2
    perdidavar = suma 
    return suma
filename = "v3/RESULTADOS/registrosPerdidas/registroPerdidas.pt"
filename = "RESULTADOS/registrosPerdidas/registro.tesis"

try:
    archivo = open(filename,"xb")
except:
    archivo = open(filename,"wb")

for learningRate in list(np.logspace(np.log10(1e-2),np.log10(1e-5),20)):
    pickle.dump("rata de aprendizaje",archivo)
    pickle.dump(learningRate,archivo)
    epochs = 8000
    optimizer = torch.optim.Adam(redDinamica.parameters(), lr=learningRate)
    registro_perdida=[]
    tiempoInicial = time.time()
    i = 0
    termino = False
    while time.time()-tiempoInicial<3600*1.75 and not termino:
        # Compute prediction and loss
        loss = perdida2()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            #print(loss.item()/len(puntos))
            #print("supermax",promAct)
            actualizarPuntos()
        if i % 20 == 0:
            registro_perdida.append(perdidaParaRevisar().item()/len(puntos))
            termino = revisador()
        if i % 300 == 0:
            torch.save(redDinamica.state_dict(),"RESULTADOS/estados/senoParalelo"+str(i)+" epochs"+str(round(np.log10(learningRate),2))+".tar")
        i+=1
    pickle.dump(registro_perdida,archivo)
    if termino:
        pickle.dump("termino en",archivo)
        pickle.dump(time.time()-tiempoInicial,archivo)
    torch.save(redDinamica.state_dict(),"RESULTADOS/terminados/senoParaleloFinales"+str(np.log10(learningRate))+".tar")
    print(learningRate)
    print(i)
    redDinamica = NeuralNetworkPrueba().to(device)
archivo.close()

print("termino")
