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
import datetime
os.chdir(r"/home/externo/Documents/nico/efficientPINN/pythontesis/v9/RESULTADOS/comparacionDeTiempo") # Esta linea hace que el cuaderno corra sobre el directorio deseado
device = "cuda" # Esta linea hace que el cuaderno se ejecute con la gpu de envidia
dtype = torch.float64 # Esta linea hace que el tipo de dato sean floats de 64 bits
plt.ioff()
f = 1.5
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


puntosAleatorios = torch.rand(40)
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
            suma += torch.abs(y-torch.sin(f*j))
    varPerdidaCondicionParada = suma
    return suma

varPerdidaUniforme = 1e2


def revisador():
    return varPerdidaCondicionParada<0.01*len(puntosPrueba)


# %%
def perdida2():
    global perdidavar # esta variable guarda la perdida en una ubicación afuera de la función.
    f = 1.5
    x0 =  torch.tensor([0.0],device=device,requires_grad =True) # Ubicación en X de la primera condición de frontera
    # el 0.0 es importante para que pytorch lo identifique como un float.
    # acá el argumento device se usa para indicar que el vector debe guardarse en la memoria de la gpu
    # Acá el rgumento requires_grad = true se usa para indicar que la variable importa a la hora de la diferenciación 
    xPi = torch.linspace(3.14159/2/f,1,2,device = device) 
    xPi =  torch.tensor([xPi[0]],device=device,requires_grad =True) # Ubicación en X se la segunda condición de frontera
    suma = 0
    for j in puntosAleatorios:
        i = torch.tensor([j],device=device,requires_grad =True) 
        # hay que meter el valor en X dentro de un vector para que pytorch lo tome como una operación de algebra lineal
        # lo anterior es necesario con la función nn.Sequential()
        y = redDinamica(i) 
        yprima=torch.autograd.grad(y,i,create_graph=True)[0]
        yprimaprima=torch.autograd.grad(yprima,i,create_graph=True)[0]
        suma+=(yprimaprima+y*f*f)**2
    # Acá se usa alpha = 100
    suma+=100*(redDinamica(x0))**2
    suma+=100*(1-redDinamica(xPi))**2
    perdidavar = suma
    return suma
perdida2()

# %%
def plottear(filename_):
    if True:
            figura = plt.figure(figsize = (20,10))
            ygrafica = []
            puntosGrafica = torch.linspace(0,10,250)
            for j in puntosGrafica:
                ytemp=redDinamica(torch.tensor([j],device = device))
                ygrafica.append(ytemp.cpu().detach().numpy()[0])
                #ygrafica.append(ytemp.detach().numpy()[0])
            import numpy as np
            puntosGrafica = np.linspace(0,10,250)
            plt.plot(puntosGrafica,ygrafica,label = "red")
            plt.plot(puntosGrafica,np.sin(f*np.array(puntosGrafica)),label = "referencia",linestyle="-.")
            plt.plot(7,np.sin(7),marker = '|',ms= 10, label = 'Fin del dominio')
            plt.legend()
            plt.title(f"{i} epochs")
            plt.savefig(filename)
            plt.close(figura)
            nombreParaGuardarRedIntermedia = "estados/uniforme "+str(i)+".tar"
            torch.save(redDinamica.state_dict(),nombreParaGuardarRedIntermedia)


# %%
ahora = datetime.datetime.now()
filename = f"terminados/{ahora.year}-{ahora.month}-{ahora.day}-{ahora.hour}-{ahora.minute}-{ahora.second}-soluciones_uniformes.tar"
try:
    archivo = open(filename,"xb")
except:
    archivo = open(filename,"wb")
for ronda_solucion in range(1):
    optimizer = torch.optim.Adam(redDinamica.parameters(), lr=1e-3)
    registro_perdida=[]
    registro_promedio=[]
    registro_tiempo = []
    tiempoInicial = time.time()
    i = 0
    termino = False
    while  not termino and tiempoInicial+3600>time.time() :
        # Compute prediction and loss
        loss = perdida2()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(loss.item()/len(puntosAleatorios))
            actualizarPuntosAleatorios()
        if i % 20 == 0:
            if True:
                registro_perdida.append(perdidaParaRevisar().item()/len(puntosAleatorios))
                registro_promedio.append(-1)
                registro_tiempo.append(time.time()-tiempoInicial)
            else:
                inutil = perdidaParaRevisar().item()/len(puntos)
            termino = revisador()
        if i % 200 == 0:
            filename = f"graficas/{ahora.year}-{ahora.month}-{ahora.day}-{ahora.hour}-{ahora.minute}-PROC-Epoch:{i}-{ronda_solucion}"
            plottear(filename)
        i+=1
    metodoTradicional.append(ultimaComparacion(
                                                registro_tiempo,
                                                registro_perdida,
                                                registro_promedio,
                                                f"uniforme {i}"
        ))
    print("TERMINO")
    plottear(f"graficas/{ahora.year}-{ahora.month}-{ahora.day}-{ahora.hour}-{ahora.minute}-TERM-Epoch:{i}-{ronda_solucion}")
    redDinamica = NeuralNetworkPrueba().to(device) 
pickle.dump(metodoTradicional,archivo)
archivo.close

# %%
