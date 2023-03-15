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
os.chdir(r"/home/externo/Documents/nico/efficientPINN/pythontesis/v6/RESULTADOS/comparacionDeTiempo") # Esta linea hace que el cuaderno corra sobre el directorio deseado
device = "cuda" # Esta linea hace que el cuaderno se ejecute con la gpu de envidia
dtype = torch.float64 # Esta linea hace que el tipo de dato sean floats de 64 bits
plt.ioff()

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
            suma += torch.abs(y-torch.sin(j))
    varPerdidaCondicionParada = suma
    return suma

varPerdidaUniforme = 1e2

def perdidaUniforme():
    global puntosAleatorios,varPerdidaUniforme
    suma = 0
    for j in puntosAleatorios:
        i = torch.tensor([j],device=device)
        print(i)
        y = redDinamica(i)
        print(y)
        return
        yprima=torch.autograd.grad(y,i,create_graph=True)[0]
        yprimaprima=torch.autograd.grad(yprima,i,create_graph=True)[0]
        suma+=(yprimaprima+y)**2
        #suma += torch.abs(y-torch.sin(j))
    x0 =  torch.tensor([0.0],device=device,requires_grad =True) # Ubicación en X de la primera condición de frontera
    xPi = torch.linspace(3.14159/2,1,2,device = device) 
    xPi =  torch.tensor([xPi[0]],device=device,requires_grad =True) # Ubicación en X se la segunda condición de frontera
    suma+=100*(redDinamica(x0))**2
    suma+=100*(1-redDinamica(xPi))**2
    varPerdidaUniforme = suma.item()
    return suma

redDinamica = NeuralNetworkPrueba().to(device) # Acá se declara la función de la red neuronal y se pone a correr en gpu
redDinamica.float()
dtype = torch.float32 

def inicializarPuntos():
    global puntos,promAct
    puntos = torch.linspace(0,1,20, device = device)
    pesos = [1]*20
    promAct = 0.1

pesos = [1]*20
xsup = 7
def limitador(x):
    xint = x
    while (xint>xsup):
        xint -= xsup
    return xint

promAct = 0
c1 = 1e-6
c2 = 1.2
sigma = 1

def actualizarPuntosConPesos():
     # Esta función va actualizar los puntos. Para mantener un balance entre puntos nuevos y puntos anteriores
    #  tambien va a cubir el espacio recorrido por puntos anteriores. Un objetivo de esta función es correr 
    # con un numero bajo de puntos
    global perdidavar,puntos,puntosCreados,promAct,c1,c2,sigma, pesos
    if isinstance(pesos,List):
        nuevosPesos = pesos.copy()
    else:
        nuevosPesos = pesos.detach().tolist()
    nuevosPuntos = puntos.detach().tolist()
    promAct = limitador(min(promAct+c1/(perdidavar/len(puntos))**c2,promAct+0.25))  # acá se da el avance en promedio
    nuevoPunto = max(min(rd.normalvariate(promAct,sigma),xsup),0) # acá se da el avance como una distribución normal
    nuevosPuntos.append(nuevoPunto) 
    nuevosPesos.append(1)   
    tamTensor = 40
    borra = 35
    anade = 25
    if len(nuevosPuntos)>tamTensor:
        for i in range(borra):
            indice = rd.randint(0,tamTensor-borra-2)
            nuevosPuntos.pop(indice)
            nuevosPesos.pop(indice)
        for i in range(anade):
            nuevosPuntos.append(rd.uniform(0,promAct))
            nuevosPesos.append(20)
    puntos = torch.tensor(nuevosPuntos,device = device,dtype=dtype)
    puntos.requires_grad = True
    pesos = nuevosPesos
    if type(promAct) == torch.Tensor:
        promAct = promAct.item()


puntos = torch.linspace(0,0.1,20,device = device, dtype=dtype)
puntos.requires_grad = True
def perdidaConPesos():
    f = 1.5
    global perdidavar # esta variable guarda la perdida en una ubicación afuera de la función.
    x0 =  torch.tensor([0.0],device=device,requires_grad =True) # Ubicación en X de la primera condición de frontera
    # el 0.0 es importante para que pytorch lo identifique como un float.
    # acá el argumento device se usa para indicar que el vector debe guardarse en la memoria de la gpu
    # Acá el rgumento requires_grad = true se usa para indicar que la variable importa a la hora de la diferenciación 
    xPi = torch.linspace(3.14159/2/f,1,2,device = device) 
    xPi =  torch.tensor([xPi[0]],device=device,requires_grad =True) # Ubicación en X se la segunda condición de frontera
    suma = 0
    contador = 0
    for j in puntos:
        i = torch.tensor([j],device=device,requires_grad =True) 
        # hay que meter el valor en X dentro de un vector para que pytorch lo tome como una operación de algebra lineal
        # lo anterior es necesario con la función nn.Sequential()
        y = redDinamica(i) 
        yprima=torch.autograd.grad(y,i,create_graph=True)[0]
        yprimaprima=torch.autograd.grad(yprima,i,create_graph=True)[0]
        suma+=pesos[contador]*(f*yprimaprima+y)**2
        contador +=1
    # Acá se usa alpha = 100
    suma+=100*(redDinamica(x0))**2
    suma+=100*(1-redDinamica(xPi))**2
    perdidavar = suma
    return suma

def revisador():
    return varPerdidaCondicionParada<0.01*len(puntosPrueba)


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
            plt.plot(puntosGrafica,np.sin(puntosGrafica),label = "referencia",linestyle="-.")
            plt.plot(7,np.sin(7),marker = '|',ms= 10, label = 'Fin del dominio')
            plt.legend()
            plt.title(f"{i} epochs")
            plt.savefig(filename)
            plt.close(figura)
            nombreParaGuardarRedIntermedia = "estados/uniforme "+str(i)+".tar"
            torch.save(redDinamica.state_dict(),nombreParaGuardarRedIntermedia)



# %%
ahora = datetime.datetime.now()
filename = f"terminados/{ahora.year}-{ahora.month}-{ahora.day}-{ahora.hour}-{ahora.minute}-{ahora.second}-soluciones_dinamicas.tar"
try:
    archivo = open(filename,"xb")
except:
    archivo = open(filename,"wb")
for ronda_solucion in range(2):
    optimizer = torch.optim.Adam(redDinamica.parameters(), lr=1e-3)
    registro_perdida=[]
    registro_promedio=[]
    registro_tiempo = []
    tiempoInicial = time.time()
    i = 0
    termino = False
    while  not termino and tiempoInicial+3600*10>time.time() :
        # Compute prediction and loss
        loss = perdidaConPesos()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        actualizarPuntosConPesos()
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
            filename = f"graficas/{ahora.year}-{ahora.month}-{ahora.day}-{ahora.hour}-{ahora.minute}-PRDN-Epoch:{i}-{ronda_solucion}"
            plottear(filename)
        i+=1
    metodoTradicional.append(ultimaComparacion(
                                                registro_tiempo,
                                                registro_perdida,
                                                registro_promedio,
                                                f"uniforme {i}"
        ))
    print("TERMINO")
    plottear(f"graficas/{ahora.year}-{ahora.month}-{ahora.day}-{ahora.hour}-{ahora.minute}-TRDN-Epoch:{i}-{ronda_solucion}")
    redDinamica = NeuralNetworkPrueba().to(device)
    inicializarPuntos() 
pickle.dump(metodoTradicional,archivo)
archivo.close

# %%
