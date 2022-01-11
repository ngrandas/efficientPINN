# %% [markdown]
# # Solución de movimiento armonico simple a traves de PINN
# 
# El objetivo de este cuaderno es explicar la solución del movimiento armonico simple a traves del uso de PINNS.
# 
# ## Orden
# * declaracion del problema
# * importación de librerias
# * declaración de la red neuronal
# * Declaración y explicación del conjunto de puntos sobre el cual se va a optimizar la función.
# 
# ### To do
# * Ya se hicieron funciones para usar pesos pero no se han implementado
# * Hablar de diferenciación automatica

# %% [markdown]
# ## Problema
# Se desea resolver la siguiente ecuación diferencial con las siguientes condiciones de frontera.
# $$y'' = y$$
# $$y(0) = 0$$
# $$y(\frac{\pi}{2}) = 1$$
# En el dominio
# $${x : x\in \mathbb{R} ^ 1 :  x\in [0,7]}$$
# Residual
# Sea $y_r(x)$ la respuesta de la red neuronal a la entrada $x$ y $y_r''(x)$ la segunda respuesta de la derivada con respecto a x de la red se va a usar la siguiente función de perdida. Acá los puntos $x_i$ pertenecen a un conjunto $\mathbf{X}$ que sera descrito más adelante.
# $$L = \sum_{i=1}^n{(y_r''(x_i)-y_r(x_i))^2}+\alpha * (y_r(0)^2+(y_r(\frac{\pi}{2}) - 1)^2)$$

# %% [markdown]
# ## Importación de las librerias
# Se usan las siguientes librerias por las sigueintes razones
# | libreria Usada | Razón |
# |---|---|
# |Torch|Ofrece documentación más accesible que Julia y tiene modos de diferenciación automatica que facilitan el trabajo|
# |torch.nn|Con este modulo se declara la clase con la que se genera la red neuronal|
# |matplotlib|Graficación |
# |random|Generación de puntos con una distribución Normal|
# |time|Seguimiento del tiempo tomado por cada proceso |
# |Typing|Generación de listas con tipos de datos estáticos|
# |pickle|serialización y guardado de variables|
# |OS| manejo de carpetas|

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

# %% [markdown]
# ## Declaración de la red neuronal
# 
# Se va a usar una red normal con 5 capas de 50 neuronas

# %%
os.chdir("/home/UANDES/n.grandas/pythontesis/pythontesis/v3") # Esta linea hace que el cuaderno corra sobre el directorio deseado
device = "cuda" # Esta linea hace que el cuaderno se ejecute con la gpu de envidia
dtype = torch.float64 # Esta linea hace que el tipo de dato sean floats de 64 bits
estaResolviendo = True
buscandoLearningRate = True


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

redDinamica = NeuralNetworkPrueba().to(device) # Acá se declara la función de la red neuronal y se pone a correr en gpu

# %% [markdown]
# ## Declaración y explicación del conjunto de puntos sobre el cual se va a optimizar la función.
# 
# ### Orden interno
# 1. Introducción y justificación para la exitencia del algoritmo
# 2. Variables usadas en el algoritmo
# 3. Funciones usadas en el algoritmo
# 4. Algoritmo
# 5. Explicación del algoritmo
# 
# ### Introducción y justificación para la exitencia del algoritmo
# 
# Para empezar hay que tomar en cuenta que una gran mayoria de las ecuaciones diferenciales usadas son homogeneas y tienen una cantidad infinita de respuestas posibles (la cantidad de respuestas en este caso es del tamano de $\mathbb{R}^2$ porque la ecuación diferencial es de segundo grado ). desafortunadamente esto lleva a que la respuesta $y = 0$ sea una solución para estas ecuaciones. En este caso la respuesta que queremos es diferente de 0 y está dada a partir de las condiciones de frontera. 
# 
# A la hora de resolver la ecuación diferencial mediante PINNs hay que escoger un conjunto de puntos sobre el cual se va a optimizar la función. El acercamiento predeterminado para PINNs es escoger varios sets de puntos aleatorios en el dominio. No obstante, en experimentación previa se noto que esto es un proceso poco eficiente. Esto se debe a que el optimizador va a intentar llevar los puntos hacia la solución de la ecuación diferencial más cercana. Lo anterior significa que en las etapas iniciales de la optimización los puntos lejanos a las condiciones de frontera van a converger a soluciones que si bien pueden pertenecer al conjunto de soluciones de la ecuación diferencial, no son la ecuación diferencial que se necesita. Esto lleva a que se este optimizando parametros para llegar a una forma inutil. Con el objetivo de eliminar este problema se propone el siguiente método
# 
# Para resolver el problema anterior se propone un método en el cual no se resuelven los puntos que esten alejados de las condiciones de frontera hasta que la solución de la ecuación diferencial se acerque a ellos. Para lograr esto se propone empezar con puntos que esten cercanos a la condición de frontera inicial e ir moviendo estos puntos de prueba mientras se va a resolviendo la ecuación diferencial. De esta forma se espera que todos los puntos convergan a la solución esperada y no a otra posible  solución de la ecuación diferencial. A continuación se detalla el algoritmo para la selección De estos puntos.
# 
# ### Variables usadas en el algoritmo
# 
# 
# 
# |Variable|valor Inicial|Interpretación|
# |---|---|---|
# |$\mu$|0|Es el punto en el cual se encuentra el solucionador, es una variable que varia mientras avanza el solucionador|
# |$\sigma$|0.2|Es la desviación estandar con la cual se ponen puntos cerca a la frontera definida por $\mu$|
# |$n_{max}$|200|Es el largo máximo que puede tener el conjunto $\mathbf{X}$|
# |$n_{ret}$|60|El numero de puntos que se borran del conjunto $\mathbf{X}$|
# |$n_{pas}$|50|El numero de puntos que se añaden al conjunto $\mathbf{X}$|
# |$n_{ini}$|20|Numero de puntos con los cuales se inicializa el conjunto $\mathbf{X}$|
# |$c_1$|$10^{-6}$|Un coeficiente que define linealmente que tan rapido se avanza con respecto a la rata de solución |
# |$c_2$|1|Un criterio para definir la agresividad con la cual se quiere avanzar con respecto a la solución|
# |$L_i$|dinamico|Valor de la perdida obtenida en la iteración i|
# |$x_{inf}$|0|Limite inferior del dominio sobre ,el cual se va a resolver la ecuación diferencial|
# |$x_{sup}$|0|Limite superior del dominio sobre ,el cual se va a resolver la ecuación diferencial|
# 
# ### Funciones usadas en el algoritmo
# 1. Funcion limitadora del dominio:
# 
#     Esta es una función hecha para que si el algoritmo recorre todo el dominio sin converger, este pueda volver al punto inicial  $l : \mathbb{R}\rightarrow \mathbb{R}.$ $$ l(x) = mod(x_{sup}-x_{inf},x-x_{inf})+x_{inf} $$
# 2. relación generadora de puntos
#     f ($\mu $,$\sigma$) retorna un punto aleatorio con distribución normal de promedio $\mu$ y desviación estandar $\sigma $
# 
# ### Algoritmo
# 
# Se inicializan los siguientes valores en $\mu _0 = 0, \sigma = 0.05, c_1 = 1e-6, c_2 = 1$. Luego de esto se inicializa un conjunto con $n_{ini}$ puntos repartidos uniformemente a traves del dominio. Estos puntos solo estan para asegurar que la perdida sea diferente de 0 en el primer momento de la optimización.
# Despues de esto para cada momento i+1 en el tiempo
# 1. Se define $ \mu_{i+1} = \mu_{i} + \frac{c_1}{L_i^{c_2}} $
# 2. Se genera un punto nuevo tal que $ x_{i+1} = l(g(\mu_{i+1},\sigma))$
# 3. Se anade el elemento $x_{i+1}$ al conjunto $\mathbf{X}$
# 4. Si $|\mathbf{X}|>n_{max}$ aleatoriamente se retiran $n_{ret}$ elementos del conjunto.
# 5. Si se retiran elementos del conjunto, Se anaden $n_{pas}$ elementos al conjunto. Cada uno de estos elementos se elige como un numero aleatorio entre 0 y $\mu_{i+1}$
# 6. Se guarda el valor de $L(x_{i+1})$
# 
# 
# ### Explicación de los pasos del algoritmo
# <i>Se eligen los valores del paso 0 a partir de un poco de prueba de error, se esta realizando una aproximación numerica con polinomios de chebyscheff para justificar mejor la elección de estos parametros. </i>
# 
# Para empezar hay que tener en cuenta que el algoritmo divide sobre el valor de la perdida. Para evitar que este valor empiece en 0, se eligen 20 puntos repartidos aleatoriamente dentro del dominio. Despues de esto, se va a empezar el proceso de anadir puntos. Un indicador de que tan bien se esta resolviendo la ecuación diferencial es la función de perdida. Consecuentemente se quiere que los puntos avancen cuando disminuya la función de perdida. Por esto se tomo $\Delta \mu  = \frac{c_1}{L^c_2}$ . 
# 
# Desafortunadamente el algoritmo hasta ese punto aumenta significativamente la complejidad computacional en el tiempo del problema. Esto se debe a que la velocidad de las epocas de entrenamiento depende linealmente del largo del conjunto. Consecuentemente no es ideal aumentar el largo del conjunto de forma indefinida y proporcional a las epocas de entrenamiento. Es por esto que en el paso 4 se introduce un filtro tal que el tamaño del conjunto sea limitado.
# 
# Este filtro no es un filtro tan convencional, se eliminan más puntos de los que se añaden. Esto se debe a que si se añadiera la misma cantidad de puntos que se elimina, en promedio la cantidad de puntos cerca a la frontera se reducirian en una proporción $n_{ret}/n_{max}$ con cada iteración. A traves de experimentación se noto que estos puntos se reducian de manera excesivamente rápida. Esto es resuelto haciendo que despues del filtro el tamano del conjunto sea unas unidades (10 unidades funciona) menor al limite impuesto por $n_{max}$. De esta forma la cantidad de puntos cerca a la fronterea se reduciran en una proporcion menor. Esta proporcion la indica la ecuación a continuación.
# 
# Despues de una reducción de puntos del paso 4 en la iteración i+1 se espera que la proporción de puntos cerca a la frontera sea reducida. Para expresar esto, la cantidad de puntos cercanos a la frontera seran representados por la variable $\eta_{i+1}$. Estos se reduzca de la siguiente forma: $$\frac{\eta_{i+1}}{\eta_{i}} = \frac{n_{ret}}{n_{max}}$$ No obstante, dado que este paso solo ocurre cada $n_{ret} - n_{pas}$ actualizaciones, el cambio esperado por la proporción de puntos estaría dado por la siguiente formula.
# 
# $$\eta_{i+n_{ret} - n_{pas}} = n_{ret} - n_{pas} + \frac{n_{ret}}{n_{max}} * \eta_{i}$$
# 
# De esta forma el limite cuando i tiende a infinito, el numero de puntos cerca a la frontera no tiende a 0. <i><b>¿valdría la pena colocar una gráfica de como i converge a un valor diferente de 0?</b></i>
# 
# Adicionalmente, se estan añadiendo puntos nuevos en zonas que en teoría ya fueron resueltas. Si bien este acercamiento pareceria reducir la eficiencia del algoritmo, hay que tomar en cuenta que el problema esta siendo resuelto mediante metodos de descenso de gradiente. Consecuentemente con el objetivo de que al resolver puntos actuales no se eliminen los resultados anteriores, se colocan puntos en las soluciones anteriores. De esta forma, cuando se altere la respuesta de la red en puntos anteriores, el descenso de gradiente las devolvera a la solución anterior. 
# 
# #### Falta explicar
#  * Uso de funciones de peso
#  * por que se usan los numeros de largo del vector, retiro y posicion de puntos.
#  * Tradeoff entre velocidad y estabilidad

# %%
puntos = torch.linspace(0,7,20, device = device) #
puntos.requires_grad = True
perdidavar = 1e9
promAct = 0 
xsup = 7

def limitador(x):
    xint = x
    while (xint>xsup):
        xint -= xsup
    return xint
    
c2 = 1
c1 = 1e-6 #este coeficiente es para alentizar el avance con respecto a la función de perdiad
sigma = 0.2
def actualizarPuntos():
    # Esta función va actualizar los puntos. Para mantener un balance entre puntos nuevos y puntos anteriores
    #  tambien va a cubir el espacio recorrido por puntos anteriores. Un objetivo de esta función es correr 
    # con un numero bajo de puntos
    global perdidavar,puntos,puntosCreados,promAct,c1,c2,sigma
    
    nuevosPuntos = puntos.detach().tolist()
    promAct = limitador(promAct+c1/(perdidavar*len(puntos))**c2)  # acá se da el avance en promedio
    #print("promedio actual ", promAct)
    #print("perdida actual  ", perdidavar)
    #print("avance          ", coeficiente/perdidavar*len(puntos))
    nuevoPunto = max(min(rd.normalvariate(promAct,sigma),xsup),0) # acá se da el avance como una distribución normal
    nuevosPuntos.append(nuevoPunto)    
    tamTensor = 120
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

pesos = [1]*20
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
    promAct = limitador(promAct+c1/(perdidavar/len(puntos))**c2)  # acá se da el avance en promedio
    nuevoPunto = max(min(rd.normalvariate(promAct,sigma),xsup),0) # acá se da el avance como una distribución normal
    nuevosPuntos.append(nuevoPunto) 
    nuevosPesos.append(1)   
    tamTensor = 40
    borra = 20
    anade = 10
    if len(nuevosPuntos)>tamTensor:
        for i in range(borra):
            indice = rd.randint(0,tamTensor-borra-2)
            nuevosPuntos.pop(indice)
            nuevosPesos.pop(indice)
        for i in range(anade):
            nuevosPuntos.append(rd.uniform(0,promAct))
            nuevosPesos.append(20)
    puntos = torch.tensor(nuevosPuntos,device = device)
    puntos.requires_grad = True
    pesos = nuevosPesos



# %%


# %% [markdown]
# ## Declaración de la función de perdida dentro del código
# ### Orden
# 1. Problema a Resolver
# 2. Notas de la función de perdida
# 
# ### Problema
# #### Este problema fue copiado y pegado de la sección inicial
# 
# Se desea resolver la siguiente ecuación diferencial con las siguientes condiciones de frontera.
# $$y'' = y$$
# $$y(0) = 0$$
# $$y(\frac{\pi}{2}) = 1$$
# En el dominio
# $${x : x\in \mathbb{R} ^ x\in {0,7}}$$
# Residual
# Sea $y_r(x)$ la respuesta de la red neuronal a la entrada x y $y_r''(x)$ la segunda respuesta de la derivada con respecto a x de la red se va a usar la siguiente función de perdida. Acá los puntos $x_i$ pertenecen al conjunto $\mathbf{X}$.
# $$L = \sum_{i=1}^n{(y_r''(x_i)-y_r(x_i))^2}+\alpha * (y_r(0)^2+(y_r(\frac{\pi}{2}) - 1)^2)$$ 
# 
# ### Notas de la función de perdida.
# Para empezar hay que notar que se estan usando tecnicas de aprendizaje profundo. En general estas tecnicas dependen de optimizar una función con una gran cantidad de parametros a traves de su gradiente. Para evitar que esto sea un problema significativo en terminos de complejidad computacional se usa un algoritmo de diferenciación automatica para que la complejidad de generar el vector gradiente no dependa de la dimensión del vector gradiente.
# 
# Lo anterior significa que se tiene que usar una implementación que tenga en cuenta la diferenciación automatica, para esto se usa Pytorch. El requisito que pide esta libreria es que se indiquen cuales variables van a ser necesitadas a la hora de la diferenciación. Estas están identificadas con el atributo requires_grad = True
# 
# ### Paralelización
# La función que genera cálcula la perdida deberia de correr en tiempo $O(1)$ para ciertos $n_{max}$, pues esta corriendo en GPU y se supone que hay al menos 3000 nucleos libres, este no es el caso. Debido a la restricción GIL de python, que no permite utilizar más de un hilo en simultaneo, no fue posible lograr que la esta función corra adecuadamente. Se intento utilizar torchscript de forma extensiva, este es un sublenguaje de python que corre en C y en teoría podia paralelizar. Desafortunadamente este lenguaje no pudo mantener los arboles asociados a la diferenciación avanzada para las derivadas de segundo grado. Consecuentemente fue utilizada una versión sin paralelismo que corria en tiempo $O(n)$. 

# %%
def perdida2():
    global perdidavar # esta variable guarda la perdida en una ubicación afuera de la función.
    x0 =  torch.tensor([0.0],device=device,requires_grad =True) # Ubicación en X de la primera condición de frontera
    # el 0.0 es importante para que pytorch lo identifique como un float.
    # acá el argumento device se usa para indicar que el vector debe guardarse en la memoria de la gpu
    # Acá el rgumento requires_grad = true se usa para indicar que la variable importa a la hora de la diferenciación 
    xPi = torch.linspace(3.14159/2,1,2,device = device) 
    xPi =  torch.tensor([xPi[0]],device=device,requires_grad =True) # Ubicación en X se la segunda condición de frontera
    suma = 0
    for j in puntos:
        i = torch.tensor([j],device=device,requires_grad =True) 
        # hay que meter el valor en X dentro de un vector para que pytorch lo tome como una operación de algebra lineal
        # lo anterior es necesario con la función nn.Sequential()
        y = redDinamica(i) 
        yprima=torch.autograd.grad(y,i,create_graph=True)[0]
        yprimaprima=torch.autograd.grad(yprima,i,create_graph=True)[0]
        suma+=(yprimaprima+y)**2
    # Acá se usa alpha = 100
    suma+=100*(redDinamica(x0))**2
    suma+=100*(1-redDinamica(xPi))**2
    perdidavar = suma
    return suma

def perdidaConPesos():
    global perdidavar # esta variable guarda la perdida en una ubicación afuera de la función.
    x0 =  torch.tensor([0.0],device=device,requires_grad =True) # Ubicación en X de la primera condición de frontera
    # el 0.0 es importante para que pytorch lo identifique como un float.
    # acá el argumento device se usa para indicar que el vector debe guardarse en la memoria de la gpu
    # Acá el rgumento requires_grad = true se usa para indicar que la variable importa a la hora de la diferenciación 
    xPi = torch.linspace(3.14159/2,1,2,device = device) 
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
        suma+=pesos[contador]*(yprimaprima+y)**2
        contador +=1
    # Acá se usa alpha = 100
    suma+=100*(redDinamica(x0))**2
    suma+=100*(1-redDinamica(xPi))**2
    perdidavar = suma
    return suma

# %% [markdown]
# ## Condiciones de parada
# 
# Se van a usar dos condiciones de parada, la primera va a ser con respecto al tiempo y la segunda va a ser con respecto a un criterio de calidad 
# 
# ### Tiempo
# 
# Para la condición de parada con respecto al tiempo se pondra un limite de 1.75 horas por caso de solución. Este criterio esta ahí porque el computador con el cual se resuelve este problema no esta disponible de forma ilimitada. Consecuentemente se desea que si se pasa el limite de tiempo por lo menos se guarden las respuestas y avances.
# 
# ### Criterio de calidad
# 
# La segunda condición de parada va a ocurrir en caso de que la perdida promedio para cada punto sea de 0.001. Para evitar evaluar dos veces la misma función se guarda el primer resultado y se usa en el segundo <b>Este numero fue escogido arbitrareamente. Creo que funciona bien pero no tengo ni idea de como justificar esta elección</b>
# 
# <b> Hacer cambio para que no se calcule dos veces el check</b>

# %%
# Puntos Prueba : 
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

def revisador():
    return varPerdidaCondicionParada<0.01*len(puntosPrueba)

# %% [markdown]
# ## Serialización de soluciones y análisis de rata de aprendizaje.
# La serialización es el proceso mediante el cual se guardan algunas variables para que en análisis futuros se puedan usar sin necesitar correr todo el código. 
# ### subclases de pytorch
# Los objetos de pytorch que se van a serializar vienen con un método que lo permite hacer de forma sencilla. En este caso se usa este método (save_state_dict())
# ### Resto de objetos
# En este caso se usa la libreria Pickle para este proposito. Para serializar primero se debe abrir un archivo que contenga la información. Para esto se utiliza la estructura try/except. esta va a intentar crear y abrir un archivo con el nombre en filename, si esto falla intentara editar sobre un archivo con ese nombre.

# %%
if estaResolviendo:
    os.chdir("/home/UANDES/n.grandas/pythontesis/pythontesis/v4")
    filename = "RESULTADOS/registrosPerdidas/registro2.tesis"
else:
    os.chdir("/home/UANDES/n.grandas/pythontesis/pythontesis/basura")
    filename = "RESULTADOS/registrosPerdidas/registro2.tesis"


try:
    archivo = open(filename,"xb")
except:
    archivo = open(filename,"wb")


# %% [markdown]
# ## Análisis de los distintos parametros de los que depende el algoritmo.
# ### Orden 
# 1. Tasa de aprendizaje

# %% [markdown]
# ### Tasa de aprendizaje
# 
# En la siguiente celda se prueban 20 tasa de aprendizaje distribuidas logaritmicamente entre $10^{-2}$ y $10^{-5}$. los resultados de esto con el registro de las perdidas para cada tasa de aprendizaje quedan serializados en el siguiente orden.
# 
# 1. Separador llamado "tasa de aprendizaje"
# 2. tasa de aprendizaje
# 3. registro de la evolución de la perdida con respecto al tiempo
# 4. Indicador de que termino o no termino
# 5. Tiempo en el que termino
#  

# %%
if buscandoLearningRate:
    for learningRate in list(np.logspace(np.log10(5e-2),np.log10(1e-6),12)):
        pickle.dump("rata de aprendizaje",archivo)
        pickle.dump(learningRate,archivo)
        epochs = 8000
        optimizer = torch.optim.Adam(redDinamica.parameters(), lr=learningRate)
        registro_perdida=[]
        tiempoInicial = time.time()
        i = 0
        termino = False
        while time.time()-tiempoInicial<3600*1.75 and not termino and estaResolviendo:
            # Compute prediction and loss
            loss = perdidaConPesos()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                #print(loss.item()/len(puntos))
                #print("supermax",promAct)
                actualizarPuntosConPesos()
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
c1 = 1e-5
c2 = 1
sigma = 0.05
if False:
        optimizer = torch.optim.Adam(redDinamica.parameters(), lr=1.6e-3)
        registro_perdida=[]
        tiempoInicial = time.time()
        i = 0
        termino = False
        while time.time()-tiempoInicial<3600*1.75 and not termino:
            # Compute prediction and loss
            loss = perdidaConPesos()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                #print(loss.item()/len(puntos))
                #print("supermax",promAct)
                actualizarPuntosConPesos()
            if i % 20 == 0:
                registro_perdida.append(perdidaParaRevisar().item()/len(puntos))
                print(f"Perdida global  :{registro_perdida[-1]}")
                print(f"promedio        : {promAct.item()}")
                print(f"perdida Interna : {perdidavar.item()}")
                termino = revisador()
            if i % 200 == 0:
                puntosViejos = [round(puntos[i].item(),4) for i in range(20) if pesos[i] == 20]
                puntosNuevos = [round(puntos[i].item(),4) for i in range(20) if pesos[i] == 1]
                pesosPuteados = [peso for peso in pesos if peso != 1 and peso != 20]
                if len(pesosPuteados)>0:
                    print(f"pesos Puteados: {pesosPuteados}")
                print(f"puntosActuales: {puntosNuevos}")
                print(f"puntosAnteriores  : {puntosViejos}")
            i+=1


# %%
pesos

# %% [markdown]
# ### Analisis de la tasa de aprendizaje
# 
# <b> la siguiente celda esta mal porque hay un error en el solucionador que llame para generar los datos. Este error ya fue corregido y se estan generando nuevos datos</b>

# %%
os.chdir("/home/UANDES/n.grandas/pythontesis/pythontesis/v4/RESULTADOS/estados")
archivos = [archivito for archivito in os.listdir() if archivito[-8:]=="3.58.tar"  ]
print(archivos)
plt.figure(figsize = (20,10))
alpha = 0
for nombre in archivos:
    redDinamica.load_state_dict(torch.load(nombre))
    ygrafica = []
    puntosGrafica = torch.linspace(0,10,250)
    for i in puntosGrafica:
        ytemp=redDinamica(torch.tensor([i],device = device))
        ygrafica.append(ytemp.cpu().detach().numpy()[0])
        #ygrafica.append(ytemp.detach().numpy()[0])
    import numpy as np
    puntosGrafica = np.linspace(0,10,250)
    if nombre != 'senoParalelo0 epochs-3.58.tar':
        epoca = int(nombre[12:16])
    else:
        epoca = 0
    alpha = epoca/3600
    plt.plot(puntosGrafica,ygrafica,label = epoca,alpha = alpha)
    
plt.plot(puntosGrafica,np.sin(puntosGrafica),label = "referencia",LineStyle="-.")
plt.legend()

# %% [markdown]
# ### Proceso de optimización
# 
# Se va a optimizar para encontrar como el tiempo que toma llegar a una solución es afectado por las siguientes variables. Dado que el tiempo que toma llegar a una solución es una variable no deterministica, no se considera que sea apropiado usar metodos que utilicen gradiente para esta optimización. Esto se debe a que las gradientes que se van a encontrar van a estar fuertemente influenciadas por el ruido. Para remediar esto se va a optimizar a partir de interpolaciones con procesos iterativos de polinomios de chebyscheff. Esto se va a explicar a continuación.
# 
# * Rata de aprendizaje de ADAM
# * $\sigma$
# * $c_1$
# * $c_2$
# 
# 
# ### Punto de chebyscheff
# 
# Para el proceso de interpolación se utilizará el método de interpolación de lagrange. Este método generea una un polinomio $p_n(x)$ de grado n cuyo valor es igual a una función $f(x)$ en al menos n puntos. Los puntos en los cuales son iguales se pueden elegir, deben ser diferentes entre sí y estan representados en el siguiente conjunto {$x_0 , ... , x_n$}. Dado que la interpolación no necesariamente es exacta para todos los puntos dentro del dominio, puede existir un error para ciertos puntos del dominio. La elección de los puntos afecta significativamente la distribución del error dentro de la función. A continuación se describe la formula asociada al error numérico por aproximación polinomial y la razón por la cual se eligen los puntos de chebyscheff. 
# 
# 
# $$f(x) - p_n(x) = \frac{f^{(n + 1)}(\xi)}{(n+1)!} \blue {\prod_{i=0}^n{x-x_i}} $$
# 
# En la formula anterior $\xi$ es un punto dentro del intervalo de interpolación. Actualmente no se conoce cual punto es ni como asignarlo a algún punto deseado. Consecuentemente la unica parte controlable es la que esta resaltada en azul. Los puntos que minimizan el valor de la multiplicatoria son los puntos de chebyscheff. La comprobación de esto se puede encontrar en está <a href = https://math.okstate.edu/people/binegar/4513-F98/4513-l16.pdf > página </a>. Dado que se esta minimizando la parte controlable del error, se considera que se está minimizando el error. la formula para los puntos {x_0,..., x_n} se encuentra a continuación.
# 
# Para cada punto $x_i$ en el intervalo [-1,1]
# 
# $$x_i = cos (\frac{\pi (k + 0.5)}{n}) \text{ para } k = 0,...,n $$
# 
# Para pasar al intervalo [a,b] se introduce la transformación $t(x) = \frac{a + b + (b-a)x}{1}$  que lleva a la siguiente expresión.
# 
# $$x_i = \frac{a + b + (b-a)*cos (\frac{\pi (k + 0.5)}{n})}{2} \text{ para } k = 0,...,n $$
# 
# 

# %% [markdown]
# ## Implementación de los puntos de chebyscheff y optimización de parametros
# 
# La implementación del algoritmo de optimización propuesto se hace de la siguiente forma
# 1. Se define la función que da los puntos de chebyscheff
# 2. Se Prueba el algoritmo para encontrar el minimo de la función $f(x) = \frac{1}{1+(x-1)^2}$ cuyo minimo esta en $x = 1$
# 3. Se implementa el algoritmo para encontrar el parametro $c_2$ que menos tiempo toma para converger
# 4. Se implementa el algoritmo para encontrar el parametro $c_1$ que menos tiempo toma para converger
# 5. Se implementa el algoritmo para encontrar el parametro $\sigma$ que menos tiempo toma para converger. 
# 6. Se implementa el paso 3, 4 y 5 dos veces más para ya que $\sigma$, $c_1$ y $c_2$ deberían depender entre si.
# 

# %% [markdown]
# ### Implementación de la función de puntos y Prueba del algoritmo

# %%
def puntos_chebyscheff(n,a,b):
    inicializacion = np.array(range(n+1))
    inicializacion = np.flip(inicializacion)
    return (a +b + (b-a)*np.cos(np.pi/n*(inicializacion + 0.5)))/2

# Demostración del método de optimización
extremoInferiorIntervalo = 0.25
extremoSuperiorIntervalo = 2
minimoEnRonda : np.float128 = 3
for rondaDeOptimizacion in range(3):
    anchoIntervalo = extremoSuperiorIntervalo - extremoInferiorIntervalo
    tiemposTomados = []
    valores = puntos_chebyscheff(5,extremoInferiorIntervalo,extremoSuperiorIntervalo)
    f_prueba = lambda x: -(1/(1+(np.tan(x)-1)**2))
    for valor in valores:
        # redefinición de la función de actualización con el nuevo valor

        # Generación de variable con el tiempo de referencia para la nueva solución.
        # Solución con el nuevo valor.

        # Toma del tiempo de la solución.
        tiemposTomados.append(f_prueba(valor))
        pass

    # Interpolación polinomica con los tiempos de la nueva solución.
    
    interpolacion = np.polynomial.Polynomial.fit(valores,tiemposTomados,len(valores)-1)
    
    x_grafica = np.linspace(extremoInferiorIntervalo,extremoSuperiorIntervalo)
    plt.plot(valores,tiemposTomados,linewidth = 0, marker = "*",label = "Puntos de prueba de la función",ms = 15)
    plt.plot(x_grafica,f_prueba(x_grafica),label = "función original",linestyle = "--",linewidth=3)
    plt.plot(x_grafica,interpolacion(x_grafica),label = "Interpolación de la función")
    plt.xlim(extremoInferiorIntervalo,extremoSuperiorIntervalo)
    # minimo en ronda es el punto en el cual se encontro el minimo DENTRO del polinomio.
    minimoEnRonda : np.float128 = x_grafica[np.argmin(f_prueba(x_grafica))]
    # Reducción del intervalo de busqueda a un 25% para la siguiente ronda de optimización.
    extremoInferiorIntervalo = minimoEnRonda - 0.2/2 * anchoIntervalo
    extremoSuperiorIntervalo = minimoEnRonda + 0.2/2 * anchoIntervalo
    plt.plot(minimoEnRonda,interpolacion(minimoEnRonda),marker = "2", ms = 25,label = "Minimo encontrado",linewidth = 0)
    print("Minimo Encontrado",minimoEnRonda)
    print("extremos",[extremoInferiorIntervalo,extremoSuperiorIntervalo])
    print("cantida de puntos de prueba",len(valores))
    print("grado del polinomio",interpolacion.degree())
    plt.legend()
    plt.show()

print("Minimo Encontrado : ",minimoEnRonda)
print("Minimo Teorico : ",1)
print("Error          : ",minimoEnRonda-1)


# %%
#objetoInutil = np.polynomial.Polynomial.fit(valores,tiemposTomados)
valores
tiemposTomados

# %%
if estaResolviendo:
    os.chdir("/home/UANDES/n.grandas/pythontesis/pythontesis/v4")
    filename = "RESULTADOS/parametros/parametrosEncontrados"
else:
    os.chdir("/home/UANDES/n.grandas/pythontesis/pythontesis/basura")
    filename = "RESULTADOS/parametros/parametrosEncontrados"

print(os.getcwd())
try:
    archivo = open(filename,"xb")
except FileExistsError:
    archivo = open(filename,"wb")

# %% [markdown]
# ### Implementación con el parametro C1 y C2
# ### To do
# * Generar clase que reciba los parametros guardados en solucionar y generar clase que reciba los parametros guardados por ronda de optimizacion

# %%
def puntos_chebyscheff(n,a,b):
    inicializacion = np.array(range(n+1))
    inicializacion = np.flip(inicializacion)
    return (a +b + (b-a)*np.cos(np.pi/n*(inicializacion + 0.5)))/2

class statSolucion():
    def __init__(self,variable,learningRate,registro_perdida,registro_promedio,tiempo):
        self.variable = variable
        self.learningRate = learningRate
        self.registro_perdida = registro_perdida
        self.registro_promedio = registro_promedio
        self.tiempo = tiempo
        self.epocasPorSample = 20 

# La función solucionar funciona para hacer el código de abajo más legible
# Esta función solo soluciona el problema enunciado en el principio.
learningRate = 1.62e-3

def solucionar(serializar = False,variable :str = "",archivo = archivo,ronda = -1):
    # El modulo serializar no esta listo y necesitara un archivo y un cerrador del archivo
    ronda = str(ronda)
    if serializar == True and archivo.closed: 
        raise Exception("El archivo en la serialización esta vacio")
    optimizer = torch.optim.Adam(redDinamica.parameters(), lr=learningRate)
    if serializar:
       registro_perdida=[]
       registro_promedio=[]
    tiempoInicial = time.time()
    i = 0
    termino = False
    while time.time()-tiempoInicial<3600*1.75 and not termino and estaResolviendo:
        # Compute prediction and loss
        loss = perdidaConPesos()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            #print(loss.item()/len(puntos))
            #print("supermax",promAct)
            actualizarPuntosConPesos()
        if i % 20 == 0:
            if serializar:
                registro_perdida.append(perdidaParaRevisar().item()/len(puntos))
                registro_promedio.append(promAct)
            else:
                inutil = perdidaParaRevisar().item()/len(puntos)
            termino = revisador()
            print(varPerdidaCondicionParada)
        if i % 300 == 0:
            if serializar:
                nombreParaGuardarRedIntermedia = "RESULTADOS/parametros/redesIntermedias/senoParalelo"+str(i)+" epochs variable "+variable+"ronda"+ronda +".tar"
                torch.save(redDinamica.state_dict(),nombreParaGuardarRedIntermedia)
        i+=1
    if termino:
        tiempo = time.time()-tiempoInicial
    else:
        tiempo = -1

    if serializar:
        nombreParaGuardarRedFinal = "RESULTADOS/parametros/redesIntermedias/senoParalelo"+str(i)+" epochs variable "+variable+"ronda"+ronda +".tar"
        torch.save(redDinamica.state_dict(),nombreParaGuardarRedFinal)
        estadisticasSolucion = statSolucion(variable = variable,
                                            learningRate = learningRate,
                                            registro_perdida=registro_perdida,
                                            registro_promedio=registro_promedio,
                                            tiempo=tiempo)
        pickle.dump("estadisticas de la solución",archivo)
        pickle.dump(estadisticasSolucion,archivo)
        return estadisticasSolucion
    else:
        return None
estaResolviendo = True
solucionar(serializar=True,variable = "prueba",ronda=-2)


# %%
class optimizacionParametro():
    def __init__(self,variable):
        self.intervalos = []
        self.minimos = []
        self.soluciones = []
        self.polinomios = []
        self.puntosPolTiempo = []
        self.puntosPolX = []
        self.variable = variable
    
    def anadir(self,intervalo,minimo,polinomio,statSol,puntoPolTiempo,puntoPolX):
        self.intervalos.append(intervalo)
        self.minimos.append(minimo)
        self.polinomios.append(polinomio)
        self.soluciones.append(statSol)
        self.puntosPolTiempo.append(puntoPolTiempo)
        self.puntosPolX.append(puntoPolX)
        
estadisticasC1 = optimizacionParametro("C1")
estadisticasC2 = optimizacionParametro("C2")
estadisticassigma= optimizacionParametro("sigma")

# %%
for rondaDeOptimizacionGlobal in range(1):
    # Optimización para el parametro C1
    extremoInferiorIntervalo = 0.25e-6
    extremoSuperiorIntervalo = 5e-5
    minimoEnRonda : np.float128 = 3
    for rondaDeOptimizacion in range(3):
        anchoIntervalo = extremoSuperiorIntervalo - extremoInferiorIntervalo
        tiemposTomados = []
        valores = puntos_chebyscheff(5,extremoInferiorIntervalo,extremoSuperiorIntervalo)
        estadisticasDeSolucionesEnPDC= []
        for valor in valores:
            # redefinición de la función de actualización con el nuevo valor
            c1 = valor
            # Generación de variable con el tiempo de referencia para la nueva solución.
            t0 = time.time()
            # Reestablecimiento de los parametros de la red Neuronal
            redDinamica = NeuralNetworkPrueba().to(device)
            # Solución con el nuevo valor.
            estadisticasDeSolucionesEnPDC.append( solucionar(serializar=True))
            # Toma del tiempo de la solución.
            tiemposTomados.append(time.time()-t0)

        # Interpolación polinomica con los tiempos de la nueva solución.
        interpolacion = np.polynomial.Polynomial.fit(valores,tiemposTomados,len(valores)-1)
        
        #Generación de un eje X para las gráficas y obetner el estimado del valor minimo.
        x_grafica = np.linspace(extremoInferiorIntervalo,extremoSuperiorIntervalo,200)

        # minimo en ronda es el punto en el cual se encontro el minimo DENTRO del polinomio.
        minimoEnRonda : np.float128 = x_grafica[np.argmin(interpolacion(x_grafica))]
        c1 = minimoEnRonda

        plt.plot(valores,tiemposTomados,linewidth = 0, marker = "*",label = "Puntos de prueba de la función")
        plt.plot(x_grafica,interpolacion(x_grafica),label = "Interpolación de la función")
        plt.xlim(extremoInferiorIntervalo,extremoSuperiorIntervalo)
        plt.plot(minimoEnRonda,interpolacion(minimoEnRonda),marker = "+", ms = 5,label = "Minimo encontrado")
        plt.title(f"c1, Ronda de optimización interna{rondaDeOptimizacion} , Ronda de optimización externa {rondaDeOptimizacionGlobal}")
        nombreGuardar = f"c1, int {rondaDeOptimizacion},ext{rondaDeOptimizacionGlobal}"
        plt.legend()
        plt.savefig("RESULTADOS/parametros/graficas/"+nombreGuardar,dpi = 300)

        # Serialización de la solución
        estadisticasC1.anadir([extremoInferiorIntervalo,extremoSuperiorIntervalo],
                                minimoEnRonda,
                                interpolacion,
                                estadisticasDeSolucionesEnPDC,
                                tiemposTomados,
                                valores)
    


        # Reducción del intervalo de busqueda a un 20% para la siguiente ronda de optimización.
        extremoInferiorIntervalo = minimoEnRonda - 0.2/2 * anchoIntervalo
        extremoSuperiorIntervalo = minimoEnRonda + 0.2/2 * anchoIntervalo
        
        plt.legend()
        plt.show()
        
        
    
    # Optimización para el parametro C2
    extremoInferiorIntervalo = 0.5
    extremoSuperiorIntervalo = 2
    minimoEnRonda : np.float128 = 3
    for rondaDeOptimizacion in range(3):
        anchoIntervalo = extremoSuperiorIntervalo - extremoInferiorIntervalo
        tiemposTomados = []
        valores = puntos_chebyscheff(5,extremoInferiorIntervalo,extremoSuperiorIntervalo)
        estadisticasDeSolucionesEnPDC= []
        for valor in valores:
            # redefinición de la función de actualización con el nuevo valor
            c2 = valor
            # Generación de variable con el tiempo de referencia para la nueva solución.
            t0 = time.time()
            # Reestablecimiento de los parametros de la red Neuronal
            redDinamica = NeuralNetworkPrueba().to(device)
            # Solución con el nuevo valor.
            estadisticasDeSolucionesEnPDC.append( solucionar(serializar=True))
            # Toma del tiempo de la solución.
            tiemposTomados.append(time.time()-t0)

        # Interpolación polinomica con los tiempos de la nueva solución.
        interpolacion = np.polynomial.Polynomial.fit(valores,tiemposTomados,len(valores)-1)
        
        #Generación de un eje X para las gráficas y obetner el estimado del valor minimo.
        x_grafica = np.linspace(extremoInferiorIntervalo,extremoSuperiorIntervalo,200)

        # minimo en ronda es el punto en el cual se encontro el minimo DENTRO del polinomio.
        minimoEnRonda : np.float128 = x_grafica[np.argmin(interpolacion(x_grafica))]
        c2 = minimoEnRonda
        plt.plot(valores,tiemposTomados,linewidth = 0, marker = "*",label = "Puntos de prueba de la función")
        plt.plot(x_grafica,interpolacion(x_grafica),label = "Interpolación de la función")
        plt.xlim(extremoInferiorIntervalo,extremoSuperiorIntervalo)
        plt.plot(minimoEnRonda,interpolacion(minimoEnRonda),marker = "+", ms = 10,label = "Minimo encontrado")
        plt.title(f"c2, Ronda de optimización interna{rondaDeOptimizacion} , Ronda de optimización externa {rondaDeOptimizacionGlobal}")
        nombreGuardar = f"c2, int {rondaDeOptimizacion},ext{rondaDeOptimizacionGlobal}"
        plt.legend()
        plt.savefig("RESULTADOS/parametros/graficas/"+nombreGuardar,dpi = 300)
        plt.show()

        # Serialización de la solución
        estadisticasC2.anadir([extremoInferiorIntervalo,extremoSuperiorIntervalo],
                                minimoEnRonda,
                                interpolacion,
                                estadisticasDeSolucionesEnPDC,
                                tiemposTomados,
                                valores)
    


        # Reducción del intervalo de busqueda a un 20% para la siguiente ronda de optimización.
        extremoInferiorIntervalo = minimoEnRonda - 0.2/2 * anchoIntervalo
        extremoSuperiorIntervalo = minimoEnRonda + 0.2/2 * anchoIntervalo
        
        


    # Optimización para el parametro sigma

    #Dado que sigma es un parametro real, se va a usar la transformación sigma = 10**x y se va a optimizar x
    extremoInferiorIntervalo = -2
    extremoSuperiorIntervalo = 0
    minimoEnRonda : np.float128 = 3
    for rondaDeOptimizacion in range(3):
        anchoIntervalo = extremoSuperiorIntervalo - extremoInferiorIntervalo
        tiemposTomados = []
        valores = puntos_chebyscheff(5,extremoInferiorIntervalo,extremoSuperiorIntervalo)
        estadisticasDeSolucionesEnPDC= []
        for valor in valores:
            # redefinición de la función de actualización con el nuevo valor
            sigma = 10**valor
            # Generación de variable con el tiempo de referencia para la nueva solución.
            t0 = time.time()
            # Reestablecimiento de los parametros de la red Neuronal
            redDinamica = NeuralNetworkPrueba().to(device)
            # Solución con el nuevo valor.
            estadisticasDeSolucionesEnPDC.append( solucionar(serializar=True))
            # Toma del tiempo de la solución.
            tiemposTomados.append(time.time()-t0)

        # Interpolación polinomica con los tiempos de la nueva solución.
        interpolacion = np.polynomial.Polynomial.fit(valores,tiemposTomados,len(valores)-1)
        
        #Generación de un eje X para las gráficas y obetner el estimado del valor minimo.
        x_grafica = np.linspace(extremoInferiorIntervalo,extremoSuperiorIntervalo,200)

        # minimo en ronda es el punto en el cual se encontro el minimo DENTRO del polinomio.
        minimoEnRonda : np.float128 = x_grafica[np.argmin(interpolacion(x_grafica))]
        sigma = 10**minimoEnRonda
        plt.plot(valores,tiemposTomados,linewidth = 0, marker = "*",label = "Puntos de prueba de la función")
        plt.plot(x_grafica,interpolacion(x_grafica),label = "Interpolación de la función")
        plt.xlim(extremoInferiorIntervalo,extremoSuperiorIntervalo)
        plt.plot(minimoEnRonda,interpolacion(minimoEnRonda),marker = "+", ms = 5,label = "Minimo encontrado")
        plt.title(f"sigma, Ronda de optimización interna{rondaDeOptimizacion} , Ronda de optimización externa {rondaDeOptimizacionGlobal}")
        nombreGuardar = f"sigma, int {rondaDeOptimizacion},ext{rondaDeOptimizacionGlobal}"
        plt.legend()
        plt.savefig("RESULTADOS/parametros/graficas/"+nombreGuardar,dpi = 300)
        plt.show()

        # Serialización de la solución
        estadisticassigma.anadir([extremoInferiorIntervalo,extremoSuperiorIntervalo],
                                minimoEnRonda,
                                interpolacion,
                                estadisticasDeSolucionesEnPDC,
                                tiemposTomados,
                                valores)
    


        # Reducción del intervalo de busqueda a un 20% para la siguiente ronda de optimización.
        extremoInferiorIntervalo = minimoEnRonda - 0.2/2 * anchoIntervalo
        extremoSuperiorIntervalo = minimoEnRonda + 0.2/2 * anchoIntervalo
        
        


print(f"empezo a serializar en {os.getcwd()}")
print("c1",c1)
print("c2",c2)
print("sigma",sigma)

pickle.dump("Estadisticas c1")
pickle.dump(estadisticasC1,archivo)
pickle.dump("Estadisticas c2")
pickle.dump(estadisticasC2,archivo)
pickle.dump("Estadisticas sigma")
pickle.dump(estadisticassigma,archivo)
print("Termino de serializar")

archivo.close()

# %% [markdown]
# ## Comparación contra la solución no optimizada y la solución a partir de puntos aleatorios
# 
# Para la comparación contra la solución no optimizada se va a observar la convergencia en tiempo de cada función. Para esto se va a grabar como avanza la solucion vs las epocas y vs el tiempo.

# %%

# %% [markdown]
# 

# %% [markdown]
# 


