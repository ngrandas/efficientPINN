#!/usr/bin/env python
# coding: utf-8

# # OPTIMIZACIÓN DE LA SOLUCIÓN DE ECUACIONES DIFERENCIALES A TRAVES DE REDES NEURONALES
# 
# ## Resumen
# 
# En el siguiente proyecto se generará la solución de un caso de redes neuronales con inferencias físicas, posteriormente se analizará el proceso de entrenamiento de la red y se propondrán maneras de acelerarlo. Para esto se intentará optimizar la rata de aprendizaje usada por ADAM y se usaran funciones de distribución de probabilidad dinámicas.  
# 
# ## Orden
# 
# * Resumen.
# * Orden.
# * Introducción.
# * Metodología
# * Explicación de la razón por la cual se están implementando funciones de densidad de probabilidad dinámicas.
# * Declaración del problema.
# * Importación de librerías.
# * Declaración de la red neuronal.
# * Declaración y explicación del conjunto de puntos sobre el cual se va a optimizar la función.
# * Declaración de la función de perdida dentro del código.
# * Condiciones de parada.
# * Serialización de las soluciones y análisis de rata de aprendizaje.
# * Análisis de los distintos parámetros de los que depende el algoritmo.
# * Resultados de Rata de aprendizaje.
# 
# ## Introducción
# 
# La solución de ecuaciones diferenciales es un reto fundamental para la ingeniería y la física. Esto se puede ver en la avaluación de compañías como Ansys o Star CCM+, que se dedican a esto. En los últimos años la prominencia de las técnicas de inteligencia artificial ha aumentado significativamente, una pregunta natural ante este escenario es ¿Cómo es que estas pueden ayudar a solucionar ecuaciones diferenciales? Respondiendo a esta pregunta surgen las PINNs (Physically Inferred Neural Networks), cuyo acrónimo significa redes neuronales con inferencias físicas.
# 
# Estas son redes neuronales entrenadas para interpolar o generar respuestas a ecuaciones diferenciales. Tradicionalmente, las interpolaciones de las redes neuronales solo tienen en cuenta los resultados de una simulación o respuesta anterior. Por otro lado, las simulaciones tradicionales solo tienen en cuenta las condiciones de frontera. Estas usualmente no usan resultados de simulaciones pasadas ni experimentos de laboratorio o reglas empíricas. Las PINN por el contrario, pueden solucionar/interpolar usando resultados de simulaciones pasadas, experimentos de laboratorios, condiciones de frontera y/o reglas empíricas.
# 
# Esto se debe a que las PINNs usan el descenso de gradiente en reemplazo de los sistemas de ecuaciones lineales usados por los métodos de runge-kutta. Esto permite que la inclusión de condiciones al solucionador, pues no hay una matriz/sistema de ecuaciones para sobredimensionalizar. Adicionalmente, estas redes usan algoritmos como la diferenciación automática, que le permite a las redes diferenciar con respecto a muchas variables con mejor precisión y velocidad que la diferenciación numérica tradicional.
# 
# Desafortunadamente, la solución de ecuaciones diferenciales a través de PINN resulta altamente costosa computacionalmente para una gran variedad de implementaciones. Consecuentemente a través del cuaderno se intentará reducir este costo. Esto se hará a través de un proceso en el que primero se intenta entender como es que se converge a las respuestas y luego se intentará optimizar este proceso.
# 
# ## Metodología usada.
# 
# El proyecto empezó con el objetivo de simular la ecuación de Navier Stokes para el caso conocido como "Backwards Flowing Step". Esto falló debido a problemas con la implementación del termino de aceleración local en el módulo de diferenciación automática. Desafortunadamente, el proceso de solución era demasiado costoso computacionalmente como para poder iterar nuevas respuestas. Consecuentemente, se intentó solucionar una ecuación significativamente más simple, optimizarla y aplicar esos resultados al caso de Navier Stokes.
# 
# Durante el proceso de optimización, se consideró que podía ser más valioso explorar a fondo el efecto de realizar cambios sobre el entrenamiento de la red. Consecuentemente el objetivo del proyecto cambió a explorar estos efectos. La ecuación que se decidió resolver fue la de movimiento armónico simple. Se tomó esta decisión porque la respuesta de esta está limitada por el intervalo [1,-1]; es una ecuación de grado mayor a 1; su respuesta es conocida y su forma es simple. A continuación, se explicará la importancia de cada una de estas razones.
# 
# Para empezar, se deseaba que la respuesta estuviese dada entre 1 y -1 pues así se podía notar fácilmente si existía algún problema con números negativos (No hubo, por eso no se menciona en el resto del documento). Esto podría llegar a ser un problema con otro tipo de función de activación dentro de la red neuronal como RELU o la función sigmoide debido a que estas dos funciones solo producen números mayores a 0. 
# 
# Por otra parte, se quería que la ecuación fuese de grado mayor a 1 para poder probar el módulo de diferenciación automática. Esto no fue trivial, pues la implementación de pytorch presentaba inconsistencias a la hora de optimizar con respecto a variables generadas por un Hessiano. Consecuentemente se quería comprobar que se pudiese realizar el proyecto con segundas derivadas.
# 
# Continuando con la respuesta conocida y forma simple. Se perdió mucho tiempo probando e intentando arreglar la función de Navier Stokes. Consecuentemente, se considero un mejor uso del tiempo usar una función fácil de probar y arreglar.
# 
# Después de programar el entrenamiento de la ecuación diferencial, se realizo el proceso durante 10 horas en un computador lento e inestable. Lo anterior incitó a agilizar el proceso. Dado que la solución tomaba mucho tiempo en converger, se pudieron observar patrones a través de gráficas que mostraban el progreso de la solución. 
# 
# Luego de esto, tomando inspiración de la forma en la que los algoritmos de búsqueda lograron ganar eficiencia a través de minimizar los pasos innecesarios, se intentó este acercamiento con la optimización de zonas innecesarias en esta solución. Para esto se implementaron funciones de densidad de probabilidad dinámicas y se maximizó la tasa de aprendizaje.
# 
# 
# ## Problema
# Se desea resolver la siguiente ecuación diferencial con las siguientes condiciones de frontera.
# $$y'' = y$$
# $$y(0) = 0$$
# $$y(\frac{\pi}{2}) = 1$$
# En el dominio
# $${x : x\in \mathbb{R} ^ 1 :  x\in [0,7]}$$
# Residual
# Sea $y_r(x)$ la respuesta de la red neuronal a la entrada $x$ y $y_r''(x)$ la segunda respuesta de la derivada con respecto a x de la red se va a usar la siguiente función de perdida. Acá los puntos $x_i$ pertenecen a un conjunto $\mathbf{X}$ que será descrito más adelante.
# $$L = \sum_{i=1}^n{\Big(w_i * (y_r''(x_i)-y_r(x_i))\Big)^2}+\alpha * (y_r(0)^2+(y_r(\frac{\pi}{2}) - 1)^2)$$
# 
# ## Explicación de la razón por la cual se están implementando funciones de densidad de probabilidad dinámicas.
# 
# Para realizar esto es conveniente hablar de como es que las PINNs resuelven ecuaciones diferenciales. Esto se hará a través de una explicación de porque se usan redes neuronales; como es que estas resuelven ecuaciones diferenciales; el problema generado por el paso anterior y finalmente como es que se espera que las funciones de probabilidad de densidad dinámicas lo resuelvan.
# 
# Empezando con las redes neuronales, estás no son más que funciones con una gran cantidad de parámetros que tienen la capacidad de acercarse mucho a las formas de otras funciones. Esto significa que para una cantidad adecuada de parámetros/neuronas se debería poder obtener virtualmente cualquier función deseada. Lo anterior implica que a la hora de resolver una ecuación diferencial, con una red neuronal que tenga una cantidad adecuada de parámetros, se podrá obtener una función que satisface la ecuación diferencial. Adicionalmente, en este caso se están usando funciones diferenciables como las funciones de activación (Parte de las neuronas). En resumen, se están usando redes neuronales porque se espera que estas tengan la capacidad de tomar la forma de la respuesta a la ecuación diferencial.
# 
# La pregunta que surge naturalmente ante el parrado anterior es ¿Cómo lograr que la red tome la forma de la función deseada? Esto se realiza mediante del uso del aprendizaje por descenso de gradiente estocástico. Este es el nombre de una familia de algoritmos que resuelve el problema a través de variar una gran cantidad de parámetros dentro de la red. Para realizar esto el algoritmo utiliza una función (en este caso la función $L$) que le indique que tan bien funciona cada combinación de parámetros. De está forma el algoritmo llega a los parámetros que maximicen la función anterior. 
# 
# La forma tradicional para realizar lo anteriormente descrito consiste en:
# 1. Seleccionar un conjunto de puntos aleatorios que pertenezcan al dominio donde se espera resolver la ecuación diferencial.
# 2. Evaluar que tan bien estos puntos cumplen la ecuación diferencial.
# 3. Evaluar que tan bien se cumplen las condiciones de frontera.
# 4. Actualizar los parámetros para mejorar los puntajes de los puntos 2 y 3 a través de descenso de gradiente.
# 5. Repetir el paso 1.
# 
# Un aspecto fundamental del proceso es que para cada punto sobre el que se está optimizando, el punto solo va a intentar mejorar basándose en la información de la ecuación diferencial. Esto significa que en este caso especifico, para cualquier punto $x_i$ este solo va a intentar que la concavidad sea igual a $y$ es decir que $y''(x_i) =  y(x_i)$. Lo anterior significa que a este punto no le importa la condición de frontera del problema sino su valor actual de $y$ y $y''$ pues la red neuronal solo quiere igualar a esos valores. Para que el punto llegue a la solución deseada, los puntos con un valor de x un poco menores deben tener los valores de $y,y'$ y $y''$ correctos. Esto significa que la solución se va a propagar desde las condiciones de frontera al resto del dominio.
# 
# Lo anterior se puede ver desde una serie de Taylor y una inducción. Suponiendo que se tienen infinitos puntos separados por distancias muy pequeñas. Si para un punto, el valor de la función y las derivadas es correcto, según las expansiones de Taylor, el valor de $y$ del siguiente punto será correcto. Consecuentemente, si las derivadas de este punto son correctas, se espera que el siguiente punto sea correcto. Dado que el punto puesto sobre la condición de frontera va a tener el valor de $y$ apropiado, se espera que después de ajustar las derivadas acordes a la ecuación diferencial, el siguiente a este se llegue al valor de $y$ correcto. Después de esto se espera que este lleve al siguiente a la posición correcta y que este proceso se propague hasta que se resuelva la ecuación diferencial.
# 
# Para explicar el problema asociado al procedimiento listado anteriormente, hay que tomar en cuenta que una gran mayoría de las ecuaciones diferenciales usadas son homogéneas y tienen una cantidad infinita de respuestas posibles (la cantidad de respuestas en este caso es del tamaño del conjunto $\mathbb{R}^2$ porque la ecuación diferencial es de grado 2). desafortunadamente esto lleva a que existan muchas respuestas que resuelven la ecuación diferencial pero no las condiciones de frontera. Ejemplos de esto son  $y(x) = 0 \text{ ; } y(x) = cos (x) \text{ ; } y(x) = 0.1 sin(x) $. Lo anterior significa que se necesita resolver la ecuación diferencial que resuelva las condiciones de frontera. 
# 
# El método tradicional primero se acopla a las condiciones de frontera y luego propaga la ecuación diferencial al resto del dominio. Esto significa que no todas las partes del dominio se van a resolver de manera simultanea. Las regiones más cercanas a las condiciones de frontera se van a resolver antes que el resto. Después de que estas se resuelvan, debido a que se están usando funciones diferenciables, eventualmente se va a propagar la solución correcta al resto del dominio.
# 
# Lo anterior significa que si se están colocando puntos sobre las zonas en las cuales la solución no se ha propagado, se está perdiendo el tiempo y esfuerzo computacional. Consecuentemente se espera que al utilizar una función de densidad de probabilidad que no coloque los puntos de manera uniformemente repartida, sino solo cerca a las condiciones de frontera, se logre aumentar la velocidad de convergencia.
# 

# ## Importación de las librerías
# Se usan las siguientes librerías por las siguientes razones
# 
# | librería Usada | Razón |
# |---|---|
# |Torch|Ofrece documentación más accesible que Julia y tiene modos de diferenciación automatica que facilitan el trabajo|
# |torch.nn|Con este modulo se declara la clase con la que se genera la red neuronal|
# |matplotlib|Graficación |
# |random|Generación de puntos con una distribución Normal|
# |time|Seguimiento del tiempo tomado por cada proceso |
# |Typing|Generación de listas con tipos de datos estáticos|
# |pickle|serialización y guardado de variables|
# |OS| manejo de carpetas|
# 

# In[2]:


import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import time
from typing import List
import pickle
import os 


# ## Declaración de la red neuronal
# 
# Se va a usar una red normal con 5 capas de 50 neuronas. Esto se eligió basandose en la literatura.

# In[3]:


os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
os.chdir('v6') # Esta linea hace que el cuaderno corra sobre el directorio deseado
device = "cuda" # Esta linea hace que el cuaderno se ejecute con la gpu de envidia
dtype = torch.float64 # Esta linea hace que el tipo de dato sean floats de 64 bits
estaResolviendo = True # Esta variable se usa para controlar si se quiere que se resuelvan las redes neuronales.
# Se pone en false si no se desea resolver nada.
buscandoLearningRate = False # Esta variable se usa para controlar si se requiere hacer el analisis de learning rate.
# como este analisis toma mucho tiempo, el default es falso
serializar = True

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
#redDinamicaCPU = NeuralNetworkPrueba().to("")


# ## Declaración y explicación del conjunto de puntos sobre el cual se va a optimizar la función.
# 
# ### Orden interno
# 1. Introducción y justificación para la existencia del algoritmo
# 2. Variables usadas en el algoritmo
# 3. Funciones usadas en el algoritmo
# 4. Algoritmo
# 5. Explicación del algoritmo
# 
# ### Introducción y justificación para la existencia del algoritmo para la colocación de puntos
# 
# A la hora de resolver la ecuación diferencial mediante PINNs hay que escoger un conjunto de puntos sobre el cual se va a optimizar la función. El acercamiento predeterminado para PINNs es escoger varios sets de puntos aleatorios en el dominio. No obstante, en experimentación previa se noto que esto es un proceso poco eficiente. Esto se debe a que el optimizador va a intentar llevar los puntos hacia la solución de la ecuación diferencial más cercana. Lo anterior significa que en las etapas iniciales de la optimización los puntos lejanos a las condiciones de frontera van a converger a soluciones que si bien pueden pertenecer al conjunto de soluciones de la ecuación diferencial, no son la ecuación diferencial que se necesita. Esto lleva a que se este optimizando parámetros para llegar a una forma inútil. Con el objetivo de eliminar este problema se propone el siguiente método
# 
# Para resolver el problema anterior se propone un método en el cual no se resuelven los puntos que estén alejados de las condiciones de frontera hasta que la frontera de lo que está resuelto de la ecuación diferencial se acerque a ellos. Para lograr esto se propone empezar con puntos que estén cercanos a la condición de frontera inicial e ir moviendo estos puntos hacia el resto del dominio. De esta forma se espera que todos los puntos converjan a la solución esperada y no a otra posible  solución de la ecuación diferencial. A continuación se detalla el algoritmo para la selección De estos puntos.
# 
# ### Variables usadas en el algoritmo para la colocación de puntos
# 
# 
# 
# |Variable|valor Inicial|Interpretación|
# |---|---|---|
# |$\mu$|0|Es el punto en el cual se encuentra el solucionador, es una variable que varia mientras avanza el solucionador|
# |$\sigma$|0.2|Es la desviación estándar con la cual se ponen puntos cerca a la frontera definida por $\mu$|
# |$n_{max}$|200|Es el largo máximo que puede tener el conjunto $\mathbf{X}$|
# |$n_{ret}$|60|El numero de puntos que se borran del conjunto $\mathbf{X}$|
# |$n_{pas}$|50|El numero de puntos que se añaden al conjunto $\mathbf{X}$|
# |$n_{ini}$|20|Numero de puntos con los cuales se inicializa el conjunto $\mathbf{X}$|
# |$c_1$|$10^{-6}$|Un coeficiente que define linealmente que tan rapido se avanza con respecto a la rata de solución |
# |$c_2$|1|Un criterio para definir la agresividad con la cual se quiere avanzar con respecto a la solución|
# |$L_i$|dinámico|Valor de la perdida obtenida en la iteración i|
# |$x_{inf}$|0|Limite inferior del dominio sobre ,el cual se va a resolver la ecuación diferencial|
# |$x_{sup}$|0|Limite superior del dominio sobre ,el cual se va a resolver la ecuación diferencial|
# 
# ### Funciones usadas en el algoritmo para la colocación de puntos
# 1. Funcion limitadora del dominio:
# 
#     Esta es una función hecha para que si el algoritmo recorre todo el dominio sin converger, este pueda volver al punto inicial  $l : \mathbb{R}\rightarrow \mathbb{R}.$ $$ l(x) = mod(x_{sup}-x_{inf},x-x_{inf})+x_{inf} $$
# 2. relación generadora de puntos
#     f ($\mu $,$\sigma$) retorna un punto aleatorio con distribución normal de promedio $\mu$ y desviación estandar $\sigma $
# 
# ### Algoritmo para la colocación de puntos
# 
# Se inicializan los siguientes valores en $\mu _0 = 0, \sigma = 0.05, c_1 = 1e-5, c_2 = 1$. Luego de esto se inicializa un conjunto con $n_{ini}$ puntos repartidos uniformemente a través del dominio. Estos puntos solo están para asegurar que la perdida sea diferente de 0 en el primer momento de la optimización. Al final del documento se elige un set diferente de estos numeros y se justifica mediante de un proceso de optimización.
# 
# Despues de esto para cada momento i+1 en el tiempo
# 1. Se define $ \mu_{i+1} = \mu_{i} + \frac{c_1}{L_i^{c_2}} $
# 2. Se genera un punto nuevo tal que $ x_{i+1} = l(g(\mu_{i+1},\sigma))$
# 3. Se anade el elemento $x_{i+1}$ al conjunto $\mathbf{X}$
# 4. Si $|\mathbf{X}|>n_{max}$ aleatoriamente se retiran $n_{ret}$ elementos del conjunto.
# 5. Si se retiran elementos del conjunto, Se añaden $n_{pas}$ elementos al conjunto. Cada uno de estos elementos se elige como un numero aleatorio entre 0 y $\mu_{i+1}$
# 6. Se guarda el valor de $L(x_{i+1})$
# 
# 
# ### Explicación de los pasos del algoritmo
# 
# Para empezar hay que tener en cuenta que el algoritmo divide sobre el valor de la perdida. Para evitar que este valor empiece en 0, se eligen 20 puntos repartidos aleatoriamente dentro del dominio. Después de esto, se va a empezar el proceso de añadir puntos. Un indicador de que tan bien se esta resolviendo la ecuación diferencial es la función de perdida. Consecuentemente se quiere que los puntos avancen cuando disminuya la función de perdida. Por esto se tomo $\Delta \mu  = \frac{c_1}{L^c_2}$ . 
# 
# Desafortunadamente el algoritmo hasta ese punto aumenta significativamente la complejidad computacional en el tiempo del problema. Esto se debe a que la velocidad de las épocas de entrenamiento depende linealmente del largo del conjunto. Consecuentemente no es ideal aumentar el largo del conjunto de forma indefinida y proporcional a las épocas de entrenamiento. Es por esto que en el paso 4 se introduce un filtro tal que el tamaño del conjunto sea limitado.
# 
# Este filtro no es un filtro tan convencional, se eliminan más puntos de los que se añaden. Esto se debe a que si se añadiera la misma cantidad de puntos que se elimina, en promedio la cantidad de puntos cerca a la frontera se reducirían en una proporción $n_{ret}/n_{max}$ con cada iteración. A través de experimentación se noto que estos puntos se reducían de manera excesivamente rápida. Esto es resuelto haciendo que después del filtro el tamaño del conjunto sea unas unidades (10 unidades funciona) menor al limite impuesto por $n_{max}$. De esta forma la cantidad de puntos cerca a la frontera se reducirán en una proporción menor. Esta proporción la indica la ecuación a continuación.
# 
# Después de una reducción de puntos del paso 4 en la iteración i+1 se espera que la proporción de puntos cerca a la frontera sea reducida. Para expresar esto, la cantidad de puntos cercanos a la frontera serán representados por la variable $\eta_{i+1}$. Estos se reduzca de la siguiente forma: $$\frac{\eta_{i+1}}{\eta_{i}} = \frac{n_{ret}}{n_{max}}$$ No obstante, dado que este paso solo ocurre cada $n_{ret} - n_{pas}$ actualizaciones, el cambio esperado por la proporción de puntos estaría dado por la siguiente formula.
# 
# $$\eta_{i+n_{ret} - n_{pas}} = n_{ret} - n_{pas} + \frac{n_{ret}}{n_{max}} * \eta_{i}$$
# 
# De esta forma el limite cuando i tiende a infinito, el numero de puntos cerca a la frontera no tiende a 0.
# 
# Adicionalmente, se están añadiendo puntos nuevos en zonas que en teoría ya fueron resueltas. Si bien este acercamiento parecería reducir la eficiencia del algoritmo, hay que tomar en cuenta que el problema esta siendo resuelto mediante métodos de descenso de gradiente. Consecuentemente con el objetivo de que al resolver puntos actuales no se eliminen los resultados anteriores, se colocan puntos en las soluciones anteriores. De esta forma, cuando se altere la respuesta de la red en puntos anteriores, el descenso de gradiente las devolverá a la solución anterior. Finalmente, en estos puntos la variable $w_i$ dentro de la sumatoria toma un valor de 20, de esta forma, estos puntos pesan 20 veces más que el resto de los puntos dentro de la función de perdida. Lo anterior les da prioridad a estos puntos dentro de la función.
# 

# In[4]:


puntos = torch.linspace(0,7,20, device = device) #Inicialización de los puntos de prueba
puntos.requires_grad = True # Atributo que indica que estos puntos se usan para la optimización 
perdidavar = 1e9 #Variable posicionada acá para que quede fuera del espacio de memoria de las funciones siguientes
promAct = 0  # Lo mismo que arriba
xsup = 7 # limite superior del dominio

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
# Para empezar hay que notar que se están usando técnicas de aprendizaje profundo. En general estas técnicas dependen de optimizar una función con una gran cantidad de parámetros a través de su gradiente. Para evitar que esto sea un problema significativo en términos de complejidad computacional se usa un algoritmo de diferenciación automática para que la complejidad de generar el vector gradiente no dependa de la dimensión del vector gradiente.
# 
# Lo anterior significa que se tiene que usar una implementación que tenga en cuenta la diferenciación automática, para esto se usa Pytorch. El requisito que pide esta librería es que se indiquen cuales variables van a ser necesitadas a la hora de la diferenciación. Estas están identificadas con el atributo requires_grad = True
# 
# ### Paralelización
# La función que genera calcula la perdida debería de correr en tiempo $O(1)$ para ciertos $n_{max}$, pues esta corriendo en GPU y se supone que hay al menos 3000 núcleos libres, este no es el caso. Debido a la restricción GIL de Python, que no permite utilizar más de un hilo en simultaneo, no fue posible lograr que la esta función corra adecuadamente. Se intento utilizar torchscript de forma extensiva, este es un sublenguaje de python que corre en C y en teoría podía paralelizar. Desafortunadamente este lenguaje no pudo mantener los arboles asociados a la diferenciación avanzada para las derivadas de segundo grado. Consecuentemente fue utilizada una versión sin paralelismo que corría en tiempo $O(n)$. 

# In[5]:


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


# ## Condiciones de parada
# 
# Se van a usar dos condiciones de parada, la primera va a ser con respecto al tiempo y la segunda va a ser con respecto a un criterio de calidad 
# 
# ### Tiempo
# 
# Para la condición de parada con respecto al tiempo se pondrá un limite de 1.75 horas por caso de solución. Este criterio esta ahí porque el computador con el cual se resuelve este problema no esta disponible de forma ilimitada. Consecuentemente se desea que si se pasa el limite de tiempo por lo menos se guarden las respuestas y avances.
# 
# ### Criterio de calidad
# 
# La segunda condición de parada va a ocurrir en caso de que la diferencia promedio con respecto a la solución analítica sea de 0.001. Para evitar evaluar dos veces la misma función se guarda el primer resultado y se usa en el segundo

# In[6]:


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


# ## Serialización de soluciones y análisis de rata de aprendizaje.
# La serialización es el proceso mediante el cual se guardan algunas variables para que en análisis futuros se puedan usar sin necesitar correr todo el código. 
# ### subclases de pytorch
# Los objetos de pytorch que se van a serializar vienen con un método que lo permite hacer de forma sencilla. En este caso se usa este método (save_state_dict())
# ### Resto de objetos
# En este caso se usa la libreria Pickle para este propósito. Para serializar primero se debe abrir un archivo que contenga la información. Para esto se utiliza la estructura try/except. esta va a intentar crear y abrir un archivo con el nombre en filename, si esto falla intentara editar sobre un archivo con ese nombre.

# In[7]:


if estaResolviendo:
    os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
    os.chdir("v5")
    filename = "RESULTADOS/registrosPerdidas/registro.tesis"
else:
    os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
    os.chdir("basura")
    filename = "RESULTADOS/registrosPerdidas/registro.tesis"


try:
    archivo = open(filename,"xb")
except:
    archivo = open(filename,"wb")


# ## Análisis de los distintos parametros de los que depende el algoritmo.
# ### Orden 
# 1. Tasa de aprendizaje

# ### Tasa de aprendizaje
# 
# En la siguiente celda se prueban 20 tasa de aprendizaje distribuidas logarítmicamente entre $10^{-2}$ y $10^{-5}$. los resultados de esto con el registro de las perdidas para cada tasa de aprendizaje quedan serializados en el siguiente orden.
# 
# 1. Separador llamado "tasa de aprendizaje"
# 2. tasa de aprendizaje
# 3. registro de la evolución de la perdida con respecto al tiempo
# 4. Indicador de que termino o no termino
# 5. Tiempo en el que termino
#  

# In[8]:


if buscandoLearningRate:
    for learningRate in list(np.logspace(np.log10(5e-2),np.log10(1e-6),18)):
        pickle.dump("rata de aprendizaje",archivo) # Escribe un flag dentro del archivo de serialziación
        pickle.dump(learningRate,archivo) # Escribe la tasa de aprendizaje
        optimizer = torch.optim.Adam(redDinamica.parameters(), lr=learningRate) # Se reinicializa el optimizador para cada ciclo
        registro_perdida=[]
        tiempoInicial = time.time()
        i = 0 # i mantiene la cuenta de cuantas epocas se están resolviendo
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
c2 = 1## RESULTADOS DE ANALISIS DE RATA DE APRENDIZAJE.


# ## RESULTADOS DE ANALISIS DE RATA DE APRENDIZAJE.
# 
# ### Tiempo de convergencia vs rata de aprendizaje.
# 
# Los primeros resultados de la rata de convergencia indican que el resultado ideal es alrededor de 1e-3. Estos se pueden ver en la gráfica de la siguiente celda. Los valores probados fueron sugeridos por esta <a href = https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/> fuente </a> Esto es un valor convencional para el optimizador ADAM. Adicionalmente los resultados se ven similares a los de esta <a href = "https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2"> fuentes </a>. Vale la pena recapitular que las estrellas verdes corresponden a los intentos que no cumplieron el objetivo de calidad en 1.75 horas.
# 
# ### Forma de los residuales para distintos grupos de rata de aprendizaje.
# 
# La gráfica de la forma de los residuales para distintos grupos de rata de aprendizaje muestra como se comportaron los residuales o valores de la función de perdida a través del proceso. Las líneas azules corresponden a las ratas de aprendizaje demasiado agresivas. Acá se puede ver como inicialmente logran disminuir la función de perdida pero rápidamente se estancan. Por otro lado las líneas rosadas corresponden a las ratas de aprendizaje demasiado lentas. Acá las líneas nunca dejan de progresar, pero tampoco se acercan a terminar. Finalmente, las líneas amarillas corresponden a los casos en los que se termina antes de 1.75 horas. Hay que notar que para este problema en especifico, es esperable que el solucionador acabe en menos de 5 000 épocas.

# In[9]:


class convergencia1(): #comentario
    def __init__(self,archivo):
        self.rata: float = 0
        self.tiempo: float = -1
        self.perdidas = []
        self.generador(archivo)
    def lectura(self,archivo):
        try:
            leido = pickle.load(archivo)
        except Exception as e:
            if e.args==('Ran out of input',):
                raise Exception("ya no hay nada mas que leer indicado por pickle")
        return leido
    def generador(self,archivo):
        global lecturaDeTiempo
        if not lecturaDeTiempo:
            self.rata = self.lectura(archivo)
        elif self.lectura(archivo) == "rata de aprendizaje":
            self.rata = self.lectura(archivo)
        else:
            raise Exception("Se putio el orden de lectura")
        self.perdidas = self.lectura(archivo)
        if self.lectura(archivo) == "termino en":
            self.tiempo = self.lectura(archivo)
            lecturaDeTiempo = True
        else:
            lecturaDeTiempo = False

convergencias1 = []
# importada de archivos pasados
os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
os.chdir("v4/RESULTADOS/registrosPerdidas")
lecturaDeTiempo = True
archivo = open("registro.tesis","rb")
while True:
    try:
        objetoTemporal = convergencia1(archivo)
        convergencias1.append(objetoTemporal)
    except Exception as e:
        print (e.args)
        print (e.__traceback__)
        break
archivo.close()
lecturaDeTiempo = True
archivo = open("registro2.tesis","rb")
while True:
    try:
        objetoTemporal = convergencia1(archivo)
        convergencias1.append(objetoTemporal)
    except Exception as e:
        print (e.args)
        print (e.__traceback__)
        break
archivo.close()
os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
os.chdir("v5/RESULTADOS/registrosPerdidas")
lecturaDeTiempo = True
archivo = open("registro.tesis","rb")
while True:
    try:
        objetoTemporal = convergencia1(archivo)
        convergencias1.append(objetoTemporal)
    except Exception as e:
        print (e.args)
        print (e.__traceback__)
        break
archivo.close()
print(len(convergencias1))

convergencias1 = sorted(convergencias1,key = lambda x:x.rata)

ratasDeConvergencia1 = [simulacion.rata for simulacion in convergencias1 if simulacion.tiempo != -1]
tiemposDeConvergencia1 = [simulacion.tiempo for simulacion in convergencias1 if simulacion.tiempo != -1]

ratasDeConvergenciasFallidas = [simulacion.rata for simulacion in convergencias1 if simulacion.tiempo == -1]
yTemporal = [max(tiemposDeConvergencia1)]*len(ratasDeConvergenciasFallidas)


eleccion = np.argpartition(tiemposDeConvergencia1,5)
eleccion = eleccion[1]
os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
os.chdir("graficas2")
plt.figure(figsize=(8,6))
plt.loglog(ratasDeConvergencia1,tiemposDeConvergencia1,marker = "o",ms = 6,linewidth = 0,label = "Tiempos tomados para la convergencia")
plt.loglog(ratasDeConvergencia1[eleccion],tiemposDeConvergencia1[eleccion],marker = "+",ms=15,label = "tiempo Elegido para el resto de simulaciones",linewidth = 0)
plt.loglog(ratasDeConvergenciasFallidas,yTemporal,marker = "*",ms=10,label = "Convergencia fallida",linewidth = 0)
plt.title("Resultados del analisis de tasa de aprendizaje")
plt.grid(True,"both")
plt.legend(bbox_to_anchor = (0.8,-0.1))
plt.ylabel("Tiempo tomado para la convergencia")
plt.xlabel("Rata de aprendizaje")
plt.savefig("tiempoDeConvergenciaVsLearningRate.png",dpi = 300, bbox_inches = "tight")
plt.show(block = False)
import scipy.signal as signal
convergieron = [i.perdidas for i in convergencias1 if i.tiempo != -1]
ratasTemporales = [i.rata for i in convergencias1 if i.tiempo != -1]
noConvergieronArriba = [i.perdidas for i in convergencias1 if i.tiempo == -1 and i.rata > 1e-3]
noConvergieronAbajo = [i.perdidas for i in convergencias1 if i.tiempo == -1 and i.rata < 1e-3]

def filtrador(x_):
    b, a = signal.butter(1, 0.075, 'low')
    return signal.filtfilt(b, a, x_)

fig, ax = plt.subplots(figsize = (12,9))


os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
os.chdir("graficas2")
plt.title ("Velocidad de convergencia vs epocas para distintas ratas de aprendizaje")


transparencia = 1

mejorOpcion = 0
for j in range(len(convergieron)):
    i = convergieron[j]
    if convergencias1[eleccion].rata == ratasTemporales[j]:
        mejorOpcion = j
    else:
        senal_filtrada = filtrador(i)
        eje_x = np.linspace(0,len(senal_filtrada)*20,len(senal_filtrada)) # Se tomo un sample cada 300 epocas.  
        ax.semilogy(eje_x,senal_filtrada,color = "#F4B942",label = "Rata de aprendizaje adecuada",alpha = transparencia)

senal_filtrada =  filtrador(convergieron[mejorOpcion],)
eje_x = np.linspace(0,len(senal_filtrada) * 20,len(senal_filtrada)) 
ax.semilogy(eje_x,senal_filtrada,color = "#4059AD",label = "Rata de aprendizaje elegida",alpha = 1,linewidth = 1.5,linestyle = "--")      

for i in noConvergieronArriba:
    senal_filtrada = filtrador(i)
    eje_x = np.linspace(0,len(senal_filtrada)* 20,len(senal_filtrada)) # Se tomo un sample cada 300 epocas.  
    ax.semilogy(eje_x,senal_filtrada,color = "#6B9AC4",label = "Rata de aprendizaje Demasiado Agresiva",alpha = transparencia)

for i in noConvergieronAbajo:
    senal_filtrada = filtrador(i)
    eje_x = np.linspace(0,len(senal_filtrada)*20,len(senal_filtrada)) # Se tomo un sample cada 300 epocas.  
    ax.semilogy(eje_x,senal_filtrada,color = "#FC94AD",label = "Rata de aprendizaje Demasiado Lenta",alpha = transparencia)
ax.set_ylabel("Residuales")
ax.set_xlabel("epocas")
ax.grid(which="both")
display = [10,15,24,len(convergieron)-1]
handles, labels = ax.get_legend_handles_labels()
ax.legend([handle for i,handle in enumerate(handles) if i in display],
      [label for i,label in enumerate(labels) if i in display])
plt.style.context("seaborn-colorblind")
plt.savefig("Convergencia para distintas ratas de aprendizaje",dpi = 300, bbox_inches = "tight")


# ### Analisis de la solución de uno de los casos anteriores.
# 
# <b> la siguiente celda no muestra resultados porque no hay una gpu disponible para que los muestre. Intentar correrla en cpu mata el kernel</b>
# 
# Como se puede ver en la solución de la celda a continuación, inicialmente se converje a la respuesta desde la parte del dominio cercana a las condiciones de frontera y luego esta solución se propaga al resto del dominio.
# 

# In[10]:


os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
os.chdir("v4/RESULTADOS/estados")
archivos = [archivito for archivito in os.listdir() if archivito[-8:]=="4.66.tar"  ]
print(archivos)
plt.figure(figsize = (20,10))
alpha = 0
for nombre in archivos:
    redDinamica.load_state_dict(torch.load(nombre))
    ygrafica = []
    puntosGrafica = torch.linspace(0,10,250)
    for i in puntosGrafica:
        ytemp=redDinamica(torch.tensor([i],device = device))
        ygrafica.append(ytemp.detach().item())
        #ygrafica.append(ytemp.detach().numpy()[0])
    import numpy as np
    puntosGrafica = np.linspace(0,10,250)
    if nombre != 'senoParalelo0 epochs-4.66.tar':
        epoca = int(nombre[12:16])
    else:
        epoca = 0
    alpha = epoca/1e5
    plt.plot(puntosGrafica,ygrafica,label = epoca,alpha = alpha)
    
plt.plot(puntosGrafica,np.sin(puntosGrafica),label = "referencia",linestyle="-.")
plt.legend()


# ### Proceso de optimización
# 
# Se va a optimizar para encontrar como el tiempo que toma llegar a una solución es afectado por las siguientes variables. Dado que el tiempo que toma llegar a una solución es una variable no determinística, no se considera que sea apropiado usar métodos que utilicen gradiente para esta optimización. Esto se debe a que las gradientes que se van a encontrar van a estar fuertemente influenciadas por el ruido. Para remediar esto se va a optimizar a partir de interpolaciones con procesos iterativos de polinomios de chebyscheff. Esto se va a explicar a continuación.
# 
# * Rata de aprendizaje de ADAM
# * $\sigma$
# * $c_1$
# * $c_2$
# 
# ### Punto de Chebyscheff
# 
# Para el proceso de interpolación se utilizará el método de interpolación de LaGrange. Este método genera una un polinomio $p_n(x)$ de grado n cuyo valor es igual a una función $f(x)$ en al menos n puntos. Los puntos en los cuales son iguales se pueden elegir, deben ser diferentes entre sí y están representados en el siguiente conjunto {$x_0 , ... , x_n$}. Dado que la interpolación no necesariamente es exacta para todos los puntos dentro del dominio, puede existir un error para ciertos puntos del dominio. La elección de los puntos afecta significativamente la distribución del error dentro de la función. A continuación se describe la formula asociada al error numérico por aproximación polinómica y la razón por la cual se eligen los puntos de chebyscheff. 
# 
# $$f(x) - p_n(x) = \frac{f^{(n + 1)}(\xi)}{(n+1)!} \blue {\prod_{i=0}^n{x-x_i}} $$
# 
# En la formula anterior $\xi$ es un punto dentro del intervalo de interpolación. Actualmente no se conoce cual punto es ni como asignarlo a algún punto deseado. Consecuentemente la única parte controlable es la que esta resaltada en azul. Los puntos que minimizan el valor de la multiplicatoria son los puntos de chebyscheff. La comprobación de esto se puede encontrar en está <a href = https://math.okstate.edu/people/binegar/4513-F98/4513-l16.pdf > página </a>. Dado que se esta minimizando la parte controlable del error, se considera que se está minimizando el error. la formula para los puntos ${x_0,..., x_n}$ se encuentra a continuación.
# 
# Para cada punto $x_i$ en el intervalo [-1,1]
# 
# $$x_i = cos (\frac{\pi (k + 0.5)}{n}) \text{ para } k = 0,...,n $$
# 
# Para pasar al intervalo [a,b] se introduce la transformación $t(x) = \frac{a + b + (b-a)x}{1}$  que lleva a la siguiente expresión.
# 
# $$x_i = \frac{a + b + (b-a)*cos (\frac{\pi (k + 0.5)}{n})}{2} \text{ para } k = 0,...,n $$
# 

# ## Implementación de los puntos de chebyscheff y optimización de parametros
# 
# La implementación del algoritmo de optimización propuesto se hace de la siguiente forma
# 1. Se define la función que da los puntos de chebyscheff
# 2. Se Prueba el algoritmo para encontrar el mínimo de la función $f(x) = -  \frac{1}{1-(tan(x)-1)^2}$
# 3. Se implementa el algoritmo para encontrar el parámetro $c_2$ que menos tiempo toma para converger
# 4. Se implementa el algoritmo para encontrar el parámetro $c_1$ que menos tiempo toma para converger
# 5. Se implementa el algoritmo para encontrar el parámetro $\sigma$ que menos tiempo toma para converger. 
# 6. Se implementa el paso 3, 4 y 5 dos veces más para ya que $\sigma$, $c_1$ y $c_2$ deberían depender entre si.

# ### Implementación de la función de puntos y Prueba del algoritmo
# 
# En la implementación a continuación se puede ver como el proceso de optimización funciona para el ejemplo dado

# In[32]:


def puntos_chebyscheff(n,a,b):
    inicializacion = np.array(range(n+1))
    inicializacion = np.flip(inicializacion)
    inicializacion = inicializacion[:-1]
    return (a +b + (b-a)*np.cos(np.pi/n*(inicializacion - 0.5)))/2

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
    minimoEnRonda : np.float128 = x_grafica[np.argmin(interpolacion(x_grafica))]
    # Reducción del intervalo de busqueda a un 30% para la siguiente ronda de optimización.
    extremoInferiorIntervalo = minimoEnRonda - 0.3/2 * anchoIntervalo
    extremoSuperiorIntervalo = minimoEnRonda + 0.3/2 * anchoIntervalo
    plt.plot(minimoEnRonda,interpolacion(minimoEnRonda),marker = "2", ms = 25,label = "Minimo encontrado",linewidth = 0)
    print("Minimo Encontrado",minimoEnRonda)
    print("extremos",[extremoInferiorIntervalo,extremoSuperiorIntervalo])
    print("cantidad de puntos de prueba",len(valores))
    print("grado del polinomio",interpolacion.degree())
    print("Minimo Encontrado : ",minimoEnRonda)
    print("Minimo Teorico    : ",0.7853981)
    print("Error Porcentual  : ",(minimoEnRonda-0.7853981)/0.7853981*100)
    plt.legend()
    plt.show(block = False)

print("Minimo Encontrado : ",minimoEnRonda)
print("Minimo Teorico    : ",0.7853981)
print("Error Porcentual  : ",(minimoEnRonda-0.7853981)/0.7853981*100)


# In[39]:


os.chdir("/home/externo/Documents/nico/efficientPINN/pythontesis")
if estaResolviendo:
    os.chdir("v6")
    filename = "RESULTADOS/parametros/parametrosEncontrados"
else:
    os.chdir("basura")
    filename = "RESULTADOS/parametros/parametrosEncontrados"

print(os.getcwd())
try:
    archivo = open(filename,"xb")
except FileExistsError:
    archivo = open(filename,"wb")


# # Falta de respuestas.
# 
# Debido a problemas con la tarjeta gráfica del servidor ninguna de las siguientes celdas se pueden correr y no hay respuestas.

# ### Implementación con el parametro C1 y C2

# In[40]:


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
def solucionar(serializar = False,variable :str = "",archivo = archivo,ronda = -1,learningRate = 1e-3):
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
            #print(varPerdidaCondicionParada)
        if i % 60 == 0:
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
c1 = 1e-4
c2 = 1
sigma = 1


# In[41]:


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


# In[42]:


print(f'valores : {valores}')
print(f'tiempos tomados : {tiemposTomados}')


# In[43]:


ygrafica = []
puntosGrafica = torch.linspace(0,10,250)
for i in puntosGrafica:
    ytemp=redDinamica(torch.tensor([i],device = device))
    ygrafica.append(ytemp.detach().item())
    #ygrafica.append(ytemp.detach().numpy()[0])
import numpy as np
puntosGrafica = np.linspace(0,10,250)
if nombre != 'senoParalelo0 epochs-4.66.tar':
    epoca = int(nombre[12:16])
else:
    epoca = 0
alpha = epoca/1e5
plt.plot(puntosGrafica,ygrafica,label = epoca,alpha = alpha)



# In[ ]:


for rondaDeOptimizacionGlobal in range(1):
    # Optimización para el parametro C1
    extremoInferiorIntervalo = 0.25e-6
    extremoSuperiorIntervalo = 5e-5
    minimoEnRonda : np.float128 = 3
    for rondaDeOptimizacion in range(3):
        anchoIntervalo = extremoSuperiorIntervalo - extremoInferiorIntervalo
        tiemposTomados = []
        valores_logaritmicos = puntos_chebyscheff(5,np.log10(extremoInferiorIntervalo),np.log10(extremoSuperiorIntervalo))
        valores = np.power(10,valores_logaritmicos)
        estadisticasDeSolucionesEnPDC= []
        # está solución solo es para quemar el primer valor
        solucionar(variable = 'Dummy', serializar = True)
        for valor in valores:
            # redefinición de la función de actualización con el nuevo valor
            c1 = valor
            # Generación de variable con el tiempo de referencia para la nueva solución.
            t0 = time.time()
            # Reestablecimiento de los parametros de la red Neuronal
            redDinamica = NeuralNetworkPrueba().to(device)
            # Solución con el nuevo valor.
            estadisticasDeSolucionesEnPDC.append( solucionar(variable = 'C1',serializar=True))
            # Toma del tiempo de la solución.
            tiemposTomados.append(time.time()-t0)
        print(f'valores : {valores}')
        print(f'tiempos tomados : {tiemposTomados}')
        # Interpolación polinomica con los tiempos de la nueva solución.
        interpolacion = np.polynomial.Polynomial.fit(valores,tiemposTomados,len(valores)-1)
        
        #Generación de un eje X para las gráficas y obetner el estimado del valor minimo.
        x_grafica = np.linspace(extremoInferiorIntervalo,extremoSuperiorIntervalo,200)

        # minimo en ronda es el punto en el cual se encontro el minimo DENTRO del polinomio.
        minimoEnRonda : np.float128 = x_grafica[np.argmin(interpolacion(x_grafica))]
        c1 = minimoEnRonda

        plt.semilogx(valores,tiemposTomados,linewidth = 0, marker = "*",label = "Puntos de prueba de la función")
        plt.semilogx(x_grafica,interpolacion(x_grafica),label = "Interpolación de la función")
        plt.xlim(extremoInferiorIntervalo,extremoSuperiorIntervalo)
        plt.semilogx(minimoEnRonda,interpolacion(minimoEnRonda),marker = "+", ms = 5,label = "Minimo encontrado")
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
        plt.show(block = False)
                
    
    # Optimización para el parametro C2
    extremoInferiorIntervalo = 0.5
    extremoSuperiorIntervalo = 2
    minimoEnRonda : np.float128 = 3
    solucionar(variable = 'Dummy', serializar = True)
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
        plt.show(block =False)

        # Serialización de la solución
        estadisticasC2.anadir([extremoInferiorIntervalo,extremoSuperiorIntervalo],
                                minimoEnRonda,
                                interpolacion,
                                estadisticasDeSolucionesEnPDC,
                                tiemposTomados,
                                valores)
    


        # Reducción del intervalo de busqueda a un 20% para la siguiente ronda de optimización.
        extremoInferiorIntervalo = minimoEnRonda * 1.1
        extremoSuperiorIntervalo = minimoEnRonda * 0.9
        
        


    # Optimización para el parametro sigma

    #Dado que sigma es un parametro real, se va a usar la transformación sigma = 10**x y se va a optimizar x
    extremoInferiorIntervalo = -2
    extremoSuperiorIntervalo = 0
    minimoEnRonda : np.float128 = 3
    solucionar(variable = 'Dummy', serializar = True)
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
        plt.show(block = False)

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
pickle.dump("Estadisticas c1",archivo)
pickle.dump(estadisticasC1,archivo)
pickle.dump("Estadisticas c2",archivo)
pickle.dump(estadisticasC2,archivo)
pickle.dump("Estadisticas sigma",archivo)
pickle.dump(estadisticassigma,archivo)
print("Termino de serializar")

archivo.close()


# Una de las conclusiones encontradas es que no se puede utilizar el ciclo de optimización propuesto porque el ruido no permite que la interpolación sea realista.
# 

# ## Comparación contra la solución no optimizada y la solución a partir de puntos aleatorios
# 
# Para la comparación contra la solución no optimizada se va a observar la convergencia en tiempo de cada función. Para esto se va a grabar como avanza la solucion vs las epocas y vs el tiempo.

# In[ ]:
if False:

    puntosAleatorios = torch.rand(40)
    def actualizarPuntosAleatorios():
        global puntosAleatorios
        puntosAleatorios = torch.rand(40)
    def solucionarPuntosAleatorios():
        pass


    # In[ ]:


    class ultimaComparacion():
        def __init__(self,nombresito):
            self.tiempos = []
            self.registrosPerdidasGlobales = []
            self.nombre = nombresito
            self.frecuenciaSampleo = 200 # Sampleo cada 200 datos
            pass
        def anadirDato(self,tiempo,perdida):
            self.tiempos.append(tiempo)
            self.registrosPerdidasGlobales.append(perdida)

    metodoTradicional = []
    metodoUsado = []
    for i in range(5):
        metodoTradicional.append(ultimaComparacion("Función de densidad de probabilidad uniforme"))
        metodoUsado.append(ultimaComparacion("Función de densidad de probabilidad móvil"))

    for i in range(5):    
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
    estaResolviendo = True
    solucionar(serializar=True,variable = "prueba",ronda=-2)



    # In[ ]:


    extremoInferiorIntervalo = 0.5e-5
    extremoSuperiorIntervalo = 1e-3
    minimoEnRonda : np.float128 = 3

    def puntos_chebyscheff(n,a,b):
        inicializacion = np.array(range(n+1))
        inicializacion = np.flip(inicializacion)
        return (a +b + (b-a)*np.cos(np.pi/(n+1)*(inicializacion + 0.5)))/2
    c2 = 1
    sigma = 0.25
    for rondaDeOptimizacion in range(3):
        anchoIntervalo = extremoSuperiorIntervalo - extremoInferiorIntervalo
        tiemposTomados = []
        valores = puntos_chebyscheff(8,extremoInferiorIntervalo,extremoSuperiorIntervalo)
        print(valores)
        estadisticasDeSolucionesEnPDC= []
        for valor in valores:
            # redefinición de la función de actualización con el nuevo valor
            c1 = valor
            # Generación de variable con el tiempo de referencia para la nueva solución.
            t0 = time.time()
            print(f"Empezo {valor}")
            for variableInutil in range(2):
                # Reestablecimiento de los parametros de la red Neuronal
                puntos = torch.linspace(0,0.5,15)
                redDinamica = NeuralNetworkPrueba().to(device)
                # Solución con el nuevo valor.
                estadisticasDeSolucionesEnPDC.append( solucionar(serializar=True))
                print(f"Termino una ronda en {time.time()-t0}")
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
        plt.show(block = False)
            

