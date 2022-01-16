# OPTIMIZACIÓN DE LA SOLUCIÓN DE ECUACIONES DIFERENCIALES A TRAVES DE REDES NEURONALES

## TO DO : METODOLOGÍA ; RESUMEN

El objetivo de este cuaderno es explicar la solución del movimiento armonico simple a traves del uso de PINNS con funciones de densidad de probabilidad dinámicas.  

## Orden

* Resumen.
* Introducción.
* Metodología
* Explicación de la razón por la cual se están implementando funciones de densidad de probabilidad dinámicas.
* Declaracion del problema.
* Importación de librerias.
* Declaración de la red neuronal.
* Declaración y explicación del conjunto de puntos sobre el cual se va a optimizar la función.
* Declaración de la función de perdida dentro del código.
* Condiciones de parada.
* Serialización de las soluciones y análisis de rata de aprendizaje.
* Análisis de los distintos parametros de los que depende el algoritmo.

## Introducción

La solución de ecuaciones diferenciales es un reto fundamental para la ingeniería y la física. <b> FILLER CUYO CONTENIDO EXACTO NO CONOZCO. </b> Con el surgimiento de las técnicas de inteligencia artificial, una pregunta natural es ¿Cómo es que estas pueden ayudar a solucionar ecuaciones diferenciales? Respondiendo a esta pregunta surgen las PINNs (Physically Inferred Neural Networks), cuyo acronimo significa redes neuronales con inferencias físicas.

Estas son redes neuronales entrenadas para interpolar o generar respuestas a ecuaciones diferenciales. Tradicionalmente, las interpolaciones de las redes neuronales solo tienen en cuenta los resultados de una simulación o respuesta anterior. Por otro lado, las simulaciones tradicionales solo tienen en cuenta las condiciones de frontera. Estas usualmente no usan resultados de simulaciones pasadas ni experimentos de laboratorio o reglas empíricas. Las PINN por el contrario, pueden solucionar/interpolar usando resultados de simualciones pasadas, experimentos de laboratorios, condiciones de frontera y/o reglas empíricas.

Esto se debe a que las PINNs no usan los sistemas de ecuaciones lineales usados por los métodos de runge-kutta. Esto permite que la inclusión de condiciones al solucionador que no sobredimensione las mátricez de los métodos tradicionales. Adicionalmente, estas redes usan algoritmos como la diferenciación automática, que le permite a las redes diferenciar con respecto a muchas variables con mayor precisión y velocidad que la diferenciación numérica.

Desafortunadamente, la solución de ecuaciones diferenciales a tráves de PINN resulta altamente costosa computacionalmente  para una gran variedad de implementaciones. Consecuentemente a traves del cuaderno se intentará reducir este costo. Esto se  hará a tráves de un proceso en el que primero se intenta entender como es que se converge a las respuestas y luego se intentará optimizar este proceso.

Esto se intentó realizar a partir de la solución de la ecuación de navier stokes para el caso conocido como "Backwards Flowing Step". Desafortunadamente, el proceso de solución era demasiado costoso computacionalmente como para poder iterar. Adicionalmente hubó problemas con el módulo de diferenciación automática y uno de los terminos de la ecuación. Debido a esto se optó por optimizar un caso más simple que fue el movimiento armónico simple.

## Explicación de la razón por la cual se están implementando funciones de densidad de probabilidad dinámicas.

Para realizar esto es conveniente hablar de como es que las PINNs resuelven ecuaciones diferenciales. Esto se hara a tráves de una explicación de porque se usan redes neuronales; como es que estas resuelven ecuaciones diferenciales; el problema generado por el paso anterior y finalmente como es que se espera que las funciones de probabilidad de densidad dinámicas lo resuelvan.

Empezando con las redes neuronales, estás no son más que funciones con una gran cantidad de parametros que tienen la capacidad de acercarse mucho a las formas de otras funciones. Esto significa que para una cantidad adecuada de parametros/neuronas se debería poder obtener virtualmente cualquier función deseada. Lo anterior significa que a la hora de resolver una ecuación diferencial, con una red neuronal que tenga una cantidad adecuada de parámetros, se podrá obtener una función que satisfazca la ecuación diferencial. Adicionalmente, en este caso se están usando funciones diferenciables. Es decir que lo que se están usando redes neuronales porque se espera que estas tengan la capacidad de tomar la forma de la respuesta a la ecuación diferencial.

La pregunta que surge naturalmente es ¿Cómo lograr que la red tome la forma de la función deseada? Esto se realiza a tráves del uso del aprendizaje por descenso de gradiente estocástico. Este es el nombre de una familia de algoritmos que resuelve el problema a traves de variar una gran cantidad de parametros. Para realizar esto el algoritmo utiliza una función que le indique que tan bien funciona cada conjunto de parámetros. De está forma el algoritmo llega a los parámetros que maximicen la función anterior. 

La forma tradicional para realizar lo anteriormente descrito consiste en:
1. Seleccionar un conjunto de puntos aleatorios que pertenezcan al dominio donde se espera resolver la ecuación diferencial.
2. Evaluar que tan bien estos puntos cumplen la ecuación diferencial.
3. Evaluar que tan bien se cumplen las condiciones de frontera.
4. Actualizar los parametros para mejorar los puntajes de los puntos 2 y 3 a traves de descenso de gradiente.
5. Repetir el paso 1.

Para explicar el problema asociado a este procedimiento, hay que tomar en cuenta que una gran mayoria de las ecuaciones diferenciales usadas son homogeneas y tienen una cantidad infinita de respuestas posibles (la cantidad de respuestas en este caso es del tamaño del conjunto $\mathbb{R}^2$ porque la ecuación diferencial es de grado 2). desafortunadamente esto lleva a que existan muchas respuestas que resuelven la ecuación diferencial pero no las condiciones de frontera. Ejemplos de esto son  $y(x) = 0 \text{ ; } y(x) = cos (x) \text{ ; } y(x) = 0.1 sin(x) $. Lo anterior significa que se necesita resolver la ecuación diferencial que resuelva las condiciones de frontera. 

El método tradicional primero se acopla a las condiciones de frontera y luego propaga la ecuación diferencial al resto del dominio. Esto significa que no todas las partes del dominio se van a resolver de manera simultanea. Las regiones más cercanas a las condiciones de frontera se van a resolver antes que el resto. Después de que estas se resuelvan, debido a que se están usando funciones diferenciables, eventualmente se va a propagar la solución correcta al resto del dominio.

Lo anterior significa que si se están colocando puntos sobre las zonas en las cuales la solución no se ha propagado, se está perdiendo el tiempo y esfuerzo computacional. Consecuentemente se espera que al utilizar una función de densidad de probabilidad que no coloque los puntos de manera uniformemente repartida, sino solo cerca a las condiciones de frontera, se logre aumentar la velocidad de convergencia.