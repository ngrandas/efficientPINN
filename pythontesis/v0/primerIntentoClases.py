import numpy as np
class neurona:
    def __init__(self,n_conexiones):
        self.w=np.random.rand(n_conexiones)
        self.valorActual=0
        self.activacion=np.tanh
        self.argumento=0
    def evaluarNeurona(self):
        self.valorActual=self.activacion(self.argumento)
        return self.valorActual
    def evaluarVector(self,x):
        self.argumento=np.dot(self.w,x)
        return self.evaluarNeurona()    


class banda:
    def __init__(self,n_neuronas,n_conexiones):
        self.n_neuronas=n_neuronas
        self.n_conexiones=n_conexiones
        self.neuronas=[]
        for i in range(n_neuronas):
            self.neuronas.append(neurona(self.n_conexiones))

class redNeuronal:
    def __init__(self,array_bandas,n_entradas):
        #Aca array bandas es el array que especifica cuantas neuronas van a haber por banda. Labanda final es la salida.
        self.n_bandas=len(array_bandas)
        self.bandas=[]
        self.n_entradas=n_entradas
        j=n_entradas
        for i in array_bandas:
            self.bandas.append(banda(i,j))
            j=i
        

a=redNeuronal([3,3,1],1)





    