import matplotlib.pyplot as plt 
import random as rd
import numpy as np
re=1
uprom=2
xinf=0
xsup=1
ubicacionSalto=0.2
altoSalto=0.1
# %%

# %%
def ysup(x):
	return 1
def yinf(x):
	#Esta funcion define el limite inferior de mi fluido en Y como una funcion de x
	if (x>ubicacionSalto):
		return 0
	else:
		return altoSalto

def nube(n_interno,n_frontera):
	#Esta funcion genera los puntos aleatorios sobre los cuales se va a probar la 
	#funcion de perdida
	#n_interno corresponde a los puntos internos del fluido
	#n_frontera corresponde a los puntos sobre los cuales se probara la condicion de frontera
    puntosInternos=[]
    puntosInlet=[]
    puntosPared=[]
    puntosOutlet=[]
    for i in range(n_interno):#Estos son los puntos Internos
        xtemp=xinf+(xsup-xinf)*rd.random()
        ytemp=yinf(xtemp)+(ysup(xtemp)-yinf(xtemp))*rd.random()
        puntosInternos.append([xtemp,ytemp])

	#Estos puntos estan en las fronteras
    for i in range(n_frontera):
        seleccion = rd.randint(1,5)
        if(seleccion == 1): #Esta va a ser la frontera de la pared arriba
            xtemp=xinf+(xsup-xinf)*rd.random()
            ytemp=ysup(xtemp)
            puntosPared.append([xtemp,ytemp])
        elif (seleccion == 2): #Esta va a ser la pared de abajo sin incluir el step
            xtemp=xinf+(xsup-xinf)*rd.random()
            ytemp=yinf(xtemp)
            puntosPared.append([xtemp,ytemp])
        elif (seleccion==3): #Esta va a ser la pared donde hay el backwards facing step
            xtemp=ubicacionSalto
            ytemp=yinf(xsup)+altoSalto*rd.random()
            puntosPared.append([xtemp,ytemp])
        elif (seleccion==4): #Este va a ser el inlet
            xtemp=xinf
            ytemp=yinf(xtemp)+(ysup(xtemp)-yinf(xtemp))*rd.random()
            puntosInlet.append([xtemp,ytemp])
        else:
            xtemp=xsup
            ytemp=yinf(xtemp)+(ysup(xtemp)-yinf(xtemp))*rd.random()
            puntosOutlet.append([xtemp,ytemp])
    return ([puntosInternos,puntosPared,puntosInlet,puntosOutlet])

def plottearNube(n_interno,n_frontera):
	labels=["Fluido","Pared","Inlet","Outlet"]
	y=0
	for j in nube(n_interno,n_frontera):
		xtemp=[]
		ytemp=[]
		for i in j:
			xtemp.append(i[0])
			ytemp.append(i[1])
		plt.plot(xtemp,ytemp,linewidth=0,marker="o",label=labels[y])
		y+=1
	plt.legend()
	plt.show()

def uini(y):
	k=-8*uprom/(ysup(xinf)-yinf(xinf))**2
	return k*(y-yinf(xinf))*(y-ysup(xinf))

def red(x): #Esta funcion va a recibir un argumento de la forma [x,y] y va a dar la respuesta en la forma [u,v,p]
	return [uini(x[1]),0,0]
	
def miniPerdidaInlet(puntos_inlet):
	sumaInterna = 0
	for i in puntos_inlet:
		sumaInterna+=np.abs(red(i)[1])
	return sumaInterna
	
def miniPerdidaPared(puntos_pared):
	sumaInterna = 0
	for i in puntos_pared:
		sumaInterna+=np.abs(red(i)[0])
		sumaInterna+=np.abs(red(i)[1])
	return sumaInterna
		
def miniPerdidaOutlet(puntos_outlet):
	sumaInterna = 0
	for i in puntos_outlet:
		sumaInterna+=np.abs(red(i)[2])
	return sumaInterna
	
def miniperdidaInterna(puntos_internos):
	return 0

alpha=1
beta=1
gamma=1
registro_perdida_interna=[]
registro_perdida_pared=[]
registro_perdida_inlet=[]
registro_perdida_outet=[]


def perdida(n_internos,n_frontera):
	puntos = nube(n_internos,n_frontera)
	suma = 0 #la variable suma es la que va a cargar la suma de las perdidas
	
	registro_perdida_interna.append(miniperdidaInterna(puntos[0]))
	suma+=registro_perdida_interna[-1] #ACA DEBE IR LA ECUACION DE NAVIER STOKES
	registro_perdida_pared.append(miniPerdidaPared(puntos[1]))
	suma+=alpha*registro_perdida_pared[-1] #ACA DEBE IR LA ECUACION DE LA PARED
	registro_perdida_inlet.append(miniPerdidaInlet(puntos[2]))
	suma+=beta*registro_perdida_inlet[-1] #ACA DEBE IR LA CONDICION PARA QUE EL FLUJO ENTRE DESARROLADO
	registro_perdida_outet.append(miniPerdidaOutlet(puntos[3]))
	suma+=gamma*registro_perdida_outet[-1] #ACA DEBE IR LA ECUACION PARA EL OUTLET, ESTA ES DE PRESION UNICAMENTE
	return suma

def sacaVelocidades(x,y):
	return red([x,y])[0],red([x,y])[1]
	
	
def graficarVelocidades():
	#Esto aun no funciona porque no termino de entender quiver
    flechasEnX=18
    flechasEnY=14
	
    flechasEnPrimeraSeccion=int(flechasEnX*(ubicacionSalto-xinf)/(xsup-xinf))	
    flechasEnSegundaSeccion=flechasEnX-flechasEnPrimeraSeccion
   	
    xtemp=np.linspace(xinf,ubicacionSalto,flechasEnPrimeraSeccion)
    ytemp=np.linspace(yinf(xinf),ysup(xinf),flechasEnY)
    Xtemp,Ytemp=np.meshgrid(xtemp,ytemp)
    utemp,vtemp=sacaVelocidades(Xtemp,Ytemp)
    plt.figure(figsize=(10,10))
    plt.ylim(yinf(xsup),ysup(xsup))
    plt.xlim(xinf,xsup)
    plt.quiver(xtemp,ytemp,utemp,vtemp,units="xy",scale=75,width=0.005)
   	
    xtemp=np.linspace(ubicacionSalto,xsup,flechasEnSegundaSeccion)
    ytemp=np.linspace(yinf(xsup),ysup(xsup),flechasEnY)	
    Xtemp,Ytemp=np.meshgrid(xtemp,ytemp)
    utemp,vtemp=sacaVelocidades(Xtemp,Ytemp)
    plt.quiver(xtemp,ytemp,utemp,vtemp,units="xy",scale=75,width=0.005)

    plt.show()

def trazadorDeStreamLine(x_ini,y_ini):	
	ntemp=50
	deltaTemp=(xsup-x_ini)/ntemp
	xtemp=[x_ini]
	ytemp=[y_ini]
	for i in range(ntemp):
		utemp,vtemp=sacaVelocidades(xtemp[-1],ytemp[-1])
		xtemp.append(xtemp[-1]+utemp*deltaTemp)
		ytemp.append(ytemp[-1]+vtemp*deltaTemp)
	plt.plot(xtemp,ytemp)
	plt.show()
	
	
def probarPerdida():
	print(perdida(100,1000))
	print("interna",registro_perdida_interna)
	print("pared",registro_perdida_pared)
	print("inlet",registro_perdida_inlet)
	print("outlet",registro_perdida_outet)

probarPerdida()