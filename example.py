from fashion_mnist_master.utils.mnist_reader import load_mnist
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import itertools



## Importando la data
X_train, y_train = load_mnist('fashion_mnist_master/data/fashion', kind='train')
X_test, y_test = load_mnist('fashion_mnist_master/data/fashion', kind='t10k')


#### Transformación de X_train
chiquitolina = (1/255)
X_train = X_train * (chiquitolina)



## X_train tiene la siguiente forma
### (observaciones, atributos)
### Cada elemento del array es una observación. 
### Según mi documento oficial juandieguístico tiene la forma n*m

L = 6 # Capas desde la X hasta la Y

### One-hot enconding para Y
y_train_one_hot = np.zeros((y_train.size, y_train.max()+1))
y_train_one_hot[np.arange(y_train.size),y_train] = 1

## FUNCIONES IMPORTANTES
def funcion_sigmoide(x, tetas):
    """Realiza la función sigmoide
        para cualquier parámetro teta
    """
    return np.power((1 + np.exp(-1 * np.dot(x, tetas.T)) ), -1) 


def funcion_costo(tetas, figuras, neuronas, x, y):
    """Realiza la función de costo
        para cualquier parámetro teta
    """
    tetas = deschatar_thetas(tetas, figuras)
    forward_propagation(tetas, neuronas, x) # Multiplicación de tetas
    return np.sum(np.multiply(y, np.log( neuronas[L-1] )) + np.multiply((1-y), np.log(1 - neuronas[L-1]))  ) / -len(y)

def precision(y_pred, y):
    enconding = 0
    for i in range(len(y_pred)):
        valorMax = max(y_pred[i])
        valor = np.where(y_pred[i] == valorMax)
        if(valor == y[i]):
            enconding += 1 
        
    return enconding / len(y)

## Creando variables para el f. forward
neuronas = [None] * L
#### Como estándar todas mis neuronas serán de tamaño 5*5 + bias(unos)
#### Cada elemento en el array contiene una capa de la red
#### Por ejemplo: [X_train, neuronas1 = [5 neuronas], neuronas2= [5 neuronas] hasta el resultado que son una capa de 10 tipos]
neuronas[0] = X_train ## agregando la primera capa



thetas = []
#### Cada elemento contiene un arreglo de thetas. Para el primero son c*m según el documento juandieguístico.
#### Después serán 5*5 y el último será 10*6
thetas1 = np.random.rand(5, 785).astype(np.float32) # X a capa1
thetas2 = np.random.rand(5, 6).astype(np.float32) # capa1 a 2
thetas3 = np.random.rand(5, 6).astype(np.float32) # capa2 a 3
thetas4 = np.random.rand(5, 6).astype(np.float32) # capa3 a 4
thetas5 = np.random.rand(10, 6).astype(np.float32) # capa4 a resultado
## agregando thetas
thetas.append(thetas1)
thetas.append(thetas2)
thetas.append(thetas3)
thetas.append(thetas4)
thetas.append(thetas5)

figuras = [(5, 785), (5, 6), (5, 6), (5, 6), (10, 6)]

## agregar columna de unos
unos = np.ones((len(X_train), 1))

def achatar_thetas(tetas):
    achatadas = np.array([])
    for i in tetas:
        achatadas = np.append(achatadas, i.flatten())
    achatadas.flatten()
    return achatadas

def deschatar_thetas(tetas, figura):
    deschatadas = []
    inicio = 0
    fin = 0
    for i in figura:
        fin = inicio + i[1]*i[0]
        arreglo = np.array(tetas[inicio:fin])
        deschatadas.append(arreglo.reshape(i))
        #print(len(tetas[inicio:fin]))
        #print(inicio, fin)
        inicio = fin
    return deschatadas


def forward_propagation(tetas, neuronas, x):
    unos = np.ones((len(x), 1))
    for i in range(len(tetas)):
        ###### La siguiente neurona
        # print(i)
        # print(neuronas[i].shape)
        # print((np.append(neuronas[i], unos, axis=1)).shape)
        # print(neuronas[i])
        # capa_mas_bias = np.append(neuronas[i], unos, axis=1) # neurona actual más capa de unos (bias)
        neuronas[i + 1] = funcion_sigmoide(
            np.append(neuronas[i], unos, axis=1), # neurona actual más capa de unos (bias)
            tetas[i]) # Thetas actuales (ya incluyen la capa bias)


forward_propagation(thetas, neuronas, X_train)

def back_propagation(tetas, figuras, neuronas, x, y):
    tetas = deschatar_thetas(tetas, figuras)
    # Back propagation
    # Paso 1 y 2.1
    Deltas = [i * 0.0 for i in tetas]
    deltas = [i * 0.0 for i in neuronas]
    # Forward Propagation (Paso 2.2)
    forward_propagation(tetas, neuronas, x)
    deltas[L - 1] = neuronas[L - 1] - y # Paso 2.3 del documento de Samuel Chávez
    for l in reversed(range(1, L - 1)):
        #print(l)
        #print(tetas[l].T.shape)
        #print(deltas[l + 1].shape)
        #print(neuronas[l].shape)
        #print(deltas[l].shape)
        #print(Deltas[l].shape)
        # Paso 2.4
        deltas[l] = np.multiply(
            np.dot(tetas[l].T[:-1], deltas[l + 1].T).T,
            np.multiply(
                neuronas[l],
                (1 - neuronas[l])
                ))
        # Paso 2.5
        Deltas[l] = Deltas[l] + np.dot(deltas[l + 1].T, np.append(neuronas[l], unos, axis=1))
        # Paso 3
        Deltas[l] = Deltas[l] * (1/len(x))
    tetas = achatar_thetas(tetas)
    return achatar_thetas(Deltas)

thetas = achatar_thetas(thetas)
Deltas = back_propagation(thetas, figuras, neuronas, X_train, y_train_one_hot)    

thetas = deschatar_thetas(thetas, figuras)

print("\n#######################################")
print(X_train.shape)
print(y_train_one_hot.shape)
print(neuronas[len(thetas)].shape)
# print(y_train_one_hot)
# thetas = np.array(thetas, dtype=np.float64)
# for i in range(len(thetas)):
#     print("T ", thetas[i].shape)
#     print("D ", Deltas[i].shape)

thetas = achatar_thetas(thetas)
costo = funcion_costo(thetas, figuras, neuronas, X_train, y_train_one_hot)
print(costo)
print(precision(neuronas[L - 1], y_train))
print(type(costo))

# achatadas = achatar_thetas(thetas)
# print(achatadas)
# print(len(achatadas))

# deschatadas = deschatar_thetas(achatadas, figuras)

# thetas = deschatar_thetas(thetas, figuras)
# for i in range(len(thetas)):
#     print(deschatadas[i] == thetas[i])


res = op.minimize(
    fun=funcion_costo,
    x0=thetas,
    args=(figuras, neuronas, X_train, y_train_one_hot),
    method='L-BFGS-B',
    jac=back_propagation,
    options={'disp': True, 'maxiter': 20})


thetas = res.x
costo = funcion_costo(thetas, figuras, neuronas, X_train, y_train_one_hot)
print(costo)
print(precision(neuronas[L - 1], y_train))







# ## Gráficando un elemento 
# X = np.array(X_train[1])
# X = X.reshape((28, 28))
# ## Poniéndolo en blanco y negro
# fig, ax = plt.subplots()
# ax.imshow(X, interpolation='nearest', cmap='gray')
# plt.show()
