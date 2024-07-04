import torch
import torch.utils.data as Data

## Esta función crea un muestreador ponderado, útil para desequilibrio de clases
## de un conjunto de datos. En este caso la probabilidad de seleccionar una muestra
## es inversamente proporcional a la frecuencia de su clase.

def Weighted_Sampling(train_label = None):

    '''

    :param train_label: Input train label (in tensor) to calculate sampling weight
    :return: Pytorch Weighted Sampler
    '''
    ## Obtiene las etiquetas únicas en "train level" de forma ordenada
    ## y calcula el número de muestras para cada clase
    class_sample_count = torch.tensor(
        [(train_label == t).sum() for t in torch.unique(train_label, sorted=True)])

    ## Convierte los conteos a tipo float
    ## y calcula el peso inverso de cada clase, las clases con menos
    ## muestras tendrán un peso mayor (Células raras).
    weight = 1. / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in train_label])

    ## Creación de variable que utiliza los pesos calculados para muestrear
    ## de manera proporcional los pesos de las clases.
    sampler = Data.WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler

