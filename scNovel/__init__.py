import torch
import numpy as np
import torch.nn as nn
import time
import torch.utils.data as Data
from scNovel.scNovel_classifier import classifier
from scNovel.weightsampling import Weighted_Sampling
from scNovel.score_functions import get_ood_scores



## Parámetros de la función:
# test: Datos a testear
# reference: datos para entrenar el modelo
# label: Conjunto de etiquetas correspondiente a los datos de entrenamiento
# score_function: Función de puntuación para obtener puntuaciones ODD.
# processing_unit: Unidad de procesamiento; gpu = CUDA, cpu = CPU default

def scNovel(test = None, reference = None, label = None, processing_unit = 'cuda',score_function="sim",iteration_number=3000):

    label.columns = ['Label'] #Renombrar columnas de "label" a "Label"
    label.Label.value_counts() 
    status_dict = label['Label'].unique().tolist() ## Lista de etiquetas única
    int_label = label['Label'].apply(lambda x: status_dict.index(x))
    label.insert(1,'transformed',int_label) ## Se transforman etiquetas en valores enteros y se añaden como nueva columna en "label"

    ## Transformación de conjuntos de datos de test y entrenamiento en objetos pyTorch
    X_train = reference.values
    X_test = test.values
    y_train = label['transformed'].values

    dtype = torch.float
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y_train = torch.from_numpy(y_train)

    ## Técnica de muestreo ponderado para el conjunto de entrenamiento
    sampler = Weighted_Sampling(y_train)

    #construct pytorch object
    train_data = Data.TensorDataset(X_train, y_train)
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=32,
                                   sampler = sampler,
                                   num_workers=1)


    test_data = Data.TensorDataset(X_test, torch.zeros(len(X_test)))
    test_loader = Data.DataLoader(dataset=test_data,
                                   batch_size=32,
                                   num_workers=1)


    input_size = X_train.shape[1]
    num_class = len(status_dict)
    iteration=iteration_number
    count_iteration=0

    ## Se inicia el modelo de clasificación con el tamaño de entrada y el nº de clases
    model = classifier(input_size, num_class)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # total_step = len(train_loader)

    ## Entrenamiento del modelo
    print("--------Start annotating----------")
    start = time.perf_counter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if processing_unit == 'cpu':
        device = torch.device('cpu')
    elif processing_unit == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("No GPUs are available on your server.")

    print("Computational unit be used is:", device)

    model.to(device)

    model.train()
    while count_iteration<iteration:
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            train_loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            count_iteration+=1
            # print(count_iteration)
            if count_iteration>=iteration:
                break

        print("Finish {}/{}".format(float(count_iteration), iteration))
    # print("Finish {}/{}".format(float(count_iteration), iteration))
    print("\n--------Annotation Finished----------")

    ## Evaluación del modelo, puntuaciones ODD con get_odd_scores.
    model.eval()

    train_score=get_ood_scores(train_loader,model,score_function,device=device)
    test_score = get_ood_scores(test_loader, model, score_function, device=device)

    ## Las puntuaciones se normalizan utilizando la puntuación máxima del conj de entrenamiento.
    max_socre=np.max(train_score)

    test_score=(test_score-1.0/num_class)/(max_socre-1.0/num_class)



    return test_score
