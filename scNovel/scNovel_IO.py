import numpy as np
import pandas as pd
import scanpy as sc
from warnings import simplefilter

## La función principal del código es el procesamiento y normalización
## de datos de expresión génica de células individuales.


## Esta función procesa dos objetos Anndata procedentes de Scanpy, normalizando
## y escalando los datos de expresión génica. Prepara los datos para la anotación
## celular.
def Scanpy_Obj_IO(test_obj=None, ref_obj=None, label_obj=None, scale = False):
    '''

    :param test_obj: Parámetro de entrada para la anotación del tipo celular.
    :param ref_obj: Anndata de referencia cell atlas, archivo csv
    :param label_obj: (IF APPLICABLE)  Ruta a un archivo CSV con etiquetas, si el conjunto de datos de referencia está dividido en una matriz de expresión y un       archivo de etiquetas.
    :return: test_matrix (pd.dataframe) for annotation,
    ref_matrix (pd.dataframe) contain reference expression matrix,
    label (pd.dataframe) contain the label list.
    :scale: Indicar si se debe escalar el conjunto de datos de referencia
    '''

    ref_adata = ref_obj
    ## Si label_obj es None, se separa la última columna como etiquetas.
    if label_obj is None:
        label = ref_adata[:, ref_adata.n_vars - 1].to_df()
        ref_adata = ref_adata[:, 0:ref_adata.n_vars - 1]
    ## Si label_obj no es None, se carga directamente.
    else:
        label = label_obj
        
    ## Se normaliza y se hace logaritmo los datos del dataset
    sc.pp.normalize_total(ref_adata, target_sum=1e4)
    sc.pp.log1p(ref_adata)
    simplefilter(action='ignore', category=FutureWarning)
    ## Se identifican los genes comunes entre ref_adata y test_obj.
    gene = ref_adata.var_names & test_obj.var_names

    ## Si test_obj ya está log-transformado, se seleccionan los genes comunes.
    if 'log1p' in test_obj.uns:
        # test whether the dataset is in log format
        ref = ref_adata[:,gene]
        test_matrix = test_obj.to_df()[gene]
    ## Si no está log-transformado, se aplica la transformación logarítmica y se seleccionan los genes comunes.
    else:
        sc.pp.log1p(test_obj)
        ref = ref_adata[:,gene]
        test_matrix = test_obj.to_df()[gene]

    ## Escalado (Opcional)
    if scale:
        sc.pp.scale(ref, max_value=10)
        ref_matrix = ref.to_df()
    else:
        ref_matrix = ref.to_df()

## test_matrix: Matriz de datos del objeto de prueba con los genes comunes.
## ref_matrix: Matriz de datos del objeto de referencia con los genes comunes y escalados si se seleccionó esa opción.
## label: DataFrame con las etiquetas.
    return test_matrix, ref_matrix, label


## Esta función carga datos de expresión génica desde archivos CSV, los normaliza ## y log-transforma
def CSV_IO(csv_path:str = None, label_path:str = None, with_label:bool = True):

    '''

    :param csv_path: Input csv path to load expression matrix
    :param label_path: Input csv path to load label
    :param label: whether we have label (whether this is the reference)
    :return:
    '''
    #normalize the reference dataset
    ## Se normalizan los datos para que el total de cada célula sume 10,000.
    ## Se aplica una transformación logarítmica natural (log1p).
    adata = sc.read_csv(csv_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ## Se carga el archivo de etiquetas, se asigna columna "label" y se
    ## transforma cada etiqueta en un número usando un dicc de estados.
    if with_label:
        #load label
        label = pd.read_csv(label_path)

        #turn to label matrix
        label.columns = ['Label']
        status_dict = label['Label'].unique().tolist()
        label['transfromed'] = label['Label'].apply(lambda x: status_dict.index(x))
        label_matrix = label['transfromed'].values

        #return result
        count_matrix = adata.to_df()
        return count_matrix, label_matrix, status_dict

    else:
        count_matrix = adata.to_df()

        return count_matrix


