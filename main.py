import pandas as pd
from gurobipy import Model, GRB, quicksum
import json

def lectura(nombre, **kw):
    #https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    return pd.read_csv(nombre, sep=";")

luminarias_ex = lectura("luminarias.csv")
#Diccionario de luminarias
luminarias = luminarias_ex.to_dict()
#Parámetros del modelo
tipos_luminarias = luminarias_ex["Indice"]
CM_l = luminarias["CM_l"]
costo_compra = luminarias["Costo de compra e instalacion"]
b_l = luminarias["b_l"]
eficiencia = luminarias["Eficiencia"]
p_l = luminarias["p_l"]

compatibilidad_ex = lectura("compatibilidad.csv")
compatibilidad = compatibilidad_ex.to_dict()

parametros_ex = lectura("parametros_globales.csv")
parametros = parametros_ex.to_dict(orient="records")
#Parámetros extra del modelo
presupuesto = parametros[0]["Valor"]
V = parametros[4]["Valor"]
F = parametros[3]["Valor"]
G = parametros[2]["Valor"]
R = parametros[1]["Valor"]

sectores_ex = lectura("sectores.csv")
sectores = sectores_ex.to_dict()
#Parámetros asociados a sectores
num_sectores = sectores["Sectores"]
alpha = sectores["alpha"]
K_s = sectores["K_s"]
Ns_max = sectores["Ns_max"]
P_min_s = sectores["P_min_s"]
P_max_s = sectores["P_max_s"]

#conjunto de sectores
S = range(1,len(num_sectores)+1)
print(S)
#Conjunto de tipos de luminarias
L = range(1,len(tipos_luminarias)+1)
#Conjunto de periodos
T = range(1,15)