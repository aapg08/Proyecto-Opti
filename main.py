import pandas as pd
from gurobipy import Model, GRB, quicksum
import json

def lectura(nombre, **kw):
    #https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    return pd.read_csv(nombre, sep=";")

def cargar_datos():

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

    #Compatibilidad por sector
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

    #############################################
    #############################################

    #CONJUNTOS
    #conjunto de sectores
    S = range(1,len(num_sectores))
    print(S)
    #Conjunto de tipos de luminarias
    L = range(1,len(tipos_luminarias))
    #Conjunto de periodos
    T = range(1,6)

    datos = {
        "S": S, "L": L, "T": T, "alpha": alpha, "Tipos luminarias": tipos_luminarias,
        "presupuesto": presupuesto, "CM_l":CM_l, "K_s": K_s, "Ns_max": Ns_max, "P_min_s": P_min_s,
        "P_max_s":P_max_s, "Sectores":num_sectores, "V": V, "R":R, "G":G, "F":F, "p_l": p_l, "b_l":b_l,
        "Eficiencia": eficiencia, "Costo compra e instalación": costo_compra
    }
    return datos

def construir_model(data):
    S = data["S"]
    L = data["L"]
    T = data["T"]
    model = Model("Proyecto E2")
    model.Params.OutputFlag=0

    #Variables de decisión

    x_s_l_t = model.addVars(S, L, T, name = "x_s_l_t", lb = 0)
    y_s_l_t = model.addVars(S, L, T, vtype= GRB.BINARY, name = "y_s_l_t")
    u_s_l_t = model.addVars(S, L, T,vtype= GRB.BINARY, name = "u_s_l_t")
    zi_s_t = model.addVars(S, T,vtype= GRB.BINARY, name = "zi_s_t")
    ze_t = model.addVars(T,vtype= GRB.BINARY, name = "ze_t")
    zf_t = model.addVars(T,vtype= GRB.BINARY, name = "zf_t")
    wi_t = model.addVars(T, name = "wi_t", lb = 0)
    we_t = model.addVars(T, name = "we_t", lb = 0)
    wf_t = model.addVars(T, name = "wf_t", lb = 0)
    j_t = model.addVars(T, name = "j_t", lb = 0)

    #Función Objetivo
    model.setObjective(quicksum(x_s_l_t[s, l, t]*data["alpha"][s]*data["b_l"][l] for s in S for l in L for t in T))

    #R2 Activación de la mantención solo si se instala la luminaria
    model.addConstrs((u_s_l_t[s, l, t] <= x_s_l_t[s,l,t] for s in S for l in L for t in T), name="xandu") 

    return model 

def resolver_modelo (model):
    model.optimize()
    return model

def imprimir(model):
    pass

def main():
    data = cargar_datos()
    modelo = construir_model(data)

if __name__ == "__main__":
    main()