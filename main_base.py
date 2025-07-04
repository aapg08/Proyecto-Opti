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
    eficiencia = luminarias["Eficiencia"]
    p_l = luminarias["p_l"]
    iluminancia_max = luminarias["Iluminancia maxima"]
    luminarias_solares_ex = luminarias_ex[luminarias_ex["Fuente de energia"] == "solar"]
    l_solar_aux = luminarias_solares_ex["Indice"].tolist()
    l_solar = range(1, len(l_solar_aux) + 1)

    #Compatibilidad por sector
    compatibilidad_ex = lectura("compatibilidad.csv")
    compatibilidad = compatibilidad_ex.to_dict()

    parametros_ex = lectura("parametros_globales.csv")
    parametros = parametros_ex.to_dict(orient="records")
    #Parámetros extra del modelo
    presupuesto = parametros[0]["Valor"]
    Ae = parametros[5]["Valor"]
    Af = parametros[6]["Valor"]
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
    # P_min_s = sectores["P_min_s"]
    # P_max_s = sectores["P_max_s"]
    M = 1e6

    periodos_ex = lectura("periodos.csv")
    P_min_s_t = {}
    P_max_s_t = {}
    for _, row in periodos_ex.iterrows():
        s = int(row["Sectores"])
        t = int(row["Periodos"])
        P_min_s_t[(s, t)] = row["P_min_s_t"]
        P_max_s_t[(s, t)] = row["P_max_s_t"]
    # periodos = periodos_ex.to_dict()
    # P_min_s = periodos["P_min_s_t"]
    # P_max_s = periodos["P_max_s_t"]
    

    #############################################
    #############################################

    #CONJUNTOS
    #conjunto de sectores
    S = range(1,len(num_sectores))
    print(S)
    #Conjunto de tipos de luminarias
    L = range(1,len(tipos_luminarias))
    #Conjunto de periodos
    # T = range(1,10)
    T = sorted(list(set(row["Periodos"] for _, row in periodos_ex.iterrows())))

    datos = {
        "S": S, "L": L, "T": T, "alpha": alpha, "Tipos luminarias": tipos_luminarias,
        "B": presupuesto, "CM_l":CM_l, "K_s": K_s, "Ns_max": Ns_max, "P_min_s": P_min_s_t,
        "P_max_s":P_max_s_t, "Sectores":num_sectores, "V": V, "R":R, "G":G, "F":F, "p_l": p_l,
        "Eficiencia": eficiencia, "C_l": costo_compra, "M":M, "I_s_max": iluminancia_max, 
        "L*": l_solar, "Ae": Ae, "Af": Af
    }
    return datos

def construir_model(data):
    S = data["S"]
    L = data["L"]
    T = data["T"]
    L_solar = data["L*"]
    model = Model("Proyecto E2")
    model.Params.OutputFlag=0

    #Variables de decisión

    x_s_l_t = model.addVars(S, L, T, name = "x_s_l_t",vtype= GRB.INTEGER, lb = 0)
    y_s_l_t = model.addVars(S, L, T, vtype= GRB.BINARY, name = "y_s_l_t")
    u_s_l_t = model.addVars(S, L, T,vtype= GRB.BINARY, name = "u_s_l_t")
    zi_s_t = model.addVars(S, T,vtype= GRB.BINARY, name = "zi_s_t")
    ze_s_t = model.addVars(S,T,vtype= GRB.BINARY, name = "ze_t")
    zf_s_t = model.addVars(S,T,vtype= GRB.BINARY, name = "zf_t")
    wi_s_t = model.addVars(S,T, name = "wi_t", vtype= GRB.CONTINUOUS, lb = 0)
    we_s_t = model.addVars(S,T, name = "we_t", vtype= GRB.CONTINUOUS,  lb = 0)
    wf_s_t = model.addVars(S,T, name = "wf_t",vtype= GRB.CONTINUOUS,  lb = 0)
    j_t = model.addVars(T, name = "j_t", vtype= GRB.CONTINUOUS, lb = 0)
    L_l_s = model.addVars(L,S, vtype=GRB.BINARY, name = "L_l_s")

    #Función Objetivo
    model.setObjective(quicksum(x_s_l_t[s, l, t]*data["alpha"][s]*data["p_l"][l] for s in S for l in L for t in T), GRB.MAXIMIZE)

    #R1 No superar el presupuesto de municipio: Flujo de caja
    model.addConstr(
        j_t[1] >= data["B"]
        - quicksum(data["C_l"][l] * x_s_l_t[s, l, 1] for s in S for l in L)
        - quicksum(data["CM_l"][l] * u_s_l_t[s, l, 1] for s in S for l in L)
        - quicksum(wi_s_t[s,1] for s in S) - quicksum(we_s_t[s,1] for s in S) + quicksum(wf_s_t[s,1] for s in S),
        name="Flujo_caja_t1"
    )
    for t in T:
        if t == 1:
            continue  # Esta restricción aplica solo para t >= 2

        model.addConstr(
            j_t[t] == j_t[t-1]
            - quicksum(data["C_l"][l] * x_s_l_t[s, l, t] for s in S for l in L)
            - quicksum(data["CM_l"][l] * u_s_l_t[s, l, t] for s in S for l in L)
            - quicksum(wi_s_t[s,t] for s in S for t in T) - quicksum(we_s_t[s,t] for s in S for t in T) + quicksum(wf_s_t[s,t] for s in S for t in T),
            name="presupuesto_t"
        )
    #R2 Activación de la mantención solo si se instala la luminaria
    #model.addConstrs((x_s_l_t[s,l,t]<= data["M"] for s in S for l in L for t in T), name="xandu") 
    #model.addConstrs((x_s_l_t[s,l,t]>= 1 for s in S for l in L for t in T), name="xandu") 

    #R3: Mínimo de iluminación diaria
    # model.addConstrs((u_s_l_t[s,l,t] <= y_s_l_t[s,l,t] for s in S for t in T for l in L), name = "min_iluminacion_diaria")
    
    #R4:  Límite de luces por sector
    ## Pendiente por ver el último periodo
    model.addConstrs((sum(x_s_l_t[s, l, t] * data["p_l"][l] for s in S for l in L for t in T)>=data["P_min_s"][(s,t)] for s in S for t in T), name = "limite_luces")
    #R5: Límite de luces por sector
    model.addConstrs((
    quicksum(x_s_l_t[s, l, t] for l in L for t in T) <= data["Ns_max"][s]
    for s in S),
    name="limite_luminarias_totales_por_sector")
    #R6: Compatibilidad luminaria con el sector
    model.addConstrs((x_s_l_t[s, l ,t]<=L_l_s[l,s]*data["M"] for s in S for l in L for t in T), name="compatibilidad")
    #R7: Límite de distintos tipos de luminarias por sector
    # model.addConstrs((
    # quicksum(y_s_l_t[s, l, t] for l in L for t in T) <= data["K_s"][s]
    # for s in S),
    # name="limite_tipos_luminarias_por_sector")
    #R8: Activar desuento asociado a superar la iluminación permitida
    model.addConstrs((sum(data["p_l"][l]*x_s_l_t[s,l,t] for s in S for l in L for t in T)>= data["P_max_s"][(s,t)]*zi_s_t[s,t] for s in S for t in T), name = "iluminacion maxima")
    model.addConstrs((sum(data["p_l"][l]*x_s_l_t[s,l,t] for s in S for l in L for t in T)<= data["P_max_s"][(s,t)] + data["M"]*zi_s_t[s,t] for s in S for t in T), name = "iluminacion maxima 2")
    # #R9: descuento asociado a la iluminacion permitido:
    model.addConstrs((wi_s_t[s,t] >= data["G"]*sum(data["p_l"][l]*x_s_l_t[s,l,t]-data["M"]*(1-zi_s_t[s,t]) for s in S for l in L for t in T) for s in S for t in T), name= "dcto por ilumnacion")
    # #R10: Activación del descuento 
    model.addConstrs((data["M"]*ze_s_t[s,t]+sum(data["Eficiencia"][l] * x_s_l_t[s,l,t] for l in L)>=data["R"] for s in S for t in T), name= "activación dcto")
    # #R11: Definición de dcto. asociado a la eficiencia energética 
    model.addConstrs((we_s_t[s, t] >= data["Ae"]*ze_s_t[s, t] for s in S for t in T), name = "descuento eficiencia")
    #R12: Activación de bonificación en caso de usar luminarias
    model.addConstrs(quicksum(x_s_l_t[s,l,t] for s in S for l in L_solar for t in T)>= data["F"]*zf_s_t[s,t] for s in S for t in T)
    #R13: Definición de bono en caso de usar luminarias tipo l*
    model.addConstrs((wf_s_t[s, t] <= data["Af"]*zf_s_t[s, t] for s in S for t in T), name = "bono solar")
    return model, x_s_l_t, y_s_l_t

def resolver_modelo (model):
    model.optimize()
    return model

def imprimir(model,data,x_s_l_t,u_s_l_t):
    if model.Status == GRB.OPTIMAL:
        # Información general del modelo
        print("Características del modelo:")
        print(f"- Valor objetivo: {model.ObjVal:.2f}")
        if model.IsMIP:
            print(f"- GAP final: {model.MIPGap:.4f}")
        print(f"- Tiempo de resolución: {model.Runtime:.2f} segundos")
        print(f"- Nº variables: {model.NumVars}")
        print(f"- Nº restricciones: {model.NumConstrs}")
        print(f"- Dimensiones: |S|={len(data['S'])}, |L|={len(data['L'])}, |T|={len(data['T'])}\n")
        

        #Imprimir solo variables activas
        print("Luminarias instaladas (x_s_l_t > 0):")
        for s in data["S"]:
            for l in data["L"]:
                for t in data["T"]:
                    val = x_s_l_t[s, l, t].X
                    if val > 1e-6:
                        print(f"- Sector {s}, Luminaria {l}, Periodo {t}: {val:.2f} unidades")
    else:
        print("No se encontró una solución óptima.")
    
def guardar_resultado(model, data, x_s_l_t):
    with open("resultado.txt", "w", encoding="utf-8") as f:
        if model.Status == GRB.OPTIMAL:
            # Información general del modelo
            f.write("Características del modelo:\n")
            f.write(f"- Valor objetivo: {model.ObjVal:.2f}\n")
            if model.IsMIP:
                f.write(f"- GAP final: {model.MIPGap:.4f}\n")
            f.write(f"- Tiempo de resolución: {model.Runtime:.2f} segundos\n")
            f.write(f"- Nº variables: {model.NumVars}\n")
            f.write(f"- Nº restricciones: {model.NumConstrs}\n")
            f.write(f"- Dimensiones: |S|={len(data['S'])}, |L|={len(data['L'])}, |T|={len(data['T'])}\n")

            # Imprimir solo variables activas
            f.write("Luminarias instaladas (x_s_l_t > 0):\n")
            for s in data["S"]:
                for l in data["L"]:
                    for t in data["T"]:
                        val = x_s_l_t[s, l, t].X
                        if val > 1e-6:
                            f.write(f"- Sector {s}, Luminaria {l}, Periodo {t}: {val:.2f} unidades\n")

        else:
            f.write("No se encontró una solución óptima.\n")


def main():
    data = cargar_datos()
    modelo, x, y = construir_model(data)
    resultado = resolver_modelo(modelo)
    guardar_resultado(resultado, data, x)
    imprimir(resultado,data,x,y)

if __name__ == "__main__":
    main() 