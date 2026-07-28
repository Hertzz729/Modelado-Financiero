"""
volatilidad.py
==============
Módulo de estimación de volatilidad implícita: método de Newton-Raphson,
método de bisección, y construcción de la sonrisa de volatilidad a partir
de precios de mercado observados.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from decimal import Decimal
from typing import Sequence, TypeAlias

from precios import Black_Scholes
from griegas import Vega

ArregloComo: TypeAlias = Sequence[float]
MatrizComo: TypeAlias = Sequence[Sequence[float]]


# ================= ESTIMACIONES DE SIGMA (VOLATILIDAD)=====================================

def estimacion_sigma(s0: float, k: float, t: float, r: float, sigma_estimada: float, call_mercado: float,
                      error=0.0001, max_iter=100):
    """
    Estima la volatilidad implícita de un call europeo mediante el método
    de Newton-Raphson, buscando el sigma que iguala el precio de
    Black-Scholes al precio observado en mercado.

    Parámetros
    ----------
    s0             : Precio spot del subyacente.
    k              : Precio de ejercicio (strike).
    t              : Tiempo al vencimiento, en años.
    r              : Tasa libre de riesgo anualizada.
    sigma_estimada : Volatilidad inicial (semilla) para iniciar la iteración.
    call_mercado   : Precio de mercado observado del call.
    error          : Tolerancia de convergencia sobre la diferencia de precio.
    max_iter       : Número máximo de iteraciones.

    Regresa
    -------
    float : volatilidad implícita estimada.
    """
    for i in range(max_iter):
        # Recalcular d1 y precio con el sigma de esta iteración
        d1 = (np.log(s0/k) + (r + sigma_estimada**2/2)*t) / (sigma_estimada*np.sqrt(t))
        precio_modelo = Black_Scholes(s0, k, t, r, sigma_estimada, "call")
        diff = precio_modelo - call_mercado

        # Si la diferencia de precio es menor al error, terminamos
        if abs(diff) < error:
            print(f"iter={i+1}, "f"sigma_actual={sigma_estimada:.10f}, "f"precio={precio_modelo:.10f}, "f"error={abs(diff):.10e}")
            return sigma_estimada

        print(f"iter={i+1}, "f"sigma_actual={sigma_estimada:.10f}, "f"precio={precio_modelo:.10f}, "f"error={abs(diff):.10e}")

        v = Vega(s0, t, sigma_estimada, d1)

        # Newton-Raphson: nueva_sigma = sigma - f(sigma)/f'(sigma)
        sigma_estimada = sigma_estimada - diff / v

        # Evitar sigmas negativos o cero
        sigma_estimada = max(1e-5, sigma_estimada)

    return sigma_estimada


def sonrisa_volatilidad(s0: float, r: float, t: float, strikes: float, precios: float, sigma_inicial=0.2):
    """
    Construye y grafica la sonrisa de volatilidad implícita para un
    conjunto de strikes y sus precios de mercado observados.

    Parámetros
    ----------
    s0            : Precio spot del subyacente.
    r             : Tasa libre de riesgo anualizada.
    t             : Tiempo al vencimiento, en años.
    strikes       : Arreglo de precios de ejercicio.
    precios       : Arreglo de precios de mercado (call) correspondientes
                    a cada strike.
    sigma_inicial : Volatilidad inicial (semilla) para cada estimación.

    Regresa
    -------
    np.ndarray : arreglo de volatilidades implícitas, una por strike.
    """
    vols = []

    for K, precio in zip(strikes, precios):
        # Pasamos el precio de mercado correspondiente a cada Strike
        sigma_imp = estimacion_sigma(s0, K, t, r, sigma_inicial, precio)
        vols.append(sigma_imp)

    vols = np.array(vols)

    # --------- GRÁFICA ----------
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, vols, marker='o', linestyle='-', color='b', label='Volatilidad Implícita')
    plt.xlabel("Strike (K)")
    plt.ylabel("sigma Implícita")
    plt.title("Sonrisa de Volatilidad")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return vols


def estimacion_sigma_biseccion(s0: float, k: float, t: float, r: float, call_mercado: float,
                                sigma_low: float, sigma_high: float, error=0.0001, max_iter=100):
    """
    Estima la volatilidad implícita de un call europeo mediante el método
    de bisección, como alternativa a Newton-Raphson cuando no se cuenta
    con una buena semilla inicial.

    Parámetros
    ----------
    s0           : Precio spot del subyacente.
    k            : Precio de ejercicio (strike).
    t            : Tiempo al vencimiento, en años.
    r            : Tasa libre de riesgo anualizada.
    call_mercado : Precio de mercado observado del call.
    sigma_low    : Límite inferior del rango de búsqueda de sigma.
    sigma_high   : Límite superior del rango de búsqueda de sigma.
    error        : Tolerancia de convergencia sobre la diferencia de precio.
    max_iter     : Número máximo de iteraciones.

    Regresa
    -------
    float : volatilidad implícita estimada, o None si el rango inicial no
            encierra la solución.
    """
    # Calcular los valores de la función en los límites iniciales
    f_low = Black_Scholes(s0, k, t, r, sigma_low, "call") - call_mercado
    f_high = Black_Scholes(s0, k, t, r, sigma_high, "call") - call_mercado

    # Asegurarse de que el rango inicial encierre la solución (f_low y f_high deben tener signos opuestos)
    if f_low * f_high > 0:
        print("Las volatilidades iniciales no encierran la solución. Intenta con un rango diferente.")
        return None

    sigma_mid = 0.0  # Variable para almacenar el punto medio actual

    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        f_mid = Black_Scholes(s0, k, t, r, sigma_mid, "call") - call_mercado

        # Imprimir el progreso
        print(f"iter={i+1}, sigma_actual={sigma_mid:.10f}, precio_modelo={Black_Scholes(s0, k, t, r, sigma_mid, 'call'):.10f}, error={abs(f_mid):.10e}")

        # Si la diferencia de precio es menor al error, terminamos
        if abs(f_mid) < error:
            return sigma_mid

        # Actualizar los límites para la siguiente iteración
        if f_mid * f_low < 0:  # El cero está en el rango [sigma_low, sigma_mid]
            sigma_high = sigma_mid
            f_high = f_mid
        else:  # El cero está en el rango [sigma_mid, sigma_high]
            sigma_low = sigma_mid
            f_low = f_mid

        # Asegurarse de que sigma_low y sigma_high sean siempre positivos
        sigma_low = max(1e-5, sigma_low)
        sigma_high = max(1e-5, sigma_high)

    print(f"Máximo número de iteraciones alcanzado. La convergencia no fue completa. Retornando sigma_actual: {sigma_mid}")
    return sigma_mid
