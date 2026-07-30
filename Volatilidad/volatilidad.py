"""
volatilidad.py
==============
Módulo de estimación de volatilidad implícita: semilla inicial cerrada
(Corrado-Miller), método de Newton-Raphson, método de la secante, método
de bisección, y construcción de la sonrisa de volatilidad a partir de
precios de mercado observados.

Todas las funciones de este módulo soportan tanto CALL como PUT mediante
el parámetro `tipo='call'/'put'`.

Diseño de semillas iniciales:
- estimacion_sigma_Newton  : si no se pasa 'sigma_estimada', se calcula
                             automáticamente con semilla_corrado_miller.
- estimacion_sigma_tangente: si no se pasan 'sigma0'/'sigma1', se calculan
                             automáticamente a partir de la semilla de
                             Corrado-Miller (sigma0 = semilla,
                             sigma1 = semilla * 1.10).
- estimacion_sigma_biseccion: NO usa Corrado-Miller. 'sigma_low' y
                             'sigma_high' siempre deben ser proporcionados
                             por el usuario, ya que este método requiere
                             un rango que encierre la solución (un solo
                             valor de semilla no define un rango).

Nota sobre semilla_corrado_miller y puts: la fórmula de Corrado & Miller
(1996) está derivada específicamente para el precio de un CALL; no existe
una versión "put" nativa de la fórmula. Cuando se solicita la semilla
para un put (tipo='put'), se convierte el precio de mercado del put a su
CALL sintético equivalente mediante paridad put-call:

    C = P + S0 - K * exp(-r*T)

y se aplica la fórmula de Corrado-Miller sobre ese C sintético.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, TypeAlias

from Opciones.precios import Black_Scholes
from Opciones.griegas import Vega

ArregloComo: TypeAlias = Sequence[float]
MatrizComo: TypeAlias = Sequence[Sequence[float]]


# ================= SEMILLA INICIAL SIGMA (CORRADO-MILLER) =================
def semilla_corrado_miller(s0: float, k: float, t: float, r: float, precio_mercado: float, tipo: str = 'call'):
    """
    Estima una semilla inicial de volatilidad implícita usando la
    aproximación cerrada de Corrado & Miller (1996), útil como punto de
    partida para métodos iterativos (Newton-Raphson, secante, bisección)
    en vez de arrancar siempre con un valor arbitrario como sigma=0.2.

    La fórmula de Corrado-Miller está derivada específicamente para el
    precio de un CALL. Si tipo='put', el precio de mercado del put se
    convierte primero a su CALL sintético equivalente mediante paridad
    put-call (C = P + S0 - K*exp(-r*T)) antes de aplicar la fórmula.

    Parámetros
    ----------
    s0             : Precio spot del subyacente.
    k              : Precio de ejercicio (strike).
    t              : Tiempo al vencimiento, en años.
    r              : Tasa libre de riesgo anualizada.
    precio_mercado : Precio de mercado observado (del call o del put,
                     según 'tipo').
    tipo           : 'call' o 'put'.

    Regresa
    -------
    float : semilla de volatilidad estimada (acotada a un mínimo de 1e-4
            para evitar semillas negativas o nulas en mercados muy fuera
            del dinero, donde el discriminante clippeado a 0 podría
            producir un valor degenerado).
    """
    k_d = k * np.exp(-r * t)  # se calcula el descuento del strike

    if tipo.lower() == 'put':
        call_mercado = precio_mercado + s0 - k_d # Paridad put-call: convertimos el precio del put a su call sintético equivalente
    else:
        call_mercado = precio_mercado

    M = s0 - k_d  # Diferencia de precio
    Call_ajustado = call_mercado - M / 2
    D = Call_ajustado ** 2 - M ** 2 / np.pi  # discriminante

    # si D < 0 <=> valores de mercado imposibles o arbitraje extremo
    if D < 0:
        D = 0

    V = Call_ajustado + np.sqrt(D)  # componente temporal/volatilidad combinada
    P_base = s0 + k_d  # Precio Base

    sigma_semilla = (np.sqrt(2 * np.pi) * V) / (P_base * np.sqrt(t))

    # Salvaguarda: si el mercado está muy out-of-the-money, V (y por lo tanto sigma_semilla) podría salir negativo o casi cero.
    return max(sigma_semilla, 1e-4)


# ================= ESTIMACIONES DE SIGMA (VOLATILIDAD)=====================================

def estimacion_sigma_Newton(s0: float, k: float, t: float, r: float, precio_mercado: float,
                            sigma_estimada: float = None, tipo: str = 'call', error=0.0001, max_iter=100):
    """
    Estima la volatilidad implícita de una opción europea (call o put)
    mediante el método de Newton-Raphson, buscando el sigma que iguala
    el precio de Black-Scholes al precio observado en mercado.

    Parámetros
    ----------
    s0             : Precio spot del subyacente.
    k              : Precio de ejercicio (strike).
    t              : Tiempo al vencimiento, en años.
    r              : Tasa libre de riesgo anualizada.
    precio_mercado : Precio de mercado observado (del call o del put,
                     según 'tipo').
    sigma_estimada : Volatilidad inicial (semilla) para iniciar la
                     iteración. Si no se proporciona (None), se calcula
                     automáticamente con semilla_corrado_miller (usando
                     el mismo 'tipo').
    tipo           : 'call' o 'put'.
    error          : Tolerancia de convergencia sobre la diferencia de precio.
    max_iter       : Número máximo de iteraciones.

    Regresa
    -------
    float : volatilidad implícita estimada.
    """
    if sigma_estimada is None:
        sigma_estimada = semilla_corrado_miller(s0, k, t, r, precio_mercado, tipo)

    for i in range(max_iter):
        # Recalcular d1 y precio con el sigma de esta iteración
        # (d1 no depende de 'tipo': es el mismo para call y put)
        d1 = (np.log(s0 / k) + (r + sigma_estimada ** 2 / 2) * t) / (sigma_estimada * np.sqrt(t))
        precio_modelo = Black_Scholes(s0, k, t, r, sigma_estimada, tipo)
        diff = precio_modelo - precio_mercado

        # Si la diferencia de precio es menor al error, terminamos
        if abs(diff) < error:
            print(f"iter={i + 1}, "f"sigma_actual={sigma_estimada:.10f}, "f"precio={precio_modelo:.10f}, "f"error={abs(diff):.10e}")

            return sigma_estimada

        print(f"iter={i + 1}, "f"sigma_actual={sigma_estimada:.10f}, "f"precio={precio_modelo:.10f}, "f"error={abs(diff):.10e}")

        v = Vega(s0, t, sigma_estimada, d1)  # Vega es la misma fórmula para call y put, no depende de 'tipo'.

        sigma_estimada = sigma_estimada - diff / v  # Newton-Raphson: nueva_sigma = sigma - f(sigma)/f'(sigma)

        # Evitar sigmas negativos o cero
        sigma_estimada = max(1e-5, sigma_estimada)

    return sigma_estimada


def estimacion_sigma_tangente(s0: float, k: float, t: float, r: float, precio_mercado: float,
                              sigma0: float = None, sigma1: float = None, tipo: str = 'call',
                              error=1e-6, max_iter=100):
    """
    Estima la volatilidad implícita de una opción europea (call o put)
    mediante el método de la secante (dos puntos iniciales, sin
    necesidad de derivada analítica como Vega), como alternativa a
    Newton-Raphson o bisección.

    Parámetros
    ----------
    s0             : Precio spot del subyacente.
    k              : Precio de ejercicio (strike).
    t              : Tiempo al vencimiento, en años.
    r              : Tasa libre de riesgo anualizada.
    precio_mercado : Precio de mercado observado (del call o del put,
                     según 'tipo').
    sigma0         : Primera semilla de volatilidad. Si no se proporciona
                     (None), se calcula automáticamente con
                     semilla_corrado_miller (usando el mismo 'tipo').
    sigma1         : Segunda semilla de volatilidad (distinta de sigma0).
                     Si no se proporciona (None), se calcula
                     automáticamente como sigma0 * 1.10.
    tipo           : 'call' o 'put'.
    error          : Tolerancia de convergencia (sobre el residuo de
                     precio o sobre el cambio entre iteraciones
                     consecutivas de sigma).
    max_iter       : Número máximo de iteraciones.

    Regresa
    -------
    float : volatilidad implícita estimada (la mejor aproximación
            disponible, incluso si no se alcanzó la tolerancia deseada
            dentro de max_iter iteraciones).
    """
    if sigma0 is None:
        sigma0 = semilla_corrado_miller(s0, k, t, r, precio_mercado, tipo)
    if sigma1 is None:
        sigma1 = sigma0 * 1.10

    sigma_n0 = sigma0
    sigma_n1 = sigma1

    f_n0 = Black_Scholes(s0, k, t, r, sigma_n0, tipo) - precio_mercado
    f_n1 = Black_Scholes(s0, k, t, r, sigma_n1, tipo) - precio_mercado

    sigma_aux = sigma_n1  # valor por defecto si max_iter fuera 0

    for i in range(max_iter):

        if abs(f_n1 - f_n0) < 1e-15:
            print('la pendiente es cero; no se puede continuar con el método de la secante')
            return sigma_n1

        sigma_aux = sigma_n1 - f_n1 * (sigma_n1 - sigma_n0) / (f_n1 - f_n0)

        f_aux = Black_Scholes(s0, k, t, r, sigma_aux, tipo) - precio_mercado

        if abs(f_aux) < error or abs(sigma_aux - sigma_n1) < error:
            return sigma_aux

        sigma_n0, f_n0 = sigma_n1, f_n1
        sigma_n1, f_n1 = sigma_aux, f_aux

    print(f"Máximo número de iteraciones alcanzado. La convergencia no fue completa. Retornando sigma_actual: {sigma_aux}")

    return sigma_aux


def estimacion_sigma_biseccion(s0: float, k: float, t: float, r: float, precio_mercado: float,
                               sigma_low: float, sigma_high: float, tipo: str = 'call',
                               error=0.0001, max_iter=100):
    """
    Estima la volatilidad implícita de una opción europea (call o put)
    mediante el método de bisección, como alternativa a Newton-Raphson
    cuando no se cuenta con una buena semilla inicial.

    NOTA: a diferencia de estimacion_sigma_Newton y estimacion_sigma_tangente,
    esta función NO usa semilla_corrado_miller. 'sigma_low' y 'sigma_high'
    siempre deben ser proporcionados por el usuario, ya que la bisección
    requiere un RANGO que encierre la solución (signos opuestos en los
    extremos), y un solo valor de semilla no define ese rango.

    Parámetros
    ----------
    s0             : Precio spot del subyacente.
    k              : Precio de ejercicio (strike).
    t              : Tiempo al vencimiento, en años.
    r              : Tasa libre de riesgo anualizada.
    precio_mercado : Precio de mercado observado (del call o del put,
                     según 'tipo').
    sigma_low      : Límite inferior del rango de búsqueda de sigma.
    sigma_high     : Límite superior del rango de búsqueda de sigma.
    tipo           : 'call' o 'put'.
    error          : Tolerancia de convergencia sobre la diferencia de precio.
    max_iter       : Número máximo de iteraciones.

    Regresa
    -------
    float : volatilidad implícita estimada, o None si el rango inicial no
            encierra la solución.
    """
    # Calcular los valores de la función en los límites iniciales
    f_low = Black_Scholes(s0, k, t, r, sigma_low, tipo) - precio_mercado
    f_high = Black_Scholes(s0, k, t, r, sigma_high, tipo) - precio_mercado

    # Asegurarse de que el rango inicial encierre la solución (f_low y f_high deben tener signos opuestos)
    if f_low * f_high > 0:
        print("Las volatilidades iniciales no encierran la solución. Intenta con un rango diferente.")
        return None

    sigma_mid = 0.0  # Variable para almacenar el punto medio actual

    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        f_mid = Black_Scholes(s0, k, t, r, sigma_mid, tipo) - precio_mercado

        # Imprimir el progreso
        print(f"iter={i + 1}, sigma_actual={sigma_mid:.10f}, precio_modelo={Black_Scholes(s0, k, t, r, sigma_mid, tipo):.10f}, error={abs(f_mid):.10e}")

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


def sonrisa_volatilidad(s0: float, r: float, t: float, strikes: float, precios: float,
                        sigma_inicial=None, tipo: str = 'call'):
    """
    Construye y grafica la sonrisa de volatilidad implícita para un
    conjunto de strikes y sus precios de mercado observados.

    Parámetros
    ----------
    s0            : Precio spot del subyacente.
    r             : Tasa libre de riesgo anualizada.
    t             : Tiempo al vencimiento, en años.
    strikes       : Arreglo de precios de ejercicio.
    precios       : Arreglo de precios de mercado correspondientes a cada
                    strike (call o put, según 'tipo').
    sigma_inicial : Volatilidad inicial (semilla) para cada estimación.
                    Si no se proporciona (None), estimacion_sigma_Newton
                    calcula automáticamente una semilla distinta para cada
                    strike con semilla_corrado_miller.
    tipo          : 'call' o 'put'.

    Regresa
    -------
    np.ndarray : arreglo de volatilidades implícitas, una por strike.
    """
    vols = []

    for K, precio in zip(strikes, precios):
        # Pasamos el precio de mercado correspondiente a cada Strike
        sigma_imp = estimacion_sigma_Newton(s0, K, t, r, precio, sigma_inicial, tipo)
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