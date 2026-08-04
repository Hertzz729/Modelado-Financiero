"""
volatilidad.py
==============
Módulo de estimación de volatilidad implícita.

Incluye:
- Semilla cerrada de Corrado-Miller.
- Newton-Raphson.
- Secante.
- Bisección.
- Construcción de la sonrisa de volatilidad.

Todas las funciones soportan CALL y PUT mediante
tipo='call' o tipo='put'.

Semillas iniciales
------------------
- estimacion_sigma_Newton:
  usa semilla_corrado_miller si no se proporciona sigma_estimada.

- estimacion_sigma_tangente:
  usa automáticamente sigma0=semilla y sigma1=1.1*semilla si no se
  proporcionan.

- estimacion_sigma_biseccion:
  requiere sigma_low y sigma_high definidos por el usuario, ya que
  necesita un intervalo que encierre la solución.

Nota
----
La fórmula de Corrado-Miller fue derivada para CALLs. Para PUTs, el
precio se transforma a su CALL sintético equivalente mediante paridad
put-call antes de aplicar la aproximación.

"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, TypeAlias

from ..opciones.precios import Black_Scholes
from ..opciones.griegas import Vega

ArregloComo: TypeAlias = Sequence[float]
MatrizComo: TypeAlias = Sequence[Sequence[float]]


# ================= VALIDACIÓN COMÚN: LÍMITES DE NO-ARBITRAJE =================
def _validar_precio_mercado(s0: float, k: float, t: float, r: float, precio_mercado: float, tipo: str):
    """
    Verifica que el precio de mercado respete los límites de no-arbitraje de
    Black-Scholes.

    Si el precio observado queda fuera de dichos límites, no existe una
    volatilidad implícita positiva consistente con el modelo.

    Call:
        max(s0 - k*exp(-r*t), 0) <= precio <= s0

    Put:
        max(k*exp(-r*t) - s0, 0) <= precio <= k*exp(-r*t)

    Lanza
    -----
    ValueError : Si los parámetros son inválidos o si el precio viola los
    límites de no-arbitraje.
    """
    if s0 <= 0 or k <= 0:
        raise ValueError("'s0' y 'k' deben ser positivos")
    if t <= 0:
        raise ValueError("'t' debe ser positivo")

    tipo = tipo.lower()
    if tipo not in ('call', 'put'):
        raise ValueError(f"'tipo' debe ser 'call' o 'put', se recibió: {tipo}")

    k_d = k * np.exp(-r * t)

    if tipo == 'call':
        limite_inf = max(s0 - k_d, 0.0)
        limite_sup = s0
    else:
        limite_inf = max(k_d - s0, 0.0)
        limite_sup = k_d

    if not (limite_inf <= precio_mercado <= limite_sup):
        raise ValueError(
            f"'precio_mercado' ({precio_mercado:.6f}) viola los límites de "
            f"no-arbitraje para un {tipo} [{limite_inf:.6f}, {limite_sup:.6f}]. "
            "No existe una volatilidad implícita real que explique este precio; "
            "revisa el dato de entrada."
        )


# ================= SEMILLA INICIAL SIGMA (CORRADO-MILLER) =================
def semilla_corrado_miller(s0: float, k: float, t: float, r: float, precio_mercado: float, tipo: str = 'call'):
    """
    Estima una semilla inicial de volatilidad implícita mediante la
    aproximación cerrada de Corrado-Miller (1996).

    Útil como punto de partida para métodos iterativos de estimación de
    volatilidad implícita.

    Parámetros
    ----------
    s0             : Precio spot.
    k              : Strike.
    t              : Tiempo al vencimiento.
    r              : Tasa libre de riesgo.
    precio_mercado : Precio observado de mercado.
    tipo           : 'call' o 'put'.

    Regresa
    -------
    float : Semilla inicial de volatilidad implícita.

    Lanza
    -----
    ValueError : Si los parámetros son inválidos o el precio viola los
    límites de no-arbitraje.
    """
    _validar_precio_mercado(s0, k, t, r, precio_mercado, tipo)

    k_d = k * np.exp(-r * t)  # se calcula el descuento del strike

    if tipo.lower() == 'put':
        call_mercado = precio_mercado + s0 - k_d  # Paridad put-call: convertimos el precio del put a su call sintético equivalente
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
                            sigma_estimada: float = None, tipo: str = 'call', error=0.0001,
                            max_iter=100, vega_min=1e-8, verbose=True):
    """
    Estima la volatilidad implícita de una opción europea mediante el método
    de Newton-Raphson.

    Parámetros
    ----------
    s0             : Precio spot.
    k              : Strike.
    t              : Tiempo al vencimiento.
    r              : Tasa libre de riesgo.
    precio_mercado : Precio observado de mercado.
    sigma_estimada : Semilla inicial. Si es None, se utiliza
                     semilla_corrado_miller.
    tipo           : 'call' o 'put'.
    error          : Tolerancia de convergencia.
    max_iter       : Máximo número de iteraciones.
    vega_min       : Umbral mínimo de Vega para evitar inestabilidad
                     numérica.
    verbose        : Si True, muestra el progreso de la iteración.

    Regresa
    -------
    float : Volatilidad implícita estimada.

    Lanza
    -----
    ValueError : Si los parámetros son inválidos o el precio viola los
    límites de no-arbitraje.
    """
    _validar_precio_mercado(s0, k, t, r, precio_mercado, tipo)

    if sigma_estimada is None:
        sigma_estimada = semilla_corrado_miller(s0, k, t, r, precio_mercado, tipo)

    for i in range(max_iter):
        # Recalcular d1 y precio con el sigma de esta iteración
        # (d1 no depende de 'tipo': es el mismo para call y put)
        d1 = (np.log(s0 / k) + (r + sigma_estimada ** 2 / 2) * t) / (sigma_estimada * np.sqrt(t))
        precio_modelo = Black_Scholes(s0, k, t, r, sigma_estimada, tipo)
        diff = precio_modelo - precio_mercado

        if verbose:
            print(f"iter={i + 1}, sigma_actual={sigma_estimada:.10f}, "
                  f"precio={precio_modelo:.10f}, error={abs(diff):.10e}")

        # Si la diferencia de precio es menor al error, terminamos
        if abs(diff) < error:
            return sigma_estimada

        v = Vega(s0, t, sigma_estimada, d1)  # Vega es la misma fórmula para call y put, no depende de 'tipo'.

        if abs(v) < vega_min:
            print(
                f"Vega demasiado pequeña (|Vega|={abs(v):.3e} < {vega_min:.1e}) en "
                f"iter={i + 1}; el paso de Newton es numéricamente inestable. "
                f"Deteniendo iteración. Retornando sigma_actual: {sigma_estimada}"
            )
            return sigma_estimada

        sigma_estimada = sigma_estimada - diff / v  # Newton-Raphson: nueva_sigma = sigma - f(sigma)/f'(sigma)

        # Evitar sigmas negativos o cero
        sigma_estimada = max(1e-5, sigma_estimada)

    print(f"Máximo número de iteraciones alcanzado. La convergencia no fue completa. "
          f"Retornando sigma_actual: {sigma_estimada}")

    return sigma_estimada


def estimacion_sigma_tangente(s0: float, k: float, t: float, r: float, precio_mercado: float,
                              sigma0: float = None, sigma1: float = None, tipo: str = 'call',
                              error=1e-6, max_iter=100, verbose=True):
    """
    Estima la volatilidad implícita de una opción europea mediante el método
    de la secante.

    Parámetros
    ----------
    s0             : Precio spot.
    k              : Strike.
    t              : Tiempo al vencimiento.
    r              : Tasa libre de riesgo.
    precio_mercado : Precio observado de mercado.
    sigma0         : Primera semilla. Si es None, se utiliza
                     semilla_corrado_miller.
    sigma1         : Segunda semilla. Si es None, se utiliza
                     1.1 * sigma0.
    tipo           : 'call' o 'put'.
    error          : Tolerancia de convergencia.
    max_iter       : Máximo número de iteraciones.
    verbose        : Si True, muestra información del proceso.

    Regresa
    -------
    float : Volatilidad implícita estimada.

    Lanza
    -----
    ValueError : Si los parámetros son inválidos.
    """
    _validar_precio_mercado(s0, k, t, r, precio_mercado, tipo)

    if sigma0 is None:
        sigma0 = semilla_corrado_miller(s0, k, t, r, precio_mercado, tipo)
    elif sigma0 <= 0:
        raise ValueError("'sigma0' debe ser positiva")

    if sigma1 is None:
        sigma1 = sigma0 * 1.10
    elif sigma1 <= 0:
        raise ValueError("'sigma1' debe ser positiva")

    sigma_n0 = sigma0
    sigma_n1 = sigma1

    f_n0 = Black_Scholes(s0, k, t, r, sigma_n0, tipo) - precio_mercado
    f_n1 = Black_Scholes(s0, k, t, r, sigma_n1, tipo) - precio_mercado

    sigma_aux = sigma_n1  # valor por defecto si max_iter fuera 0

    for i in range(max_iter):

        if abs(f_n1 - f_n0) < 1e-15:
            if verbose:
                print('la pendiente es cero; no se puede continuar con el método de la secante')
            return sigma_n1

        sigma_aux = sigma_n1 - f_n1 * (sigma_n1 - sigma_n0) / (f_n1 - f_n0)

        # Evitar que la extrapolación lineal produzca un sigma no positivo
        sigma_aux = max(sigma_aux, 1e-6)

        f_aux = Black_Scholes(s0, k, t, r, sigma_aux, tipo) - precio_mercado

        if verbose:
            print(f"iter={i + 1}, sigma_actual={sigma_aux:.10f}, error={abs(f_aux):.10e}")

        if abs(f_aux) < error or abs(sigma_aux - sigma_n1) < error:
            return sigma_aux

        sigma_n0, f_n0 = sigma_n1, f_n1
        sigma_n1, f_n1 = sigma_aux, f_aux

    print(f"Máximo número de iteraciones alcanzado. La convergencia no fue completa. Retornando sigma_actual: {sigma_aux}")

    return sigma_aux


def estimacion_sigma_biseccion(s0: float, k: float, t: float, r: float, precio_mercado: float,
                               sigma_low: float, sigma_high: float, tipo: str = 'call',
                               error=0.0001, max_iter=100, verbose=True):
    """
    Estima la volatilidad implícita de una opción europea mediante el método
    de bisección.

    Nota
    ----
    A diferencia de Newton-Raphson y la secante, este método requiere que el
    usuario proporcione un intervalo [sigma_low, sigma_high] que encierre la
    solución.

    Parámetros
    ----------
    s0             : Precio spot.
    k              : Strike.
    t              : Tiempo al vencimiento.
    r              : Tasa libre de riesgo.
    precio_mercado : Precio observado de mercado.
    sigma_low      : Límite inferior del intervalo.
    sigma_high     : Límite superior del intervalo.
    tipo           : 'call' o 'put'.
    error          : Tolerancia de convergencia.
    max_iter       : Máximo número de iteraciones.
    verbose        : Si True, muestra el progreso.

    Regresa
    -------
    float | None : Volatilidad implícita estimada.

    Lanza
    -----
    ValueError : Si los parámetros son inválidos.
    """
    _validar_precio_mercado(s0, k, t, r, precio_mercado, tipo)

    if sigma_low <= 0 or sigma_high <= 0:
        raise ValueError("'sigma_low' y 'sigma_high' deben ser positivos")
    if sigma_low >= sigma_high:
        raise ValueError("'sigma_low' debe ser menor que 'sigma_high'")

    # Calcular los valores de la función en los límites iniciales
    f_low = Black_Scholes(s0, k, t, r, sigma_low, tipo) - precio_mercado
    f_high = Black_Scholes(s0, k, t, r, sigma_high, tipo) - precio_mercado

    # Asegurarse de que el rango inicial encierre la solución (f_low y f_high deben tener signos opuestos)
    if f_low * f_high > 0:
        print("Las volatilidades iniciales no encierran la solución. Intenta con un rango diferente.")
        return None

    sigma_mid = (sigma_low + sigma_high) / 2  # valor por defecto si max_iter fuera 0

    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        f_mid = Black_Scholes(s0, k, t, r, sigma_mid, tipo) - precio_mercado

        if verbose:
            print(f"iter={i + 1}, sigma_actual={sigma_mid:.10f}, "
                  f"precio_modelo={Black_Scholes(s0, k, t, r, sigma_mid, tipo):.10f}, "
                  f"error={abs(f_mid):.10e}")

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

    print(f"Máximo número de iteraciones alcanzado. La convergencia no fue completa. Retornando sigma_actual: {sigma_mid}")

    return sigma_mid


def sonrisa_volatilidad(s0: float, r: float, t: float, strikes: ArregloComo, precios: ArregloComo,
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
                    strike (call o put, según 'tipo'). Debe tener la
                    misma longitud que 'strikes'.
    sigma_inicial : volatilidad inicial (semilla) para cada estimación.
                    Si no se proporciona (None), estimacion_sigma_Newton
                    calcula automáticamente una semilla distinta para cada
                    strike con semilla_corrado_miller.
    tipo          : 'call' o 'put'.

    Regresa
    -------
    np.ndarray : arreglo de volatilidades implícitas, una por strike.

    Lanza
    -----
  ValueError : Si los arreglos tienen dimensiones incompatibles o si algún
    precio viola los límites de no-arbitraje.
    """
    strikes = np.asarray(strikes, dtype=float)
    precios = np.asarray(precios, dtype=float)

    if strikes.shape[0] != precios.shape[0]:
        raise ValueError(
            f"'strikes' ({strikes.shape[0]} elementos) y 'precios' "
            f"({precios.shape[0]} elementos) deben tener la misma longitud"
        )

    vols = []

    for K, precio in zip(strikes, precios):
        # Pasamos el precio de mercado correspondiente a cada Strike;
        # verbose=False para no imprimir el detalle de cada iteración
        # por cada strike (sería muy ruidoso en un barrido de varios strikes).
        sigma_imp = estimacion_sigma_Newton(s0, K, t, r, precio, sigma_inicial, tipo, verbose=False)
        vols.append(sigma_imp)

    vols = np.array(vols)

    # --------- GRÁFICA ----------
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, vols, marker='o', linestyle='-', color='b', label='volatilidad Implícita')
    plt.xlabel("Strike (K)")
    plt.ylabel("sigma Implícita")
    plt.title("Sonrisa de volatilidad")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return vols