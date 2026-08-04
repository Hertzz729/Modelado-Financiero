"""
griegas.py
==========

Módulo de letras griegas para:

- Black-Scholes estándar (sin dividendos).
- Opciones europeas con dividendos discretos (Theta diaria).
- Black-76 (opciones sobre futuros/forwards).
- Opciones americanas con dividendos discretos.

Las griegas de Black-Scholes y Black-76 se calculan mediante fórmulas
analíticas. Las griegas de opciones americanas con dividendos se
aproximan mediante diferencias finitas sobre el precio calculado por
`_precio_por_metodo`.
"""

from typing import Sequence, TypeAlias
import numpy as np
from scipy.stats import norm

from .precios import _precio_por_metodo, Black_76, ajuste_s0

ArregloComo: TypeAlias = Sequence[float]
MatrizComo: TypeAlias = Sequence[Sequence[float]]


# ================Letras griegas para opciones (Black-Scholes estándar)======================

def Delta(s0: float, k: float, t: float, r: float, sigma: float, tipo='call'):
    """
    Delta analítica de Black-Scholes estándar (sin dividendos).

    Parámetros
    ----------
    s0    : Precio spot del subyacente.
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    r     : Tasa libre de riesgo anualizada.
    sigma : Volatilidad anualizada.
    tipo  : 'call' o 'put'.

    Regresa
    -------
    float : Delta del call (en [0, 1]) o del put (en [-1, 0]).
    """
    d1 = (np.log(s0/k) + (r+sigma**2/2)*t)/(sigma*np.sqrt(t))
    if tipo.lower() == 'call':
        return norm.cdf(d1)
    else:
        # Identidad: Delta_put = Delta_call - 1 = N(d1) - 1
        return norm.cdf(d1) - 1


def Theta(s0: float, k: float, t: float, r: float, sigma: float, d1: float, d2: float, tipo="call"):
    """
    Theta analítica de Black-Scholes estándar (sin dividendos), anualizada.

    Parámetros
    ----------
    s0    : Precio spot actual del subyacente.
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    r     : Tasa libre de riesgo anualizada.
    sigma : volatilidad anualizada del subyacente.
    d1,d2 : d1 y d2 de Black-Scholes, calculados con los mismos s0,k,t,r,sigma.
    tipo  : 'call' o 'put'.

    Regresa
    -------
    float : Theta (tasa de cambio del precio respecto al tiempo).
    """
    if tipo.lower() == "call":
        return - (s0 * 1/np.sqrt(2 * np.pi) * sigma * np.exp(-d1**2/2)) / (2*np.sqrt(t)) - r*k*np.exp(-r*t)*norm.cdf(d2)
    elif tipo.lower() == "put":
        return - (s0 * 1/np.sqrt(2 * np.pi) * sigma * np.exp(-d1**2/2)) / (2*np.sqrt(t)) + r*k*np.exp(-r*t)*norm.cdf(-d2)
    else:
        raise ValueError("tipo debe ser 'call' or 'put'")


def Gamma(s0: float, sigma: float, t: float, d1: float):
    """
    Gamma analítica de Black-Scholes estándar (igual para call y put).

    Parámetros
    ----------
    s0    : Precio spot del subyacente.
    sigma : Volatilidad anualizada.
    t     : Tiempo al vencimiento, en años.
    d1    : Parámetro d1 de Black-Scholes.

    Regresa
    -------
    float : Gamma (segunda derivada del precio respecto a s0).
    """
    return np.exp(-d1**2/2)/(np.sqrt(2 * np.pi) * s0 * sigma*np.sqrt(t))


def Vega(s0: float, t: float, sigma: float, d1: float):
    """
    Vega analítica de Black-Scholes estándar (igual para call y put).

    Parámetros
    ----------
    s0    : Precio spot del subyacente.
    t     : Tiempo al vencimiento, en años.
    sigma : Volatilidad anualizada.
    d1    : Parámetro d1 de Black-Scholes.

    Regresa
    -------
    float : Vega (derivada del precio respecto a sigma).
    """
    return s0 * np.sqrt(t) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)


def Rho(k: float, t: float, r: float, d2: float, tipo='call'):
    """
    Rho analítica de Black-Scholes estándar (sin dividendos): derivada del
    precio respecto a la tasa libre de riesgo r.

    Parámetros
    ----------
    k    : Precio de ejercicio (strike).
    t    : Tiempo al vencimiento, en años.
    r    : Tasa libre de riesgo anualizada.
    d2   : d2 de Black-Scholes.
    tipo : 'call' o 'put'.

    Regresa
    -------
    float : Rho de la opción.
    """
    if tipo.lower() == 'call':
        return k * t * np.exp(-r * t) * norm.cdf(d2)
    else:
        return -k * t * np.exp(-r * t) * norm.cdf(-d2)


# Theta universal para dividendos y sin
def Theta_Diaria(s0: float, k: float, t: float, sigma: float, r: float,
                  dividendos: list = None, t_dividendos: list = None, tipo: str = "call") -> float:
    """
    Calcula la Theta diaria (pérdida/ganancia de valor por día transcurrido)
    mediante el método numérico de diferencias finitas.
    Funciona de forma exacta tanto para opciones CON como SIN dividendos.

    Convención de día: se usa 1/365 (día calendario), NO 1/252 (día
    hábil). Esta convención es distinta a la usada en `Theta_A_div`
    (opciones americanas con dividendos), que usa 1/252. Los resultados
    de ambas funciones NO son directamente comparables entre sí sin
    ajustar por esta diferencia.

    IMPORTANTE: 's0' debe pasarse SIN ajustar por dividendos; esta función
    ya realiza internamente el ajuste de valor presente de los dividendos
    (vía `ajuste_s0`) tanto para "hoy" como para "mañana". Pasar un s0 ya
    ajustado provocaría restar el valor presente de los dividendos dos veces.

    Parámetros
    ----------
    s0            : Precio spot del subyacente.
    k             : Precio de ejercicio (strike).
    t             : Tiempo al vencimiento, en años.
    sigma         : Volatilidad anualizada.
    r             : Tasa libre de riesgo anualizada.
    dividendos    : Dividendos discretos.
    t_dividendos  : Fechas de pago de dividendos, en años.
    tipo          : 'call' o 'put'.

    Regresa
    -------
    float : Theta diaria (precio de mañana menos precio de hoy).
            np.nan si el tiempo restante es menor o igual a un día.
    """
    if dividendos is None:
        dividendos = []
    if t_dividendos is None:
        t_dividendos = []

    dt = 1 / 365.0  # Un día en términos anuales

    if t <= dt:
        return np.nan

    # --- 1. PRECIO HOY ---
    s0_adj_hoy = ajuste_s0(s0, r, t, dividendos, t_dividendos)

    d1_hoy = (np.log(s0_adj_hoy / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2_hoy = d1_hoy - sigma * np.sqrt(t)

    if tipo.lower() == "call":
        precio_hoy = s0_adj_hoy * norm.cdf(d1_hoy) - k * np.exp(-r * t) * norm.cdf(d2_hoy)
    elif tipo.lower() == "put":
        precio_hoy = k * np.exp(-r * t) * norm.cdf(-d2_hoy) - s0_adj_hoy * norm.cdf(-d1_hoy)
    else:
        raise ValueError("tipo debe ser 'call' o 'put'")

    # --- 2. PRECIO MAÑANA (Avanzamos 1 día) ---
    t_manana = t - dt
    t_divs_manana = [td - dt for td in t_dividendos]

    s0_adj_manana = ajuste_s0(s0, r, t_manana, dividendos, t_divs_manana)

    d1_manana = (np.log(s0_adj_manana / k) + (r + 0.5 * sigma ** 2) * t_manana) / (sigma * np.sqrt(t_manana))
    d2_manana = d1_manana - sigma * np.sqrt(t_manana)

    if tipo.lower() == "call":
        precio_manana = s0_adj_manana * norm.cdf(d1_manana) - k * np.exp(-r * t_manana) * norm.cdf(d2_manana)
    else:
        precio_manana = k * np.exp(-r * t_manana) * norm.cdf(-d2_manana) - s0_adj_manana * norm.cdf(-d1_manana)

    # --- 3. THETA DIARIA ---
    return precio_manana - precio_hoy

# =============== LETRAS GRIEGAS PARA BLACK-76 (FORWARDS) ================

def Delta_B76(r: float, t: float, d1: float, tipo='call') -> float:
    """
    Delta analítica de Black-76 (opciones sobre futuros/forwards).

    Parámetros
    ----------
    r     : Tasa libre de riesgo anualizada.
    t     : Tiempo al vencimiento, en años.
    d1    : Parámetro d1 de Black-76.
    tipo  : 'call' o 'put'.

    Regresa
    -------
    float : Delta de la opción.
    """
    df = np.exp(-r * t)
    if tipo.lower() == 'call':
        return df * norm.cdf(d1)
    elif tipo.lower() == 'put':
        return -df * norm.cdf(-d1)
    else:
        raise ValueError("El tipo debe ser 'call' o 'put'")


def Gamma_B76(f0: float, r: float, t: float, sigma: float, d1: float) -> float:
    """
    Gamma analítica de Black-76 (igual para call y put).

    Parámetros
    ----------
    f0    : Precio forward o futuro.
    r     : Tasa libre de riesgo anualizada.
    t     : Tiempo al vencimiento, en años.
    sigma : Volatilidad anualizada.
    d1    : Parámetro d1 de Black-76.

    Regresa
    -------
    float : Gamma de la opción.
    """
    df = np.exp(-r * t)
    return (df * norm.pdf(d1)) / (f0 * sigma * np.sqrt(t))


def Theta_B76(f0: float, k: float, t: float, r: float, sigma: float, d1: float, tipo='call') -> float:
    """
    Theta analítica de Black-76.

    NOTA: el precio de la opción se recalcula internamente (llamando a
    Black_76) en vez de recibirse como argumento externo. La fórmula en sí
    tiene la misma forma para call y put; lo único que cambia es qué
    precio (call o put) se usa.

    Parámetros
    ----------
    f0    : Precio forward o futuro.
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    r     : Tasa libre de riesgo anualizada.
    sigma : Volatilidad anualizada.
    d1    : Parámetro d1 de Black-76.
    tipo  : 'call' o 'put'.

    Regresa
    -------
    float : Theta de la opción.
    """
    precio = Black_76(f0, k, t, r, sigma, tipo)
    df = np.exp(-r * t)
    primer_termino = -(f0 * sigma * df * norm.pdf(d1)) / (2 * np.sqrt(t))
    return primer_termino + r * precio


def Vega_B76(f0: float, r: float, t: float, d1: float) -> float:
    """
    Vega analítica de Black-76 (igual para call y put).

    Parámetros
    ----------
    f0    : Precio forward o futuro.
    r     : Tasa libre de riesgo anualizada.
    t     : Tiempo al vencimiento, en años.
    d1    : Parámetro d1 de Black-76.

    Regresa
    -------
    float : Vega de la opción.
    """
    df = np.exp(-r * t)
    return f0 * df * np.sqrt(t) * norm.pdf(d1)


def Rho_B76(t: float, precio: float) -> float:
    """
    Rho analítica de Black-76.

    En Black-76, el precio depende de la tasa libre de riesgo únicamente a
    través del factor de descuento, por lo que:

        Rho = -t * Precio

    Parámetros
    ----------
    t      : Tiempo al vencimiento, en años.
    precio : Precio de la opción (Black-76), call o put.

    Regresa
    -------
    float : Rho de la opción.
    """
    return -t * precio


# =================== LETRAS GRIEGAS PARA OPCIONES AMERICANAS CON DIVIDENDOS DISCRETOS ===============================
# VERSIÓN ÁRBOL BINOMIAL / APROXIMACIÓN CONTINUA (CRR)
"""
Igual que en el caso de Black-Scholes, para el árbol binomial no existen fórmulas
analíticas para las griegas, así que se calculan por aproximación de diferencias finitas.

Todas estas funciones llaman a `_precio_por_metodo` (definida en precios.py), que
decide internamente si el precio debe salir de la aproximación continua (Black,
solo para calls) o del árbol binomial CRR (para puts, o cuando tiempo='d').
"""


def Delta_A_div(s0:float, k:float, t:float, r:float, sigma:float, dividendos:ArregloComo, t_dividendos:ArregloComo, tipo='call', tiempo='c', n=200):
    """
    Delta de una opción americana con dividendos discretos, por diferencias
    finitas centrales sobre s0.

    Parámetros
    ----------
    s0            : Precio spot del subyacente.
    k             : Precio de ejercicio (strike).
    t             : Tiempo al vencimiento, en años.
    r             : Tasa libre de riesgo anualizada.
    sigma         : Volatilidad anualizada.
    dividendos    : Dividendos discretos.
    t_dividendos  : Fechas de pago de dividendos.
    tipo          : 'call' o 'put'.
    tiempo        : Método de valoración ('c' o 'd').
    n             : Número de pasos del árbol binomial.

    Regresa
    -------
    float : Delta aproximada.
    """
    h = max(0.01 * s0, 1e-5)
    p_mas = _precio_por_metodo(s0 + h, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p_menos = _precio_por_metodo(s0 - h, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    return (p_mas - p_menos) / (2 * h)


def Gamma_A_div(s0:float, k:float, t:float, r:float, sigma:float, dividendos:ArregloComo, t_dividendos:ArregloComo, tipo='call', tiempo='c', n=200):
    """
    Gamma de una opción americana con dividendos discretos, por diferencias
    finitas centrales de segundo orden sobre s0.

    Parámetros
    ----------
    s0            : Precio spot del subyacente.
    k             : Precio de ejercicio (strike).
    t             : Tiempo al vencimiento, en años.
    r             : Tasa libre de riesgo anualizada.
    sigma         : Volatilidad anualizada.
    dividendos    : Dividendos discretos.
    t_dividendos  : Fechas de pago de dividendos.
    tipo          : 'call' o 'put'.
    tiempo        : Método de valoración ('c' o 'd').
    n             : Número de pasos del árbol binomial.

    Regresa
    -------
    float : Gamma aproximada.
    """
    h = max(0.01 * s0, 1e-5)
    p_mas = _precio_por_metodo(s0 + h, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p = _precio_por_metodo(s0, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p_menos = _precio_por_metodo(s0 - h, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    return (p_mas - 2 * p + p_menos) / (h ** 2)


def Vega_A_div(s0:float, k:float, t:float, r:float, sigma:float, dividendos:ArregloComo, t_dividendos:ArregloComo, tipo='call', tiempo='c', n=200):
    """
    Vega de una opción americana con dividendos discretos, por diferencias
    finitas centrales sobre sigma.

    NOTA: perturbar sigma implica recalibrar u,d del árbol en cada
    evaluación (porque u,d dependen de sigma); esto ya ocurre
    automáticamente dentro de `_precio_por_metodo` -> `_resolver_u_d`.

    Parámetros
    ----------
    s0            : Precio spot del subyacente.
    k             : Precio de ejercicio (strike).
    t             : Tiempo al vencimiento, en años.
    r             : Tasa libre de riesgo anualizada.
    sigma         : Volatilidad anualizada.
    dividendos    : Dividendos discretos.
    t_dividendos  : Fechas de pago de dividendos.
    tipo          : 'call' o 'put'.
    tiempo        : Método de valoración ('c' o 'd').
    n             : Número de pasos del árbol binomial.

    Regresa
    -------
    float : Vega aproximada.
    """
    h = max(0.01 * sigma, 1e-8)
    p_mas = _precio_por_metodo(s0, k, t, r, sigma + h, dividendos, t_dividendos, tipo, tiempo, n)
    p_menos = _precio_por_metodo(s0, k, t, r, sigma - h, dividendos, t_dividendos, tipo, tiempo, n)
    return (p_mas - p_menos) / (2 * h)


def Rho_A_div(s0:float, k:float, t:float, r:float, sigma:float, dividendos:ArregloComo, t_dividendos:ArregloComo, tipo='call', tiempo='c', n=200):
    """
    Rho de una opción americana con dividendos discretos, por diferencias
    finitas centrales sobre r.

    Parámetros
    ----------
    s0            : Precio spot del subyacente.
    k             : Precio de ejercicio (strike).
    t             : Tiempo al vencimiento, en años.
    r             : Tasa libre de riesgo anualizada.
    sigma         : Volatilidad anualizada.
    dividendos    : Dividendos discretos.
    t_dividendos  : Fechas de pago de dividendos.
    tipo          : 'call' o 'put'.
    tiempo        : Método de valoración ('c' o 'd').
    n             : Número de pasos del árbol binomial.

    Regresa
    -------
    float : Rho aproximada.
    """
    h = max(0.0001 * abs(r), 1e-8)
    p_mas = _precio_por_metodo(s0, k, t, r + h, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p_menos = _precio_por_metodo(s0, k, t, r - h, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    return (p_mas - p_menos) / (2 * h)


def Theta_A_div(s0:float, k:float, t:float, r:float, sigma:float, dividendos:ArregloComo, t_dividendos:ArregloComo, tipo='call', tiempo='c', n=200):
    """
    Theta de una opción americana con dividendos discretos, por diferencia
    finita adelantada sobre t (comparando "hoy" contra "un día hábil
    después").

    Importante
    ----------
    Utiliza una convención de 1/252 (días hábiles), distinta de la empleada
    por `Theta_Diaria`.

    Parámetros
    ----------
    s0            : Precio spot del subyacente.
    k             : Precio de ejercicio (strike).
    t             : Tiempo al vencimiento, en años.
    r             : Tasa libre de riesgo anualizada.
    sigma         : Volatilidad anualizada.
    dividendos    : Dividendos discretos.
    t_dividendos  : Fechas de pago de dividendos.
    tipo          : 'call' o 'put'.
    tiempo        : Método de valoración ('c' o 'd').
    n             : Número de pasos del árbol binomial.

    Regresa
    -------
    (theta_anualizada, theta_diaria) : tuple[float, float]
        (np.nan, np.nan) si el tiempo restante es menor o igual a un día.
    """
    dt_theta = 1 / 252
    if t <= dt_theta:
        return np.nan, np.nan

    p_hoy = _precio_por_metodo(s0, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p_manana = _precio_por_metodo(
        s0, k, t - dt_theta, r, sigma, dividendos,
        [max(td - dt_theta, 0) for td in t_dividendos], tipo, tiempo, n
    )

    theta_anualizada = (p_manana - p_hoy) / dt_theta
    theta_diaria = p_manana - p_hoy
    return theta_anualizada, theta_diaria
