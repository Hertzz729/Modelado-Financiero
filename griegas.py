"""
griegas.py
==========
Módulo de letras griegas para las distintas familias de opciones del
proyecto:

- Griegas analíticas de Black-Scholes estándar (sin dividendos).
- Theta universal por diferencias finitas para opciones europeas con o
  sin dividendos discretos.
- Griegas analíticas de Black-76 (opciones sobre futuros).
- Griegas de opciones americanas con dividendos discretos, calculadas
  por diferencias finitas sobre `_precio_por_metodo` (que a su vez decide
  internamente si usar Black aproximado o el árbol binomial CRR).

Nota general: a diferencia de las griegas analíticas de Black-Scholes,
las griegas de opciones americanas con dividendos NO tienen fórmula
cerrada, por lo que se aproximan numéricamente perturbando cada
parámetro una cantidad pequeña 'h' y evaluando la diferencia en el
precio resultante.
"""

from precios import Aproximacion_Black, ArbolBinomial_crr, _resolver_u_d, _precio_por_metodo

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from decimal import Decimal
from typing import Sequence, TypeAlias

ArregloComo: TypeAlias = Sequence[float]
MatrizComo: TypeAlias = Sequence[Sequence[float]]


# ================Letras griegas para Opciones (Black-Scholes estándar)======================

def Delta_c(s0: float, k: float, t: float, r: float, sigma: float, tipo = 'call'):
    """
    Delta de un CALL europeo bajo Black-Scholes estándar (sin dividendos).
    Para la Delta de un PUT, usar la identidad: Delta_put = Delta_call - 1.

    Regresa
    -------
    float : Delta del call, en [0, 1].
    """
    d1 = (np.log(s0/k) + (r+sigma**2/2)*t)/(sigma*np.sqrt(t))
    return norm.cdf(d1)


def Theta(s0: float, k: float, t: float, sigma: float, r: float, d1: float, d2: float, tipo="call"):
    """
    Theta analítica de Black-Scholes estándar (sin dividendos), anualizada.

    Regresa
    -------
    float : Theta (tasa de cambio del precio respecto al tiempo).
    """
    if tipo == "call":
        return - (s0 * 1/np.sqrt(2 * np.pi) * sigma * np.exp(-d1**2/2)) / (2*np.sqrt(t)) - r*k*np.exp(-r*t)*norm.cdf(d2)
    else:
        return - (s0 * 1/np.sqrt(2 * np.pi) * sigma * np.exp(-d1**2/2)) / (2*np.sqrt(t)) + r*k*np.exp(-r*t)*norm.cdf(-d2)


def Gamma(s0: float, sigma: float, t: float, d1: float):
    """
    Gamma analítica de Black-Scholes estándar (igual para call y put).

    Regresa
    -------
    float : Gamma (segunda derivada del precio respecto a s0).
    """
    return np.exp(-d1**2/2)/(np.sqrt(2 * np.pi) * s0 * sigma*np.sqrt(t))


def Vega(s0: float, t: float, sigma: float, d1: float):
    """
    Vega analítica de Black-Scholes estándar (igual para call y put).

    Regresa
    -------
    float : Vega (derivada del precio respecto a sigma).
    """
    return s0 * np.sqrt(t) * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * d1**2)


# Theta universal para dividendos y sin
def Theta_Diaria(s0: float, k: float, t: float, sigma: float, r: float,
                  dividendos: list = None, t_dividendos: list = None, tipo: str = "call") -> float:
    """
    Calcula la Theta diaria (pérdida/ganancia de valor por día transcurrido)
    mediante el método numérico de diferencias finitas.
    Funciona de forma exacta tanto para opciones CON como SIN dividendos.

    IMPORTANTE: 's0' debe pasarse SIN ajustar por dividendos; esta función
    ya realiza internamente el ajuste de valor presente de los dividendos
    tanto para "hoy" como para "mañana". Pasar un s0 ya ajustado
    provocaría restar el valor presente de los dividendos dos veces.

    Regresa
    -------
    float : Theta diaria (precio de mañana menos precio de hoy).
    """
    if dividendos is None:
        dividendos = []
    if t_dividendos is None:
        t_dividendos = []

    dt = 1 / 365.0  # Un día en términos anuales

    # --- 1. PRECIO HOY ---
    # Calculamos el PV de dividendos hoy
    pv_d_hoy = sum(
        d * np.exp(-r * t_d)
        for d, t_d in zip(dividendos, t_dividendos)
        if t > t_d > 0
    )
    s0_adj_hoy = s0 - pv_d_hoy

    # Calculamos Black-Scholes para hoy
    d1_hoy = (np.log(s0_adj_hoy / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2_hoy = d1_hoy - sigma * np.sqrt(t)

    if tipo.lower() == "call":
        precio_hoy = s0_adj_hoy * norm.cdf(d1_hoy) - k * np.exp(-r * t) * norm.cdf(d2_hoy)
    else:
        precio_hoy = k * np.exp(-r * t) * norm.cdf(-d2_hoy) - s0_adj_hoy * norm.cdf(-d1_hoy)

    # --- 2. PRECIO MAÑANA (Avanzamos 1 día) ---
    t_manana = t - dt

    # Restamos 1 día a la fecha de cobro de cada dividendo
    t_divs_manana = [t_d - dt for t_d in t_dividendos]

    pv_d_manana = sum(
        d * np.exp(-r * t_d)
        for d, t_d in zip(dividendos, t_divs_manana)
        if t_manana > t_d > 0
    )
    s0_adj_manana = s0 - pv_d_manana

    # Calculamos Black-Scholes para mañana
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

    Regresa
    -------
    float : Gamma de la opción.
    """
    df = np.exp(-r * t)
    return (df * norm.pdf(d1)) / (f0 * sigma * np.sqrt(t))


def Theta_B76(f0: float, r: float, t: float, sigma: float, d1: float, precio: float) -> float:
    """
    Theta analítica de Black-76.

    Parámetros
    ----------
    precio : precio de la opción (Black-76), necesario como parte de la
             fórmula de Theta.

    Regresa
    -------
    float : Theta de la opción.
    """
    df = np.exp(-r * t)
    primer_termino = -(f0 * sigma * df * norm.pdf(d1)) / (2 * np.sqrt(t))
    return primer_termino + r * precio


def Vega_B76(f0: float, r: float, t: float, d1: float) -> float:
    """
    Vega analítica de Black-76 (igual para call y put).

    Regresa
    -------
    float : Vega de la opción.
    """
    df = np.exp(-r * t)
    return f0 * df * np.sqrt(t) * norm.pdf(d1)


# =================== LETRAS GRIEGAS PARA OPCIONES AMERICANAS CON DIVIDENDOS DISCRETOS ===============================
# VERSIÓN ÁRBOL BINOMIAL / APROXIMACIÓN CONTINUA (CRR)
"""
Igual que en el caso de Black-Scholes, para el árbol binomial no existen fórmulas
analíticas para las griegas, así que se calculan por aproximación de diferencias finitas.

Todas estas funciones llaman a `_precio_por_metodo` (definida en precios.py), que
decide internamente si el precio debe salir de la aproximación continua (Black,
solo para calls) o del árbol binomial CRR (para puts, o cuando tiempo='d').
"""


def Delta_A_div(s0, k, t, r, sigma, dividendos, t_dividendos, tipo='Call', tiempo='c', n=200):
    """
    Delta de una opción americana con dividendos discretos, por diferencias
    finitas centrales sobre s0.

    Regresa
    -------
    float : Delta aproximada.
    """
    h = max(0.01 * s0, 1e-5)
    p_mas = _precio_por_metodo(s0 + h, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p_menos = _precio_por_metodo(s0 - h, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    return (p_mas - p_menos) / (2 * h)


def Gamma_A_div(s0, k, t, r, sigma, dividendos, t_dividendos, tipo='Call', tiempo='c', n=200):
    """
    Gamma de una opción americana con dividendos discretos, por diferencias
    finitas centrales de segundo orden sobre s0.

    Regresa
    -------
    float : Gamma aproximada.
    """
    h = 0.01 * s0
    p_mas = _precio_por_metodo(s0 + h, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p = _precio_por_metodo(s0, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p_menos = _precio_por_metodo(s0 - h, k, t, r, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    return (p_mas - 2 * p + p_menos) / (h ** 2)


def Vega_A_div(s0, k, t, r, sigma, dividendos, t_dividendos, tipo='Call', tiempo='c', n=200):
    """
    Vega de una opción americana con dividendos discretos, por diferencias
    finitas centrales sobre sigma.

    NOTA: perturbar sigma implica recalibrar u,d del árbol en cada
    evaluación (porque u,d dependen de sigma); esto ya ocurre
    automáticamente dentro de `_precio_por_metodo` -> `_resolver_u_d`.

    Regresa
    -------
    float : Vega aproximada.
    """
    h = max(0.01 * sigma, 1e-8)
    p_mas = _precio_por_metodo(s0, k, t, r, sigma + h, dividendos, t_dividendos, tipo, tiempo, n)
    p_menos = _precio_por_metodo(s0, k, t, r, sigma - h, dividendos, t_dividendos, tipo, tiempo, n)
    return (p_mas - p_menos) / (2 * h)


def Rho_A_div(s0, k, t, r, sigma, dividendos, t_dividendos, tipo='Call', tiempo='c', n=200):
    """
    Rho de una opción americana con dividendos discretos, por diferencias
    finitas centrales sobre r.

    Regresa
    -------
    float : Rho aproximada.
    """
    h = max(0.0001, 1e-8)
    p_mas = _precio_por_metodo(s0, k, t, r + h, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    p_menos = _precio_por_metodo(s0, k, t, r - h, sigma, dividendos, t_dividendos, tipo, tiempo, n)
    return (p_mas - p_menos) / (2 * h)


def Theta_A_div(s0, k, t, r, sigma, dividendos, t_dividendos, tipo='Call', tiempo='c', n=200):
    """
    Theta de una opción americana con dividendos discretos, por diferencia
    finita adelantada sobre t (comparando "hoy" contra "un día hábil
    después").

    NOTA: perturbar t implica recalibrar u,d del árbol en cada evaluación
    (porque dt = t/n depende de t); esto ya ocurre automáticamente dentro
    de `_precio_por_metodo` -> `_resolver_u_d`, evitando así una
    distorsión artificial de la volatilidad implícita del árbol.

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
