"""
riesgo.py
=========
Módulo de funciones de gestión de riesgo de portafolio: volatilidad de
portafolio y Value at Risk (VaR) por método paramétrico, histórico y
Montecarlo.
"""

import numpy as np
from scipy.stats import norm
from typing import Sequence, TypeAlias

ArregloComo: TypeAlias = Sequence[float]
MatrizComo: TypeAlias = Sequence[Sequence[float]]


def Vol_portafolio(pesos: ArregloComo, covarianzas: MatrizComo):
    """
    Calcula la volatilidad (desviación estándar) de un portafolio dado
    un vector de pesos y una matriz de covarianzas.

    Parámetros
    ----------
    pesos       : Pesos de cada activo en el portafolio.
    covarianzas : Matriz de covarianzas entre los activos.

    Regresa
    -------
    float : volatilidad del portafolio (sigma_p).
    """
    # Se convierte a np.ndarray por si el usuario pasa listas/tuplas en
    # vez de arreglos de numpy, ya que se requieren operaciones matriciales.
    pesos = np.asarray(pesos, dtype=float)
    covarianzas = np.asarray(covarianzas, dtype=float)
    sigma_p = np.sqrt(pesos.T @ covarianzas @ pesos)
    return sigma_p


def VaR_parametrico(portafolio_valor: float, mu: float, sigma: float, nivel_confianza=0.95):
    """
    Calcula el VaR paramétrico asumiendo distribución normal de los
    retornos del portafolio.

    Parámetros
    ----------
    portafolio_valor : Valor total del portafolio.
    mu               : Retorno esperado del portafolio.
    sigma            : Volatilidad del portafolio.
    nivel_confianza  : Nivel de confianza (por defecto 0.95).

    Regresa
    -------
    float : VaR estimado (no negativo).
    """
    z = norm.ppf(nivel_confianza)
    var = (z*sigma - mu)*portafolio_valor
    return max(var, 0)


def VaR_historico(retornos: ArregloComo, portafolio_valor: float, nivel_confianza=0.90):
    """
    Calcula el VaR histórico buscando el percentil de los retornos reales
    observados.

    Parámetros
    ----------
    retornos         : Serie histórica de retornos del portafolio.
    portafolio_valor : Valor total del portafolio.
    nivel_confianza  : Nivel de confianza (por defecto 0.90).

    Regresa
    -------
    float : VaR estimado (no negativo).
    """
    # Calculamos el percentil (ej. para 90%, buscamos el 10% de peor rendimiento)
    percentil = (1 - nivel_confianza) * 100
    retorno_percentil = np.percentile(retornos, percentil)
    var = -retorno_percentil * portafolio_valor
    return max(var, 0)


def VaR_montecarlo(portafolio_valor: float, mu: float, sigma: float, t: float,
                    n_simulaciones=10000, nivel_confianza=0.95, seed=None):
    """
    Calcula el VaR mediante simulación Montecarlo, asumiendo que el
    portafolio sigue un movimiento browniano geométrico (GBM).

    Parámetros
    ----------
    portafolio_valor : Valor total del portafolio.
    mu               : Retorno esperado anualizado.
    sigma            : Volatilidad anualizada.
    t                : Horizonte de tiempo, en años.
    n_simulaciones   : Número de trayectorias simuladas.
    nivel_confianza  : Nivel de confianza (por defecto 0.95).
    seed             : Semilla para reproducibilidad (opcional).

    Regresa
    -------
    float : VaR estimado (no negativo).
    """
    if seed is not None:
        np.random.seed(seed)

    z = np.random.standard_normal(n_simulaciones)

    # Retornos simulados bajo GBM
    retornos_simulados = np.exp(
        (mu - 0.5 * sigma**2) * t +
        sigma * np.sqrt(t) * z
    ) - 1

    # Cola izquierda de los retornos
    retorno_critico = np.percentile(
        retornos_simulados,
        (1 - nivel_confianza) * 100
    )

    # Convertir a pérdida monetaria positiva
    var = -retorno_critico * portafolio_valor

    return max(var, 0)
