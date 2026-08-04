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
    Calcula la volatilidad (desviación estándar) de un portafolio a partir
    de sus pesos y la matriz de covarianzas.

    Parámetros
    ----------
    pesos       : Pesos de los activos.
    covarianzas : Matriz de covarianzas de los activos.

    Regresa
    -------
    float : Volatilidad del portafolio..
    """
    # Se convierte a np.ndarray por si el usuario pasa listas/tuplas en
    # vez de arreglos de numpy, ya que se requieren operaciones matriciales.
    pesos = np.asarray(pesos, dtype=float)
    covarianzas = np.asarray(covarianzas, dtype=float)
    sigma_p = np.sqrt(pesos.T @ covarianzas @ pesos)
    return sigma_p



def Vol_portafolio(pesos: ArregloComo, covarianzas: MatrizComo):
    """
    Calcula la volatilidad (desviación estándar) de un portafolio a partir
    de sus pesos y la matriz de covarianzas.

    Parámetros
    ----------
    pesos       : Pesos de los activos.
    covarianzas : Matriz de covarianzas de los activos.

    Regresa
    -------
    float : Volatilidad del portafolio.

    Lanza
    -----
    ValueError : Si la matriz de covarianzas no es válida (no cuadrada,
    no simétrica, dimensiones incompatibles o varianza negativa).
    """
    # Se convierte a np.ndarray por si el usuario pasa listas/tuplas en
    # vez de arreglos de numpy, ya que se requieren operaciones matriciales.
    pesos = np.asarray(pesos, dtype=float)
    covarianzas = np.asarray(covarianzas, dtype=float)

    if covarianzas.ndim != 2 or covarianzas.shape[0] != covarianzas.shape[1]:
        raise ValueError("'covarianzas' debe ser una matriz cuadrada")

    if pesos.shape[0] != covarianzas.shape[0]:
        raise ValueError(
            f"'pesos' ({pesos.shape[0]} elementos) y 'covarianzas' "
            f"({covarianzas.shape[0]}x{covarianzas.shape[0]}) tienen "
            "dimensiones incompatibles"
        )

    if not np.allclose(covarianzas, covarianzas.T):
        raise ValueError("'covarianzas' debe ser una matriz simétrica")

    varianza_p = pesos.T @ covarianzas @ pesos

    if varianza_p < 0:
        raise ValueError(
            "La varianza del portafolio resultó negativa "
            f"({varianza_p:.6f}); 'covarianzas' no es semidefinida "
            "positiva. Revisa los datos de entrada."
        )

    return np.sqrt(varianza_p)


def VaR_parametrico(portafolio_valor: float, mu: float, sigma: float,
                     nivel_confianza=0.95, distribucion='normal'):
    """
    Calcula el VaR paramétrico de un portafolio.

    Importante
    ----------
    Esta función no recibe un horizonte temporal explícito. Los parámetros
    'mu' y 'sigma' deben estar expresados en el mismo horizonte deseado para
    el VaR (diario, semanal, mensual, etc.).

    Distribuciones disponibles
    --------------------------
    - 'normal'    : asume retornos simples normalmente distribuidos.
    - 'lognormal' : asume un movimiento browniano geométrico (GBM),
                    consistente con VaR_montecarlo.

    La diferencia entre ambas aproximaciones aumenta conforme crecen la
    volatilidad y el horizonte temporal.

    Parámetros
    ----------
    portafolio_valor : Valor total del portafolio.
    mu               : Retorno esperado en el horizonte considerado.
    sigma            : Volatilidad en el mismo horizonte que 'mu'.
    nivel_confianza  : Nivel de confianza (default 0.95).
    distribucion     : 'normal' o 'lognormal'.

    Regresa
    -------
    float : VaR estimado (no negativo).

    Lanza
    -----
    ValueError : Si los parámetros son inválidos.
    """
    if not (0 < nivel_confianza < 1):
        raise ValueError("'nivel_confianza' debe estar en (0, 1)")

    if sigma < 0:
        raise ValueError("'sigma' no puede ser negativa")

    distribucion = distribucion.lower()
    z = norm.ppf(nivel_confianza)

    if distribucion == 'normal':
        var = (z * sigma - mu) * portafolio_valor

    elif distribucion == 'lognormal':
        retorno_critico = np.exp(mu - 0.5 * sigma**2 - z * sigma) - 1
        var = -retorno_critico * portafolio_valor

    else:
        raise ValueError(
            f"'distribucion' debe ser 'normal' o 'lognormal', se recibió: {distribucion!r}"
        )

    return max(var, 0)


def VaR_historico(retornos: ArregloComo, portafolio_valor: float, nivel_confianza=0.90):
    """
    Calcula el VaR histórico a partir de la distribución empírica de los
    retornos observados.

    Nota
    ----
    La media ya está incorporada implícitamente en la serie de retornos, por
    lo que no se requiere un parámetro 'mu'.

    Parámetros
    ----------
    retornos         : Serie histórica de retornos.
    portafolio_valor : Valor total del portafolio.
    nivel_confianza  : Nivel de confianza (default 0.90).

    Regresa
    -------
    float : VaR estimado (no negativo).

    Lanza
    -----
    ValueError : Si 'nivel_confianza' no está en (0, 1) o si la serie de
    retornos está vacía.
    """
    if not (0 < nivel_confianza < 1):
        raise ValueError("'nivel_confianza' debe estar en (0, 1)")

    retornos = np.asarray(retornos, dtype=float)
    if retornos.size == 0:
        raise ValueError("'retornos' no puede estar vacío")

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

    Nota
    ----
    Para un mismo horizonte 't', este método es directamente comparable con
    VaR_parametrico(..., distribucion='lognormal'), ya que ambos utilizan el
    mismo supuesto distribucional. Las diferencias observadas provienen del
    error de muestreo inherente a la simulación.

    Parámetros
    ----------
    portafolio_valor : Valor total del portafolio.
    mu               : Retorno esperado anualizado.
    sigma            : Volatilidad anualizada.
    t                : Horizonte temporal en años.
    n_simulaciones   : Número de simulaciones.
    nivel_confianza  : Nivel de confianza (default 0.95).
    seed             : Semilla para reproducibilidad. Se utiliza un
                       generador local (np.random.default_rng), por lo que
                       no modifica el estado aleatorio global de NumPy.

    Regresa
    -------
    float : VaR estimado (no negativo).

    Lanza
    -----
    ValueError : Si los parámetros de entrada son inválidos.
    """
    if not (0 < nivel_confianza < 1):
        raise ValueError("'nivel_confianza' debe estar en (0, 1)")

    if sigma < 0:
        raise ValueError("'sigma' no puede ser negativa")

    if t <= 0:
        raise ValueError("'t' debe ser positivo")

    if n_simulaciones <= 0 or not isinstance(n_simulaciones, (int, np.integer)):
        raise ValueError(f"'n_simulaciones' debe ser un entero positivo, se recibió: {n_simulaciones}")

    # Generador local: no muta el estado aleatorio global de numpy,
    # así que no interfiere con otras simulaciones del programa.
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_simulaciones)

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