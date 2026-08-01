"""
Librería de finanzas Cuantitativas.
"""
from .opciones import (
    OpcionEuropea,
    OpcionEuropeaDiv,
    OpcionAmericanaDiv,
    OpcionFuturos,
    OpcionForex
)

from .riesgo import (
    VaR_parametrico,
    VaR_historico,
    Vol_portafolio
)

from .volatilidad import (
    sonrisa_volatilidad,
    estimacion_sigma_Newton,
    estimacion_sigma_tangente,
    estimacion_sigma_biseccion
)

__version__ = "0.1.0"
__author__ = "Jerson y Comunidad de ESFM"

