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

from .opciones.precios import  (
    Black_Scholes,
    d1_d2,
    ajuste_s0,

    Black_76,

    Black_Scholes_Fx,
    Fx_d1_d2,

    Aproximacion_Black,

    ArbolBinomial_crr,
    graficar_arbol_bin
)


from .opciones.griegas import (
    Delta,
    Gamma,
    Theta,
    Vega,
    Rho,

    Delta_A_div,
    Gamma_A_div,
    Theta_A_div,
    Vega_A_div,
    Rho_A_div,

    Delta_B76,
    Gamma_B76,
    Theta_B76,
    Vega_B76,
    Rho_B76,

)

from .riesgo import (
    VaR_parametrico,
    VaR_historico,
    VaR_montecarlo,
    Vol_portafolio,
)

from .volatilidad import (
    sonrisa_volatilidad,
    estimacion_sigma_Newton,
    estimacion_sigma_tangente,
    estimacion_sigma_biseccion
)

__version__ = "0.1.0"
__author__ = "Jerson y Comunidad de ESFM"

