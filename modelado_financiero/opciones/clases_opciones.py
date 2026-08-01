"""
opciones.py
===========
Clases orientadas a objetos para el cálculo de precios y letras griegas
de distintas familias de opciones:

- OpcionEuropea       : Black-Scholes estándar, sin dividendos.
- OpcionEuropeaDiv    : Black-Scholes con ajuste de S0 por dividendos discretos.
- OpcionForex         : Garman-Kohlhagen (opciones sobre divisas).
- OpcionFuturos       : Black-76 (opciones sobre futuros/forwards).
- OpcionAmericanaDiv  : Americanas con dividendos discretos, vía
                        aproximación de Black (solo call, continuo) o
                        árbol binomial CRR (put, o modo discreto).

Convenciones de diseño usadas en todo el módulo:
- Orden estándar de parámetros: s0, k, t, r, sigma, [parámetros
  específicos de cada clase] (en OpcionFuturos se usa f0 en vez de s0
  por ser conceptualmente un precio forward, no spot; en OpcionForex se
  usan rd, rf en vez de una sola r).
- El precio de la opción se obtiene siempre con el método `.precio(...)`
  (nunca `.precio_bs`).
- d1 y d2 nunca se piden como argumento en las letras griegas: cada
  clase los calcula internamente a través de la property `.d1d2`.
"""

from opciones.precios import (ArregloComo, MatrizComo, Black_Scholes_Fx, Black_76, Aproximacion_Black,
                              ArbolBinomial_crr, _resolver_u_d, graficar_arbol_bin, Black_Scholes, d1_d2,
                              ajuste_s0, Fx_d1_d2, _precio_por_metodo)
from opciones.griegas import (Delta_c, Theta, Gamma, Vega, Rho, Theta_Diaria, Delta_B76, Theta_B76, Gamma_B76, Vega_B76,
                              Rho_B76, Delta_A_div, Gamma_A_div, Theta_A_div, Vega_A_div, Rho_A_div)

import numpy as np


#=====================================================================================================
#     CLASE DE OPCIONES EUROPEAS
#=====================================================================================================

class OpcionEuropea():
    """
    Opción europea (call/put) bajo el modelo de Black-Scholes estándar,
    sin dividendos.

    Parámetros
    ----------
    s0    : Precio spot del subyacente.
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    r     : Tasa libre de riesgo anualizada.
    sigma : volatilidad anualizada del subyacente.
    """

    def __init__(self, s0: float = None, k: float = None, t: float = None, r: float = None, sigma: float = None):
        self.s0 = s0
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma

    @property
    def d1d2(self):
        """Calcula (d1, d2) de Black-Scholes a partir de los atributos de la instancia."""
        return d1_d2(self.s0, self.k, self.r, self.t, self.sigma)

    def precio(self, tipo='Call'):
        """
        Precio de la opción europea (Black-Scholes).

        Parámetros
        ----------
        tipo : 'call' o 'put'.

        Regresa
        -------
        float : precio de la opción.
        """
        return Black_Scholes(self.s0, self.k, self.t, self.r, self.sigma, tipo)

    def delta(self, tipo='Call'):
        """
        Delta de la opción (Black-Scholes estándar).

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        return Delta_c(self.s0, self.k, self.t, self.r, self.sigma, tipo)

    def gamma(self):
        """Gamma de la opción (igual para call y put)."""
        d1, d2 = self.d1d2
        return Gamma(self.s0, self.sigma, self.t, d1)

    def theta(self, tipo='Call'):
        """
        Theta de la opción.

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        d1, d2 = self.d1d2
        return Theta(self.s0, self.k, self.t, self.sigma, self.r, d1, d2, tipo)

    def vega(self):
        """Vega de la opción (igual para call y put)."""
        d1, d2 = self.d1d2
        return Vega(self.s0, self.t, self.sigma, d1)

    def rho(self, tipo='Call'):
        """
        Rho de la opción.

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        d1, d2 = self.d1d2
        return Rho(self.k, self.t, self.r, d2, tipo)


#=====================================================================================================
#     CLASE DE OPCIONES EUROPEAS CON DIVIDENDOS
#=====================================================================================================

class OpcionEuropeaDiv():
    """
    Opción europea (call/put) bajo Black-Scholes, ajustando el precio
    spot por el valor presente de dividendos discretos conocidos.

    Parámetros
    ----------
    s0           : Precio spot del subyacente (sin ajustar).
    k            : Precio de ejercicio (strike).
    t            : Tiempo al vencimiento, en años.
    r            : Tasa libre de riesgo anualizada.
    sigma        : volatilidad anualizada del subyacente.
    dividendos   : Montos de los dividendos discretos.
    t_dividendos : Tiempos (en años) de cada dividendo.
    """

    def __init__(self, s0: float, k: float, t: float, r: float, sigma: float,
                 dividendos: ArregloComo, t_dividendos: ArregloComo):
        self.s0 = s0
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        self.dividendos = dividendos
        self.t_dividendos = t_dividendos

    @property
    def d1d2(self):
        """Calcula (d1, d2) de Black-Scholes usando s0 SIN ajustar por dividendos."""
        return d1_d2(self.s0, self.k, self.r, self.t, self.sigma)

    @property
    def s0_ajustado(self):
        """Precio spot ajustado (s0 menos el valor presente de los dividendos)."""
        return ajuste_s0(self.s0, self.r, self.t, self.dividendos, self.t_dividendos)

    def precio(self, tipo='call'):
        """
        Precio de la opción europea con dividendos (Black-Scholes con S0 ajustado).

        Parámetros
        ----------
        tipo : 'call' o 'put'.

        Regresa
        -------
        float : precio de la opción.
        """
        s0 = self.s0_ajustado
        return Black_Scholes(s0, self.k, self.t, self.r, self.sigma, tipo)

    # --------------------LETRAS GRIEGAS----------------------
    def delta(self, tipo='call'):
        """
        Delta de la opción, usando s0 ajustado por dividendos.

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        return Delta_c(self.s0_ajustado, self.k, self.t, self.r, self.sigma, tipo)

    def gamma(self):
        """Gamma de la opción, usando s0 ajustado por dividendos."""
        d1, d2 = self.d1d2
        return Gamma(self.s0_ajustado, self.sigma, self.t, d1)

    def theta(self, tipo='call'):
        """
        Theta de la opción con dividendos, calculada por diferencias
        finitas (Theta_Diaria).

        IMPORTANTE: se pasa self.s0 SIN ajustar, ya que Theta_Diaria
        realiza internamente el ajuste de valor presente de los
        dividendos tanto para "hoy" como para "mañana". Pasar
        self.s0_ajustado aquí restaría el valor presente de los
        dividendos dos veces.

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        return Theta_Diaria(self.s0, self.k, self.t, self.sigma, self.r,
                             self.dividendos, self.t_dividendos, tipo)

    def vega(self):
        """Vega de la opción, usando s0 ajustado por dividendos."""
        d1, d2 = self.d1d2
        return Vega(self.s0_ajustado, self.t, self.sigma, d1)

    def rho(self, tipo='call'):
        """
        Rho de la opción, usando s0 ajustado por dividendos.

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        d1, d2 = self.d1d2
        return Rho(self.k, self.t, self.r, d2, tipo)


#=====================================================================================================
#     CLASE DE OPCIONES CON DIVISAS
#=====================================================================================================

class OpcionForex():
    """
    Opción europea (call/put) sobre divisas bajo el modelo de
    Garman-Kohlhagen.

    Parámetros
    ----------
    s0    : Tipo de cambio spot actual.
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    sigma : volatilidad anualizada del tipo de cambio.
    rd    : Tasa libre de riesgo doméstica anualizada.
    rf    : Tasa libre de riesgo extranjera anualizada.
    tipo  : 'call' o 'put'.
    """

    def __init__(self, s0: float, k: float, t: float, sigma: float, rd: float, rf: float):
        self.s0 = s0
        self.k = k
        self.t = t
        self.sigma = sigma
        self.rd = rd
        self.rf = rf



    def precio(self, tipo = 'call'):
        """Precio de la opción (Garman-Kohlhagen)."""
        precio = Black_Scholes_Fx(self.s0, self.k, self.t, self.sigma, self.rd, self.rf, tipo)
        return precio

    @property
    def d1d2(self):
        """Calcula (d1, d2) de Garman-Kohlhagen a partir de los atributos de la instancia."""
        return Fx_d1_d2(self.s0, self.k, self.t, self.sigma, self.rd, self.rf)

    # _____________________ LETRAS GRIEGAS _________________________________
    """
    PENDIENTE: las funciones Delta_c, Gamma, Theta y Vega del módulo
    griegas.py fueron derivadas para Black-Scholes estándar (una sola
    tasa r), no para Garman-Kohlhagen (dos tasas rd, rf). Reutilizarlas
    aquí tal cual daría resultados matemáticamente incorrectos (por
    ejemplo, la Delta de Garman-Kohlhagen lleva un factor exp(-rf*t) que
    no aparece en Black-Scholes estándar). Por eso, en vez de "parchar"
    con una tasa que no aplica, estas griegas quedan señaladas como
    pendientes hasta derivar/implementar las fórmulas correctas de
    Garman-Kohlhagen.
    """

    def delta(self):
        """Pendiente: requiere la fórmula de Delta de Garman-Kohlhagen (usa rd y rf, no una sola r)."""
        raise NotImplementedError(
            "Delta para OpcionForex está pendiente: requiere la fórmula de "
            "Garman-Kohlhagen (con factor exp(-rf*t)), no la de Black-Scholes estándar."
        )

    def gamma(self):
        """Pendiente: requiere la fórmula de Gamma de Garman-Kohlhagen (usa rd y rf, no una sola r)."""
        raise NotImplementedError(
            "Gamma para OpcionForex está pendiente: requiere la fórmula de "
            "Garman-Kohlhagen, no la de Black-Scholes estándar."
        )

    def theta(self):
        """Pendiente: requiere la fórmula de Theta de Garman-Kohlhagen (usa rd y rf, no una sola r)."""
        raise NotImplementedError(
            "Theta para OpcionForex está pendiente: requiere la fórmula de "
            "Garman-Kohlhagen, no la de Black-Scholes estándar."
        )

    def vega(self):
        """Pendiente: requiere la fórmula de Vega de Garman-Kohlhagen (usa rd y rf, no una sola r)."""
        raise NotImplementedError(
            "Vega para OpcionForex está pendiente: requiere la fórmula de "
            "Garman-Kohlhagen, no la de Black-Scholes estándar."
        )

    def rho(self):
        """Pendiente: requiere la(s) fórmula(s) de Rho de Garman-Kohlhagen (hay Rho doméstica y Rho extranjera)."""
        raise NotImplementedError(
            "Rho para OpcionForex está pendiente: en Garman-Kohlhagen existen "
            "dos Rho distintas (respecto a rd y respecto a rf), no una sola "
            "como en Black-Scholes o Black-76."
        )


#================================================================
# CLASE DE OPCIONES CON FUTUROS
#================================================================

class OpcionFuturos():
    """
    Opción europea (call/put) sobre un futuro/forward bajo el modelo de
    Black-76.

    Parámetros
    ----------
    f0    : Precio del futuro/forward actual (se usa f0 y no s0, ya que
            conceptualmente es un precio forward, no un precio spot).
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    r     : Tasa libre de riesgo anualizada (para descuento).
    sigma : volatilidad anualizada del futuro.
    """

    def __init__(self, f0: float, k: float, t: float, r: float, sigma: float):
        self.f0 = f0
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma

    def precio(self, tipo='call'):
        """
        Precio de la opción sobre futuros (Black-76).

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        return Black_76(self.f0, self.k, self.t, self.r, self.sigma, tipo)

    @property
    def d1d2(self):
        """Calcula (d1, d2) de Black-76 a partir de los atributos de la instancia."""
        d1 = (np.log(self.f0 / self.k) + 0.5 * self.sigma ** 2 * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        return d1, d2

    # _____________________ LETRAS GRIEGAS _________________________________

    def delta(self, tipo='call'):
        """
        Delta de la opción (Black-76).

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        d1, d2 = self.d1d2
        return Delta_B76(self.r, self.t, d1, tipo)

    def gamma(self):
        """Gamma de la opción (Black-76, igual para call y put)."""
        d1, d2 = self.d1d2
        return Gamma_B76(self.f0, self.r, self.t, self.sigma, d1)

    def theta(self, tipo='call'):
        """
        Theta de la opción (Black-76). Theta_B76 recalcula el precio
        internamente a partir de (f0, k, t, r, sigma, tipo), evitando
        depender de un precio calculado aparte que pudiera no
        corresponder al 'tipo' indicado.

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        d1, d2 = self.d1d2
        return Theta_B76(self.f0, self.k, self.t, self.r, self.sigma, d1, tipo)

    def vega(self):
        """Vega de la opción (Black-76, igual para call y put)."""
        d1, d2 = self.d1d2
        return Vega_B76(self.f0, self.r, self.t, d1)

    def rho(self, tipo='call'):
        """
        Rho de la opción (Black-76). En este modelo, Rho = -t * precio
        (misma fórmula para call y put, ya que d1/d2 no dependen de r).

        Parámetros
        ----------
        tipo : 'call' o 'put'.
        """
        precio = self.precio(tipo)
        return Rho_B76(self.t, precio)


#================================================
# CLASE DE OPCIONES AMERICANAS CON DIVIDENDOS
#================================================

class OpcionAmericanaDiv:
    """
    Opción americana (call/put) con dividendos discretos conocidos.

    El precio y las letras griegas se calculan mediante `_precio_por_metodo`
    (definida en precios.py), que decide internamente el método:
    - Aproximación de Black (fórmula cerrada aproximada) si tiempo='c' y
      tipo='call'.
    - Árbol binomial CRR en cualquier otro caso (put, o tiempo='d').

    Las letras griegas no tienen fórmula cerrada en ninguno de los dos
    métodos, por lo que se aproximan por diferencias finitas.

    Parámetros
    ----------
    s0           : Precio spot del subyacente.
    k            : Precio de ejercicio (strike).
    t            : Tiempo al vencimiento, en años.
    r            : Tasa libre de riesgo anualizada.
    sigma        : volatilidad anualizada del subyacente.
    dividendos   : Montos de los dividendos discretos.
    t_dividendos : Tiempos (en años) de cada dividendo.
    n            : Número de pasos del árbol binomial (por defecto 200).
    """

    def __init__(self, s0, k, t, r, sigma, dividendos, t_dividendos, n=200):
        self.s0 = s0
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        self.dividendos = dividendos
        self.t_dividendos = t_dividendos
        self.n = n

    def precio(self, tipo='Call', tiempo='c', n=None):
        """
        Precio de la opción americana con dividendos.

        Parámetros
        ----------
        tipo   : 'call' o 'put'.
        tiempo : 'c' (continuo, usa Black aproximado si tipo='call') o
                 'd' (discreto, siempre usa el árbol binomial).
        n      : número de pasos del árbol (si no se especifica, usa self.n).
        """
        n = self.n if n is None else n
        return _precio_por_metodo(self.s0, self.k, self.t, self.r, self.sigma,
                                   self.dividendos, self.t_dividendos, tipo, tiempo, n)

    def delta(self, tipo='Call', tiempo='c'):
        """Delta aproximada por diferencias finitas sobre s0."""
        return Delta_A_div(self.s0, self.k, self.t, self.r, self.sigma,
                            self.dividendos, self.t_dividendos, tipo, tiempo, self.n)

    def gamma(self, tipo='Call', tiempo='c'):
        """Gamma aproximada por diferencias finitas sobre s0."""
        return Gamma_A_div(self.s0, self.k, self.t, self.r, self.sigma,
                            self.dividendos, self.t_dividendos, tipo, tiempo, self.n)

    def theta(self, tipo='Call', tiempo='c'):
        """Theta aproximada por diferencias finitas sobre t (anualizada y diaria)."""
        return Theta_A_div(self.s0, self.k, self.t, self.r, self.sigma,
                            self.dividendos, self.t_dividendos, tipo, tiempo, self.n)

    def vega(self, tipo='Call', tiempo='c'):
        """Vega aproximada por diferencias finitas sobre sigma."""
        return Vega_A_div(self.s0, self.k, self.t, self.r, self.sigma,
                           self.dividendos, self.t_dividendos, tipo, tiempo, self.n)

    def rho(self, tipo='Call', tiempo='c'):
        """Rho aproximada por diferencias finitas sobre r."""
        return Rho_A_div(self.s0, self.k, self.t, self.r, self.sigma,
                          self.dividendos, self.t_dividendos, tipo, tiempo, self.n)
