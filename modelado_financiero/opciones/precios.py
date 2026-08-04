"""
precios.py
==========
Módulo central de cálculo de precios de opciones.

Contiene:
- Modelo de Black-Scholes (opciones europeas sin dividendos).
- Ajuste de S0 por dividendos discretos (valor presente de dividendos).
- Modelo Black-76 (opciones sobre futuros/forwards).
- Modelo Garman-Kohlhagen (opciones sobre divisas).
- Aproximación de Black para calls americanos con dividendos discretos.
- Árbol binomial CRR (Cox-Ross-Rubinstein) para opciones americanas y
  europeas, con o sin dividendos discretos.
- Función auxiliar `_precio_por_metodo`, usada por las clases y las
  letras griegas del árbol binomial para decidir de forma centralizada
  si el precio debe salir de la aproximación continua (Black) o del
  árbol binomial.
- Función de graficación didáctica del árbol binomial.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Sequence, TypeAlias
import matplotlib.patches as mpatches
import networkx as nx

ArregloComo: TypeAlias = Sequence[float]
MatrizComo: TypeAlias = Sequence[Sequence[float]]

# ==================Modelo black and scholes====================


def Black_Scholes(s0: float, k: float, t: float, r: float, sigma: float, tipo='call'):
    """
    Calcula el precio de una opción europea (call o put) usando la fórmula
    cerrada de Black-Scholes, sin dividendos.

    Parámetros
    ----------
    s0    : Precio spot actual del subyacente.
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    r     : Tasa libre de riesgo anualizada.
    sigma : volatilidad anualizada del subyacente.
    tipo  : 'call' o 'put'.

    Regresa
    -------
    float : precio de la opción.

    Lanza
    -----
    ValueError          : si tipo no es 'call' ni 'put'.
    """
    # Guarda
    if t <= 0 or sigma <=0 or s0 <=0:
        return max(s0-k,0) if tipo.lower() == 'call' else max(k-s0,0)

    # calculamos d1 y d2
    d1 = (np.log(s0/k) + (r+sigma**2/2)*t)/(sigma*np.sqrt(t))
    d2 = d1-sigma*np.sqrt(t)  # d2 támbien se puede calcular así

    # calculos de el precio de el contrato según el tipo
    if tipo.lower() == 'call':
        nor_d1 = norm.cdf(d1)
        nor_d2 = norm.cdf(d2)
        return s0 * nor_d1 - k * np.exp(-r*t) * nor_d2
    elif tipo.lower() == 'put':
        # usamos la propiedad de simetria de la normal para calcular N(-d1)= 1-N(d1) y N(-d2) = 1-N(d2)
        nor_d1 = 1-norm.cdf(d1)
        nor_d2 = 1-norm.cdf(d2)
        return k * nor_d2 * np.exp(-r*t) - s0 * nor_d1
    else:
        raise ValueError("tipo debe ser 'call' o 'put'")


def d1_d2(s0: float, k: float, r: float, t: float, sigma: float):
    """
    Calcula d1 y d2 del modelo de Black-Scholes estándar (sin dividendos).

    Regresa
    -------
    (d1, d2) : tuple[float, float]
    """
    if t <= 0 or sigma <= 0 or s0 <= 0:
        return np.nan, np.nan

    d1 = (np.log(s0/k) + (r+sigma**2/2)*t)/(sigma*np.sqrt(t))
    d2 = d1-sigma*np.sqrt(t)  # d2 támbien se puede calcular así
    return d1, d2

# ========================= AJUSTE S_0 (OPCIONES CON DIVIDENDOS) =============================


def ajuste_s0(s0: float, r: float, t: float, dividendos: ArregloComo, t_dividendos: ArregloComo):
    """
    Ajusta el precio spot S0 restando el valor presente de los dividendos
    discretos que se pagan antes o igual del tiempo t. Este es el método estándar
    para adaptar Black-Scholes (pensado para acciones sin dividendos) al
    caso de dividendos discretos conocidos.

    Parámetros
    ----------
    s0           : Precio spot sin ajustar.
    r            : Tasa libre de riesgo anualizada.
    t            : Tiempo hasta el cual se consideran los dividendos.
    dividendos   : Montos de cada dividendo.
    t_dividendos : Tiempos (en años) en que se paga cada dividendo.

    Regresa
    -------
    float : s0 ajustado (s0 menos el valor presente de los dividendos
            pagados antes de t).
    """
    n_dividendos = len(dividendos)
    PV_d = 0
    for i in range(n_dividendos):
        # solo restamos los dividendos antes de la fecha de ejercicio
        if (t >= t_dividendos[i]>0):
            PV = dividendos[i] * np.exp(-r * t_dividendos[i])
            PV_d += PV
    s0_ajustado = s0 - PV_d
    return s0_ajustado


# ========================== modelo black-76 (opciones sobre futuros) =======================
def Black_76(F0: float, k: float, t: float, r: float, sigma: float, tipo='call'):
    """
    Calcula el precio de una opción europea sobre un futuro/forward usando
    el modelo de Black-76.

    Parámetros
    ----------
    F0    : Precio del futuro/forward actual.
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    r     : Tasa libre de riesgo anualizada (para descuento).
    sigma : volatilidad anualizada del futuro.
    tipo  : 'call' o 'put'.

    Regresa
    -------
    float : precio de la opción.

    Lanza
    -----
    ValueError          : si tipo no es 'call' ni 'put'.
    """
    d1 = (np.log(F0/k) + 0.5*sigma**2*t)/(sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)

    descuento = np.exp(-r*t)

    if tipo.lower() == 'call':
        return descuento * (
            F0*norm.cdf(d1)
            - k*norm.cdf(d2)
        )

    elif tipo.lower() == 'put':
        return descuento * (
            k*norm.cdf(-d2)
            - F0*norm.cdf(-d1)
        )

    else:
        raise ValueError("tipo debe ser 'call' o 'put'")

# ============================= GARMAN KOHLHAGEN (BS CON DIVISAS) ===============================


def Fx_d1_d2(s0: float, k: float, t: float, sigma: float, rd: float, rf: float, tipo="Call"):
    """
    Calcula d1 y d2 del modelo Garman-Kohlhagen (opciones sobre divisas).

    Parámetros
    ----------
    s0    : Tipo de cambio spot actual.
    k     : Precio de ejercicio (strike).
    t     : Tiempo al vencimiento, en años.
    sigma : volatilidad anualizada del tipo de cambio.
    rd    : Tasa libre de riesgo doméstica anualizada.
    rf    : Tasa libre de riesgo extranjera anualizada.
    tipo  : 'call' o 'put' (no afecta el cálculo de d1, d2, solo se recibe
            por consistencia de firma con otras funciones del módulo).

    Regresa
    -------
    (d1, d2) : tuple[float, float]
    """
    d1 = (np.log(s0 / k) + (rd - rf + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def Black_Scholes_Fx(s0: float, k: float, t: float, sigma: float, rd: float, rf: float, tipo="call"):
    """
    Calcula el precio de una opción sobre divisas (Garman-Kohlhagen).

    Parametros:
    s0    : Tipo de cambio spot actual (e.g., 1.20 USD por EUR)
    k     : Precio de ejercicio (Strike)
    t     : Tiempo hasta el vencimiento en años (e.g., 0.5 para 6 meses)
    sigma : volatilidad anualizada del tipo de cambio (e.g., 0.15 para 15%)
    rd    : Tasa de interés libre de riesgo doméstica anualizada (e.g., 0.05)
    rf    : Tasa de interés libre de riesgo extranjera anualizada (e.g., 0.02)
    tipo  : "call" o "put" (sensible a mayúsculas/minúsculas)

    Regresa
    -------
    precio : float

    Lanza
    ------
    ValueError          : si tipo no es 'call' ni 'put'.
    """
    # Evitar división por cero si t es 0
    if t <= 0:
        return max(s0 - k, 0) if tipo.lower() == "call" else max(k - s0, 0)

    # Cálculo de d1 y d2
    d1, d2 = Fx_d1_d2(s0, k, t, sigma, rd, rf, tipo)

    # Cálculo del precio de la opción
    if tipo.lower() == "call":
        precio = s0 * np.exp(-rf * t) * norm.cdf(d1) - k * np.exp(-rd * t) * norm.cdf(d2)
    elif tipo.lower() == "put":
        precio = k * np.exp(-rd * t) * norm.cdf(-d2) - s0 * np.exp(-rf * t) * norm.cdf(-d1)
    else:
        raise ValueError("El parámetro 'tipo' debe ser 'call' o 'put'.")

    return precio


# ======================== OPCIONES AMERICANAS =================================

# ------------------------ APROXIMACION DEL CALL EN OPCIONES AMERICANAS CON DIVIDENDOS ------------------------
def condicion_ejercicio_anticipado(k: float, t: float, r: float, dividendos: ArregloComo, t_dividendos: ArregloComo):
    """
    Verifica, para cada dividendo, si se cumple la condición técnica que
    hace potencialmente óptimo el ejercicio anticipado de un call
    americano justo antes de ese pago de dividendo:

        d_i > K * (1 - exp(-r * (t_(i+1) - t_i)))

    donde t_(i+1) es la siguiente fecha ex-dividendo (o T, el vencimiento,
    si el dividendo i es el último antes de T). Esta es la condición
    rigurosa para múltiples dividendos discretos: cada intervalo se compara
    contra su propia duración hasta el siguiente evento relevante, no
    contra el tiempo total al vencimiento. Usar T en todos los casos
    sobreestima el límite para dividendos intermedios y puede descartar
    ejercicios anticipados que sí serían óptimos.

    IMPORTANTE: se asume que `dividendos` y `t_dividendos` están ordenados
    de forma ascendente por tiempo de pago.

    Parámetros
    ----------
    k            : Precio de ejercicio (strike).
    t            : Tiempo al vencimiento, en años (T).
    r            : Tasa libre de riesgo anualizada.
    dividendos   : Montos de cada dividendo, ordenados por fecha de pago.
    t_dividendos : Tiempos (en años) en que se paga cada dividendo,
                   ordenados ascendentemente.

    Regresa
    -------
    np.ndarray[int] : 1 si la condición se cumple para el dividendo i, 0 si no.
    """
    dividendos = np.asarray(dividendos, dtype=float)
    t_dividendos = np.asarray(t_dividendos, dtype=float)

    # Vector de "siguientes" tiempos: shift de t_dividendos, con T al final
    siguientes = np.append(t_dividendos[1:], t)

    limite = k * (1 - np.exp(-r * (siguientes - t_dividendos)))
    renunciar = (dividendos > limite).astype(int)
    return renunciar


def Aproximacion_Black(s0: float, k: float, t: float, r: float, sigma: float, dividendos: ArregloComo,
                        t_dividendos: ArregloComo, tipo='call'):
    """
    Aproxima el precio de un CALL americano con dividendos discretos,
    evaluando el máximo entre:
      1) el precio europeo al vencimiento final T (con S0 ajustado por
         el valor presente de todos los dividendos), y
      2) el precio de ejercicio anticipado justo antes de cada dividendo
         para el que la condición técnica de ejercicio anticipado se
         cumple.

    Para cada candidato de ejercicio anticipado en t_i, S0 se ajusta
    restando únicamente el valor presente de los dividendos pagados
    ANTES de t_i (dividendos con t_dividendos[j] < t_i). El dividendo D_i
    que se está evaluando NO se resta, porque la valuación en ese instante
    corresponde al precio cum-dividendo (justo antes de que el dividendo
    se pague), que es el valor relevante para decidir si conviene ejercer
    en ese momento.

    IMPORTANTE: se asume que `dividendos` y `t_dividendos` están ordenados
    de forma ascendente por tiempo de pago (mismo supuesto que usa
    `condicion_ejercicio_anticipado`).

    NOTA IMPORTANTE: este resultado teórico (candidatos finitos a evaluar)
    solo es válido para CALLS americanos. Para un PUT americano no existe
    un conjunto finito de fechas candidatas de ejercicio óptimo (el put
    puede convenir ejercerlo en cualquier momento, no solo antes de
    dividendos), por lo que esta función no puede usarse para puts; en su
    lugar, usar el árbol binomial (ArbolBinomial_crr).

    Regresa
    -------
    float : precio aproximado del call americano.

    Lanza
    -----
    NotImplementedError : si tipo='put', indica usar el árbol binomial.
    ValueError          : si tipo no es 'call' ni 'put'.
    """
    if tipo.lower() == 'put':
        raise NotImplementedError('Use modelos binomiales para Puts Americanos con dividendos.')

    elif tipo.lower() == 'call':
        # conversion a np.asrray (nesesaria para las operaciones vectorizadas)
        dividendos = np.asarray(dividendos, dtype=float)
        t_dividendos = np.asarray(t_dividendos, dtype=float)

        # Europea al vencimiento final T
        s_ajustado_final = ajuste_s0(s0, r, t, dividendos, t_dividendos)
        precio_vencimiento = Black_Scholes(s_ajustado_final, k, t, r, sigma, tipo)

        candidatos = [precio_vencimiento]
        condiciones = condicion_ejercicio_anticipado(k, t, r, dividendos, t_dividendos)

        # --- Máscara vectorizada de candidatos válidos: td < T y condición cumplida ---
        validos = (t_dividendos < t) & (condiciones == 1)
        indices_validos = np.nonzero(validos)[0]

        for i in indices_validos:
            td = t_dividendos[i]
            # Máscara booleana de dividendos estrictamente previos a td
            mask_previos = t_dividendos < td
            divs_previos = dividendos[mask_previos]
            t_divs_previos = t_dividendos[mask_previos]

            s_ajustado_ti = ajuste_s0(s0, r, td, divs_previos, t_divs_previos)
            precio_ti = Black_Scholes(s_ajustado_ti, k, td, r, sigma, tipo)
            candidatos.append(precio_ti)

        return max(candidatos)

    else:
        raise ValueError("tipo debe ser 'call' o 'put'")

# ----------------------------- FUNCIONES PARA EL ARBOL BINOMIAL OPCIONES AMERICANAS CON Y SIN DIVIDENDOS --------------------

def _resolver_u_d(sigma: float, t: float, n: int, u: float = None, d: float = None):
    """
    Resuelve los factores de subida/bajada (u, d) del árbol binomial CRR.

    Si el usuario no proporciona 'u' y 'd' explícitamente, se calculan
    automáticamente a partir de 'sigma' usando la parametrización CRR:
        dt = t / n
        u  = exp(sigma * sqrt(dt))
        d  = 1 / u

    Si 'u' y 'd' sí se proporcionan, se regresan tal cual (sin recalcular).

    Regresa
    -------
    (u, d) : tuple[float, float]
    """
    if u is None or d is None:
        dt = t / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
    return u, d


def prob_neutral_riesgo(r: float, t: float, n: int, d: float, u: float) -> float:
    """
    Calcula la probabilidad neutral al riesgo de subida en cada paso del
    árbol binomial CRR.

    Regresa
    -------
    float : probabilidad p en [0, 1] (si el árbol está bien calibrado).
    """
    dt = t / n
    return (np.exp(r * dt) - d) / (u - d)


def ArbolBinomial_crr(s0: float, k: float, r: float, t: float, n: int, u: float, d: float, tipo: str = 'call',
                       es_americana: bool = True, dividendo: ArregloComo = (), t_dividendo: ArregloComo = ()):
    """
    Calcula el precio de una opción (call o put), europea o americana,
    con o sin dividendos discretos, usando el árbol binomial de
    Cox-Ross-Rubinstein (CRR).



    Parámetros
    ----------
    s0           : Precio spot inicial del subyacente.
    k            : Precio de ejercicio (strike).
    r            : Tasa libre de riesgo anualizada.
    t            : Tiempo al vencimiento, en años.
    n            : Número de pasos del árbol.
    u            : Factor de subida por paso.
    d            : Factor de bajada por paso.
    tipo         : 'call' o 'put'.
    es_americana : True para permitir ejercicio anticipado en cada nodo;
                   False para el caso europeo (solo ejercicio al vencimiento).
    dividendo    : Montos de los dividendos discretos.
    t_dividendo  : Tiempos (en años) de cada dividendo.

    Regresa
    -------
    (S, V) : tuple[np.ndarray, np.ndarray]
        S : matriz (n+1)x(n+1) de precios del subyacente en cada nodo.
        V : matriz (n+1)x(n+1) del valor de la opción en cada nodo.
        El precio de la opción es V[0, 0].

    Lanza
    -----
    ValueError si los parámetros de entrada son inconsistentes (n no
    positivo, u/d inválidos, tamaños de dividendo/t_dividendo distintos,
    probabilidad neutral fuera de [0,1], dividendos que generan precios
    del subyacente no positivos, etc.).
    """
    # ------------------------------------------------------------------
    # Validaciones de entrada
    # ------------------------------------------------------------------
    if n <= 0 or not isinstance(n, (int, np.integer)):
        raise ValueError(f"'n' debe ser un entero positivo, se recibió: {n}")

    if s0 <= 0 or k <= 0:
        raise ValueError("'s0' y 'k' deben ser valores positivos")

    if t <= 0:
        raise ValueError("'t' (tiempo total) debe ser positivo")

    if not (0 < d < u):
        raise ValueError(f"Se requiere 0 < d < u, se recibió d={d}, u={u}")

    tipo = tipo.lower()
    if tipo not in ('call', 'put'):
        raise ValueError(f"'tipo' debe ser 'call' o 'put', se recibió: {tipo}")

    # Convertimos a arreglos de numpy para validar tamaños y valores
    dividendo = np.asarray(dividendo, dtype=float)
    t_dividendo = np.asarray(t_dividendo, dtype=float)

    if dividendo.shape != t_dividendo.shape:
        raise ValueError(
            f"'dividendo' y 't_dividendo' deben tener el mismo tamaño. "
            f"Se recibió dividendo con {dividendo.shape[0]} elementos y "
            f"t_dividendo con {t_dividendo.shape[0]} elementos."
        )

    if np.any(dividendo < 0):
        raise ValueError("Los montos en 'dividendo' no pueden ser negativos")

    if np.any(t_dividendo < 0):
        raise ValueError("Los tiempos en 't_dividendo' no pueden ser negativos")

    # Filtrar dividendos relevantes: mismo criterio que ajuste_s0 (0 < t_div <= t)
    mask_relevante = (t_dividendo > 0) & (t_dividendo <= t)
    dividendo = dividendo[mask_relevante]
    t_dividendo = t_dividendo[mask_relevante]

    dt = t / n
    desc = np.exp(-r * dt)
    p = prob_neutral_riesgo(r, t, n, d, u)

    if not (0.0 <= p <= 1.0):
        raise ValueError(
            f"La probabilidad neutral al riesgo p={p:.4f} está fuera de [0,1]. "
            "Revisa que u, d, r y t sean consistentes (posible arbitraje)."
        )

    # ------------------------------------------------------------------
    # Escrowed dividend model: ajustar S0 restando el VP de TODOS los
    # dividendos relevantes (igual que ajuste_s0)
    # ------------------------------------------------------------------
    pv_div_total = np.sum(dividendo * np.exp(-r * t_dividendo))
    s0_star = s0 - pv_div_total

    if s0_star <= 0:
        raise ValueError(
            f"El valor presente de los dividendos ({pv_div_total:.4f}) es "
            f"mayor o igual a s0 ({s0}); revisa los montos ingresados."
        )

    # ------------------------------------------------------------------
    # Árbol multiplicativo PURO sobre S* (recombina exactamente)
    # ------------------------------------------------------------------
    S_star = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        for i in range(j + 1):
            S_star[i, j] = s0_star * (u ** (j - i)) * (d ** i)

    # ------------------------------------------------------------------
    # VP de dividendos remanentes (no pagados aún) en cada columna j.
    # Depende solo de j (tiempo), no de i (camino) -> no rompe recombinación.
    # ------------------------------------------------------------------
    tiempos_columna = np.array([j * dt for j in range(n + 1)])
    div_remanente_pv = np.zeros(n + 1)
    for j in range(n + 1):
        tj = tiempos_columna[j]
        pendientes = t_dividendo > tj
        if np.any(pendientes):
            div_remanente_pv[j] = np.sum(
                dividendo[pendientes] * np.exp(-r * (t_dividendo[pendientes] - tj))
            )

    # Precio "real" del subyacente en cada nodo: S* + dividendos remanentes
    S = S_star + div_remanente_pv[np.newaxis, :]

    # ------------------------------------------------------------------
    # Payoff en el vencimiento (columna n: no quedan dividendos remanentes)
    # ------------------------------------------------------------------
    V = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        if tipo == 'call':
            V[i, n] = max(S[i, n] - k, 0.0)
        else:
            V[i, n] = max(k - S[i, n], 0.0)

    # ------------------------------------------------------------------
    # Inducción hacia atrás
    # ------------------------------------------------------------------
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            v_arriba = V[i, j + 1]
            v_abajo = V[i + 1, j + 1]
            v_continuacion = desc * (p * v_arriba + (1.0 - p) * v_abajo)

            if es_americana:
                if tipo == 'call':
                    v_ejercicio = max(S[i, j] - k, 0.0)
                else:
                    v_ejercicio = max(k - S[i, j], 0.0)
                V[i, j] = max(v_continuacion, v_ejercicio)
            else:
                V[i, j] = v_continuacion

    return S, V


# -------------------- FUNCION AUXILIAR para las clases ----------------------------------
def _precio_por_metodo(s0, k, t, r, sigma, dividendos, t_dividendos, tipo='call', tiempo='c', n=200):
    """
    Función central que decide 'de dónde sale el precio' de una opción
    americana con dividendos discretos:

    - Si tiempo == 'c' (continuo) y tipo == 'call': usa Aproximacion_Black
      (fórmula cerrada aproximada, válida solo para calls).
    - En cualquier otro caso (put, o tiempo == 'd' discreto): usa el árbol
      binomial CRR, calibrando u, d a partir de sigma.

    Todas las clases y letras griegas que necesiten el precio de una
    opción americana con dividendos deben llamar a esta función en vez de
    decidir el método por su cuenta, para mantener un único punto de
    verdad sobre el criterio de selección.

    Regresa
    -------
    float : precio de la opción.
    """
    tipo = tipo.lower()
    tiempo = tiempo.lower()

    if tiempo == 'c' and tipo == 'call':
        return Aproximacion_Black(s0, k, t, r, sigma, dividendos, t_dividendos, tipo)

    u, d = _resolver_u_d(sigma, t, n)
    _, V = ArbolBinomial_crr(s0, k, r, t, n, u, d, tipo=tipo,
                              dividendo=dividendos, t_dividendo=t_dividendos)
    return V[0, 0]


def graficar_arbol_bin(S, V, k, tipo='call', es_americana=True, titulo="Árbol Binomial"):
    """
    Grafica de forma didáctica el árbol binomial ya construido (matrices
    S y V), coloreando cada nodo según la decisión óptima:

    - Verde  : conviene ejercer anticipadamente (solo aplica a nodos no
               terminales de una opción americana).
    - Azul   : conviene mantener/continuar (no ejercer).
    - Naranja: nodo de vencimiento con payoff positivo (dentro del dinero).
    - Gris   : nodo de vencimiento con payoff cero (fuera del dinero).

    Parámetros
    ----------
    S, V         : matrices de precios del subyacente y valor de la
                   opción, tal como las regresa ArbolBinomial_crr.
    k            : precio de ejercicio (strike), usado para calcular el
                   valor intrínseco en cada nodo.
    tipo         : 'call' o 'put'.
    es_americana : si el árbol corresponde a una opción americana (afecta
                   el criterio de coloreado de "conviene ejercer").
    titulo       : título de la gráfica.

    No regresa nada; muestra la gráfica con matplotlib.
    """
    n = S.shape[1] - 1

    G = nx.DiGraph()
    posiciones = {}
    etiquetas = {}
    colores = {}

    # Colores por categoría
    COLOR_EJERCICIO = '#90EE90'      # verde -> conviene ejercer
    COLOR_CONTINUACION = '#ADD8E6'   # azul  -> conviene mantener/continuar
    COLOR_VENCIMIENTO_ITM = '#FFA500'  # naranja -> vencimiento, con valor
    COLOR_VENCIMIENTO_OTM = '#D3D3D3'  # gris  -> vencimiento, sin valor

    for j in range(n + 1):
        for i in range(j + 1):
            nodo_id = f"{j}_{i}"
            x, y = j, j - (2 * i)
            posiciones[nodo_id] = (x, y)

            s_val = S[i, j]
            v_val = V[i, j]

            if tipo.lower() == 'call':
                intrinseco = max(s_val - k, 0.0)
            else:
                intrinseco = max(k - s_val, 0.0)

            # Determinar color / categoría del nodo
            if j == n:
                # Nodo terminal: solo hay payoff, no hay decisión
                color = COLOR_VENCIMIENTO_ITM if v_val > 1e-8 else COLOR_VENCIMIENTO_OTM
            else:
                if es_americana and abs(v_val - intrinseco) < 1e-8 and intrinseco > 1e-8:
                    # El valor de la opción coincide con el ejercicio inmediato
                    color = COLOR_EJERCICIO
                else:
                    color = COLOR_CONTINUACION

            colores[nodo_id] = color
            etiquetas[nodo_id] = f"S={s_val:.2f}\nV={v_val:.2f}"
            G.add_node(nodo_id)

    # Conexiones
    for j in range(n):
        for i in range(j + 1):
            actual = f"{j}_{i}"
            G.add_edge(actual, f"{j+1}_{i}")
            G.add_edge(actual, f"{j+1}_{i+1}")

    node_colors = [colores[n_] for n_ in G.nodes()]

    plt.figure(figsize=(11, 7))
    nx.draw(
        G, posiciones, labels=etiquetas, with_labels=True,
        node_color=node_colors, node_size=2600,
        font_size=8, font_weight='bold', font_color='black',
        edge_color='gray', arrows=True, arrowsize=15
    )

    # Leyenda
    handles = [
        mpatches.Patch(color=COLOR_EJERCICIO, label='Conviene ejercer'),
        mpatches.Patch(color=COLOR_CONTINUACION, label='Conviene mantener (continuación)'),
        mpatches.Patch(color=COLOR_VENCIMIENTO_ITM, label='Vencimiento: con valor'),
        mpatches.Patch(color=COLOR_VENCIMIENTO_OTM, label='Vencimiento: sin valor'),
    ]
    plt.legend(handles=handles, loc='upper left', fontsize=9)

    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
