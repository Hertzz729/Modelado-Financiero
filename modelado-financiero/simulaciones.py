"""
simulaciones.py
===============
Módulo reservado para funciones de simulación (por ejemplo, simulación
de trayectorias de precios mediante Montecarlo/GBM). Actualmente solo
define los alias de tipos comunes al resto del proyecto; las funciones
de simulación se agregarán aquí conforme se desarrollen.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from decimal import Decimal
from typing import Sequence, TypeAlias

ArregloComo: TypeAlias = Sequence[float]
MatrizComo: TypeAlias = Sequence[Sequence[float]]
