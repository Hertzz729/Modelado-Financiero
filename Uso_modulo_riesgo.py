from Riesgo.riesgo import *
import numpy as np


# ==============================================================================
# 1. PRUEBA DE Vol_portafolio
# ==============================================================================
print("=" * 60)
print("1. PRUEBA: Volatilidad de Portafolio (Vol_portafolio)")
print("=" * 60)

# Escenario Real: Portafolio de 3 activos (ej. Apple, Microsoft, Amazon)
# Entrada usando LISTAS normales de Python para probar la conversión interna a np.ndarray
pesos_lista = [0.4, 0.35, 0.25]  # Suman 1.0 (100%)

covarianzas_matriz = [
    [0.040, 0.012, 0.015],  # Vol Apple ~ 20%
    [0.012, 0.035, 0.018],  # Vol Microsoft ~ 18.7%
    [0.015, 0.018, 0.050]   # Vol Amazon ~ 22.3%
]

vol_p = Vol_portafolio(pesos_lista, covarianzas_matriz)
print(f"Pesos del portafolio    : {pesos_lista}")
print(f"Volatilidad estimada    : {vol_p:.4f} ({vol_p*100:.2f}%)")

# Verificación matemática básica:
# Si el portafolio tiene activos correlacionados positivamente, la volatilidad
# debe estar dentro del rango aproximado de las volatilidades individuales (~18% a 22%).
assert 0.15 < vol_p < 0.25, "Error: La volatilidad calculada está fuera de rango."
print("Status: PASÓ LA PRUEBA\n")


# ==============================================================================
# 2. PRUEBA DE VaR_parametrico
# ==============================================================================
print("=" * 60)
print("2. PRUEBA: VaR Paramétrico (VaR_parametrico)")
print("=" * 60)

# Escenario: Portafolio de $1,000,000 USD con un rendimiento anual esperado del 8% (0.08)
# y una volatilidad diaria calculada de 1.5% (0.015) a un horizonte de 1 día.
valor_portafolio = 1_000_000
mu_diario = 0.08 / 252       # Retorno esperado diario (~0.000317)
sigma_diario = 0.015         # Volatilidad diaria (1.5%)
confianza = 0.95             # 95% de confianza

var_param = VaR_parametrico(valor_portafolio, mu_diario, sigma_diario, nivel_confianza=confianza)

print(f"Valor del Portafolio : ${valor_portafolio:,.2f}")
print(f"Nivel de Confianza   : {confianza*100}%")
print(f"VaR Paramétrico (1 día): ${var_param:,.2f}")

# Prueba de caso límite: Si el rendimiento esperado es extremadamete alto (p.ej. mu = 50%),
# la pérdida no debería ser negativa gracias al max(var, 0).
var_extremo = VaR_parametrico(valor_portafolio, mu=0.50, sigma=0.01, nivel_confianza=0.95)
print(f"VaR con Mu alto (Control de límite no negativo): ${var_extremo:,.2f}")
assert var_extremo >= 0, "Error: El VaR no debe ser menor a 0."
print("Status: PASÓ LA PRUEBA\n")


# ==============================================================================
# 3. PRUEBA DE VaR_historico
# ==============================================================================
print("=" * 60)
print("3. PRUEBA: VaR Histórico (VaR_historico)")
print("=" * 60)

# Escenario: Simulación de 500 días de retornos con un evento de caída o 'Fat Tail'
# (Día de estrés de mercado con pérdida de -6%)
np.random.seed(42)
retornos_historicos = np.random.normal(loc=0.0005, scale=0.012, size=500)
retornos_historicos[150] = -0.065  # Insertamos una caída fuerte de -6.5%

var_hist_90 = VaR_historico(retornos_historicos, valor_portafolio, nivel_confianza=0.90)
var_hist_99 = VaR_historico(retornos_historicos, valor_portafolio, nivel_confianza=0.99)

print(f"VaR Histórico (90% Confianza): ${var_hist_90:,.2f}")
print(f"VaR Histórico (99% Confianza): ${var_hist_99:,.2f}")

# Verificación lógica: A mayor nivel de confianza (99% vs 90%), el VaR DEBE ser mayor.
assert var_hist_99 > var_hist_90, "Error: El VaR al 99% debería ser mayor que al 90%."
print("Status: PASÓ LA PRUEBA\n")


# ==============================================================================
# 4. PRUEBA DE VaR_montecarlo
# ==============================================================================
print("=" * 60)
print("4. PRUEBA: VaR Monte Carlo (VaR_montecarlo)")
print("=" * 60)

# Escenario: Portafolio a un horizonte de 10 días útiles (10/252 años)
# Rendimiento anual 10%, Volatilidad anual 20%
mu_anual = 0.10
sigma_anual = 0.20
t_dias = 10 / 252  # Horizonte de tiempo en años

# Ejecución reproducible fijando la semilla 'seed'
var_mc_1 = VaR_montecarlo(
    portafolio_valor=valor_portafolio,
    mu=mu_anual,
    sigma=sigma_anual,
    t=t_dias,
    n_simulaciones=50_000,
    nivel_confianza=0.95,
    seed=123
)

# Segunda ejecución con la misma semilla para probar la reproducibilidad
var_mc_2 = VaR_montecarlo(
    portafolio_valor=valor_portafolio,
    mu=mu_anual,
    sigma=sigma_anual,
    t=t_dias,
    n_simulaciones=50_000,
    nivel_confianza=0.95,
    seed=123
)

print(f"Horizonte de tiempo  : 10 días hábiles ({t_dias:.4f} años)")
print(f"VaR Monte Carlo (Prueba 1): ${var_mc_1:,.2f}")
print(f"VaR Monte Carlo (Prueba 2): ${var_mc_2:,.2f}")

# Verificación de reproducibilidad gracias al seed
assert var_mc_1 == var_mc_2, "Error: La semilla no garantizó la reproducibilidad."
print("Status: PASÓ LA PRUEBA\n")
