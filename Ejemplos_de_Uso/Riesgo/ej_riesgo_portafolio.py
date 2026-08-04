import numpy as np
from riesgo.riesgo import Vol_portafolio, VaR_parametrico

print("=============================================================")
print("EJEMPLO INTEGRADO: VaR PARAMÉTRICO CON VOLATILIDAD CALCULADA")
print("=============================================================")

# ------------------------------------------------------------------------------
# 1. PARÁMETROS DEL PORTAFOLIO Y MERCADO
# ------------------------------------------------------------------------------
valor_portafolio = 5_000_000  # $5,000,000 USD de inversión total
pesos = [0.40, 0.35, 0.25]     # 40% Activo A, 35% Activo B, 25% Activo C
rendimientos_esperados_anuales = [0.12, 0.09, 0.15]  # Rendimiento esperado individual (12%, 9%, 15%)

# Matriz de Covarianza DIARIA entre los 3 activos
covarianzas_diarias = [
    [0.000400, 0.000120, 0.000150],  # Vol diaria Activo A ~ 2.0% (sqrt(0.0004))
    [0.000120, 0.000300, 0.000180],  # Vol diaria Activo B ~ 1.7% (sqrt(0.0003))
    [0.000150, 0.000180, 0.000500]   # Vol diaria Activo C ~ 2.2% (sqrt(0.0005))
]

nivel_confianza = 0.95  # 95% de confianza
horizonte_dias = 1      # Horizonte de 1 día

# ------------------------------------------------------------------------------
# PASO 1: CALCULAR LA VOLATILIDAD DEL PORTAFOLIO
# ------------------------------------------------------------------------------

sigma_diaria_p = Vol_portafolio(pesos, covarianzas_diarias) # Usamos tu función Vol_portafolio para obtener la sigma_diaria del portafolio completo


mu_diario_p = np.dot(pesos, rendimientos_esperados_anuales) / 252 # Rendimiento esperado diario del portafolio (mu_p = w . mu)

print(f"Rendimiento diario esperado (µ_p) : {mu_diario_p:.6f} ({mu_diario_p * 100:.4f}%)")
print(f"Volatilidad diaria calculada (σ_p): {sigma_diaria_p:.6f} ({sigma_diaria_p * 100:.2f}%)")

# ------------------------------------------------------------------------------
# PASO 2: CALCULAR EL VaR PARAMÉTRICO USANDO LA VOLATILIDAD CALCULADA
# ------------------------------------------------------------------------------
# Le pasamos la 'sigma_diaria_p' calculada en el paso anterior a VaR_parametrico
var_1dia_95 = VaR_parametrico(
    portafolio_valor=valor_portafolio,
    mu=mu_diario_p,
    sigma=sigma_diaria_p,
    nivel_confianza=nivel_confianza, distribucion='lognormal'
)

print("-" * 60)
print(f"Valor Total del Portafolio       : ${valor_portafolio:,.2f} USD")
print(f"Nivel de Confianza               : {nivel_confianza * 100:.0f}%")
print(f"VaR Paramétrico ({horizonte_dias} día)           : ${var_1dia_95:,.2f} USD")
print(f"Porcentaje de pérdida máxima     : {(var_1dia_95 / valor_portafolio) * 100:.2f}%")
print("-" * 60)
