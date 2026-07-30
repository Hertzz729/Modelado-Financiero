from Riesgo.riesgo import VaR_historico
import numpy as np
# ==============================================================================
# EJEMPLO DE VaR_historico
# ==============================================================================
print("=" * 60)
print("EJEMPLO VaR Histórico (VaR_historico)")
print("=" * 60)

valor_portafolio = 1_000_000

# Escenario: Simulación de 500 días de retornos con un evento de caída o 'Fat Tail'
# (Día de estrés de mercado con pérdida de -6%)
np.random.seed(42)
retornos_historicos = np.random.normal(loc=0.0005, scale=0.012, size=500)
retornos_historicos[150] = -0.065  # Insertamos una caída fuerte de -6.5%

var_hist_90 = VaR_historico(retornos_historicos, valor_portafolio, nivel_confianza=0.90)
var_hist_99 = VaR_historico(retornos_historicos, valor_portafolio, nivel_confianza=0.99)

print(f"VaR Histórico (90% Confianza): ${var_hist_90:,.2f}")
print(f"VaR Histórico (99% Confianza): ${var_hist_99:,.2f}")

