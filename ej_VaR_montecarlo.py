from Riesgo.riesgo import VaR_montecarlo

# ==============================================================================
# 4. PRUEBA DE VaR_montecarlo
# ==============================================================================
print("=" * 60)
print("4. PRUEBA: VaR Monte Carlo (VaR_montecarlo)")
print("=" * 60)

# Escenario: Portafolio a un horizonte de 10 días útiles (10/252 años)
# Rendimiento anual 10%, Volatilidad anual 20%
valor_portafolio = 1_000_000
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
