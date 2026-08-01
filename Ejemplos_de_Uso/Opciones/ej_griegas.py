# opcion 1 mas sencilla
from opciones.griegas import *

# opcion 2 de importacion si se quiere ser especifico con cuanto a que griegas importar
# from opciones.griegas import (
#    # Griegas de Black-Scholes
#    Delta,
#    Rho,
#    # griegas de Black-76 (opciones sobre futuros)
#    Gamma_B76,
#    Vega_A_div,
#)

# ======================================================================
# GRIEGAS BLACK-SCHOLES ESTANDAR (sin dividendos)
# ======================================================================
print("\nBlack-Scholes estándar")

# --- datos ya calculados ---
s0, k, t, r, sigma = 100, 105, 0.5, 0.02, 0.30
d1, d2 = -0.0768, -0.2889  # d1, d2 ya calculados con d1_d2(s0, k, r, t, sigma)

print(f"Delta = {Delta(s0, k, t, r, sigma, tipo='call'):.4f}")
print(f"Gamma = {Gamma(s0, sigma, t, d1):.4f}")
print(f"Theta = {Theta(s0, k, t, sigma, r, d1, d2, tipo='call'):.4f}   (anualizada)")
print(f"Vega  = {Vega(s0, t, sigma, d1):.4f}")
print(f"Rho   = {Rho(k, t, r, d2, tipo='call'):.4f}")

# ======================================================================
# GRIEGAS BLACK-76 (opciones sobre futuros)
# ======================================================================
print("\nBlack-76 (futuros)")

# --- datos ya calculados ---
f0, k2, t2, r2, sigma2 = 180, 185, 5 / 12, 0.06, 0.22
d1_76 = -0.1219  # d1 ya calculado para Black-76
precio_call_fut = 7.8239  # precio de la opción, ya calculado con Black_76(...)

print(f"Delta = {Delta_B76(r2, t2, d1_76, tipo='call'):.4f}")
print(f"Gamma = {Gamma_B76(f0, r2, t2, sigma2, d1_76):.4f}")
print(f"Theta = {Theta_B76(f0, k2, t2, r2, sigma2, d1_76, tipo='call'):.4f}")
print(f"Vega  = {Vega_B76(f0, r2, t2, d1_76):.4f}")
print(f"Rho   = {Rho_B76(t2, precio_call_fut):.4f}   (usa el precio ya calculado)")

# ======================================================================
# GRIEGAS DE AMERICANAS CON DIVIDENDOS DISCRETOS (diferencias finitas)
# ======================================================================
print("\nAmericana con dividendos discretos")

# --- datos ya calculados / dados ---
s0_3, k_3, t_3, r_3, sigma_3 = 45, 50, 8 / 12, 0.08, 0.35
dividendos, t_dividendos = [1.00, 1.20], [2 / 12, 5 / 12]

# Nota: estas griegas no tienen fórmula cerrada; la función misma se
# encarga de perturbar cada parámetro y revaluar (diferencias finitas),
# por lo que aquí no hace falta dar d1/d2/precio de antemano.

print(f"Delta = {Delta_A_div(s0_3, k_3, t_3, r_3, sigma_3, dividendos, t_dividendos, tipo='Call'):.4f}")
print(f"Gamma = {Gamma_A_div(s0_3, k_3, t_3, r_3, sigma_3, dividendos, t_dividendos, tipo='Call'):.4f}")
theta_anual, theta_dia = Theta_A_div(s0_3, k_3, t_3, r_3, sigma_3, dividendos, t_dividendos, tipo='Call')
print(f"Theta = {theta_anual:.4f}  (anualizada)  /  {theta_dia:.4f}  (diaria)")
print(f"Vega  = {Vega_A_div(s0_3, k_3, t_3, r_3, sigma_3, dividendos, t_dividendos, tipo='Call'):.4f}")
print(f"Rho   = {Rho_A_div(s0_3, k_3, t_3, r_3, sigma_3, dividendos, t_dividendos, tipo='Call'):.4f}")
