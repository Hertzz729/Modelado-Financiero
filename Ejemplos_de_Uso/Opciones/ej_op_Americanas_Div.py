print('============================================\nOPCIONES CON AMERICANAS CON DIVIDENDOS DISCRETOS\n============================================')

from opciones.clases_opciones import OpcionAmericanaDiv
import numpy as np
"""
Modificar el metodo de aproximación por el metodo Bjerksund-Stensland
"""

s0 = 80
k = 80
r = 0.10
t = 6/12
sigma = 0.2
# Nota: Se puede trabajar con np.array en lugar de listas
dividendos = [0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t_dividendos = [1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12]

opcion_americana_div = OpcionAmericanaDiv(s0, k, t, r, sigma, dividendos, t_dividendos)

# tipo='call' -> usa Aproximacion_Black (tiempo='c' por defecto)
# tipo='put'  -> cae automáticamente al árbol binomial (no existe Black para puts)
p_call_A_Div = opcion_americana_div.precio('call')
p_put_A_Div = opcion_americana_div.precio('put')
print(f'precio call = {p_call_A_Div}, precio put = {p_put_A_Div}')

delta_call_adiv = opcion_americana_div.delta('call')
delta_put_adiv = opcion_americana_div.delta('put')
print(f'delta = {delta_call_adiv}, delta put = {delta_put_adiv}')

gamma_adiv = opcion_americana_div.gamma('call')
print(f'gamma = {gamma_adiv}')

theta_call_adiv, _ = opcion_americana_div.theta('call') # devuelve theta anualizada y diaria
theta_put_adiv, _ = opcion_americana_div.theta('call') # devuelve theta anualizada y diaria
print(f'theta call = {theta_call_adiv}, theta put = {theta_put_adiv}')

vega_call_adiv = opcion_americana_div.vega('call')
vega_put_adiv = opcion_americana_div.vega('put')
print(f'vega = {vega_call_adiv}, {vega_put_adiv} ')

rho_adiv = opcion_americana_div.rho('call')
print(f'rho = {rho_adiv}')
