"""
Uso_Class_Opciones.py
======================
Script de ejemplo/demostración de uso de las clases definidas en
clases/opciones.py: OpcionEuropea, OpcionEuropeaDiv, OpcionFuturos y
OpcionAmericanaDiv.
"""

#=========================================================================================
#|                                    OPCIONES EUROPEAS                                  |
#=========================================================================================
from clases.opciones import OpcionEuropea

s0 = 80
k = 80
r = 0.10
t = 6/12
sigma = 0.2


op_europea = OpcionEuropea(s0, k, t, r, sigma)  # Declaramos el objeto de opcion europea

d1, d2 = op_europea.d1d2  # obtener d1 y d2 (calculados internamente por la property)
print(f'resultado: d1 = {d1}, d2 = {d2}')

precio_call = op_europea.precio()  # precio de la opcion europea tipo call
precio_put = op_europea.precio('Put')  # precio de la opcion europea tipo put
print(f'precio call = {precio_call}, precio_put = {precio_put}')

# --------- letras griegas de opcion europea --------------------
delta_call_europeo = op_europea.delta('call')
delta_put_europeo = op_europea.delta('put')
print(f'delta call = {delta_call_europeo}, delta put = {delta_put_europeo}')

gama_europea = op_europea.gamma()
print(f'gama europea = {gama_europea}')

thetaC_europea = op_europea.theta('call')
thetaP_europea = op_europea.theta('put')
print(f'theta call = {thetaC_europea}, theta Put = {thetaP_europea}')

vega = op_europea.vega()
print(f'el valor de vega = {vega}')

rhoC_europea = op_europea.rho('call')
rhoP_europea = op_europea.rho('put')
print(f'rho call = {rhoC_europea}, rho put = {rhoP_europea}')

print('============================================\n============================================')
#=========================================================================================
#|                            OPCIONES EUROPEAS CON DIVIDENDOS                           |
#=========================================================================================

from clases.opciones import OpcionEuropeaDiv

s0 = 80
k = 80
r = 0.10
t = 6/12
sigma = 0.2
dividendos = [0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t_dividendos = [1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12]

# Orden correcto de argumentos: s0, k, t, r, sigma, dividendos, t_dividendos
europea_dividendos = OpcionEuropeaDiv(s0, k, t, r, sigma, dividendos, t_dividendos)

d1_div, d2_div = europea_dividendos.d1d2
print(f'd1 = {d1_div}, d2 = {d2_div}')

s0_ajustado = europea_dividendos.s0_ajustado
print(f's0_ajustado = {s0_ajustado}')

p_call_div = europea_dividendos.precio()
p_put_div = europea_dividendos.precio('put')
print(f'precio del call con dividendos = {p_call_div}, precio del put con dividendos = {p_put_div}')

delta_call_div = europea_dividendos.delta('call')
delta_put_div = europea_dividendos.delta('put')
print(f'delta call con dividendos = {delta_call_div}, delta put con dividendos = {delta_put_div}')

gama_div = europea_dividendos.gamma()
print(f'gama con dividendos = {gama_div}')

theta_call_div = europea_dividendos.theta('call')
theta_put_div = europea_dividendos.theta('put')
print(f'theta del call con dividendos = {theta_call_div}, theta del put con dividendos = {theta_put_div}')

vega_div = europea_dividendos.vega()
print(f'vega con dividendos = {vega_div}')

rho_call_div = europea_dividendos.rho('call')
rho_put_div = europea_dividendos.rho('put')
print(f'rho call con dividendos = {rho_call_div}, rho put con dividendos = {rho_put_div}')


print('============================================\n============================================')
#=========================================================================================
#|                            OPCIONES CON FORWARDS (FUTUROS)                            |
#=========================================================================================
from clases.opciones import OpcionFuturos

f0 = 100
r = 0.05
t = 1
sigma = 0.2
k = 105

op_forward = OpcionFuturos(f0, k, t, r, sigma)

d1, d2 = op_forward.d1d2
print(f'd1 = {d1}, d2 = {d2}')

p_call_forward = op_forward.precio('call')
p_put_forward = op_forward.precio('put')
print(f'call forward = {p_call_forward}, put forward = {p_put_forward}')

delta_c_fw = op_forward.delta('call')
delta_p_fw = op_forward.delta('put')
print(f'delta call = {delta_c_fw}, delta put = {delta_p_fw}')

gamma_fw = op_forward.gamma()
print(f'gamma fw = {gamma_fw}')

theta_c_fw = op_forward.theta('call')
theta_p_fw = op_forward.theta('put')
print(f'theta call = {theta_c_fw}, theta put = {theta_p_fw}')

vega_fw = op_forward.vega()
print(f'vega fw = {vega_fw}')

rho_c_fw = op_forward.rho('call')
rho_p_fw = op_forward.rho('put')
print(f'rho call = {rho_c_fw}, rho put = {rho_p_fw}')



print('============================================\n============================================')
#=========================================================================================
#|                   OPCIONES CON AMERICANAS CON DIVIDENDOS DISCRETOS                    |
#=========================================================================================

from clases.opciones import OpcionAmericanaDiv

s0 = 80
k = 80
r = 0.10
t = 6/12
sigma = 0.2
dividendos = [0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
t_dividendos = [1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12]

opcion_americana_div = OpcionAmericanaDiv(s0, k, t, r, sigma, dividendos, t_dividendos)

# tipo='call' -> usa Aproximacion_Black (tiempo='c' por defecto)
# tipo='put'  -> cae automáticamente al árbol binomial (no existe Black para puts)
p_call_A_Div = opcion_americana_div.precio('call')
p_put_A_Div = opcion_americana_div.precio('put')
print(f'p_call_A_Div = {p_call_A_Div}, p_put_A_Div = {p_put_A_Div}')

delta_adiv = opcion_americana_div.delta('call')
print(f'delta_adiv = {delta_adiv}')

gamma_adiv = opcion_americana_div.gamma('call')
print(f'gamma_adiv = {gamma_adiv}')

theta_adiv, _ = opcion_americana_div.theta('call')
print(f'theta_adiv = {theta_adiv}')

vega_adiv = opcion_americana_div.vega('call')
print(f'vega_adiv = {vega_adiv}')

rho_adiv = opcion_americana_div.rho('call')
print(f'rho_adiv = {rho_adiv}')
