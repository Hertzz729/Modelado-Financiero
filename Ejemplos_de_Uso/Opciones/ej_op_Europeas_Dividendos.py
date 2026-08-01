print('============================================\nOPCIONES EUROPEAS CON DIVIDENDOS\n============================================')

from opciones.clases_opciones import OpcionEuropeaDiv

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
print(f'precio call = {p_call_div}, precio put = {p_put_div}')

delta_call_div = europea_dividendos.delta('call')
delta_put_div = europea_dividendos.delta('put')
print(f'delta call = {delta_call_div}, delta put = {delta_put_div}')

gama_div = europea_dividendos.gamma()
print(f'gama = {gama_div}')

theta_call_div = europea_dividendos.theta('call')
theta_put_div = europea_dividendos.theta('put')
print(f'theta del call = {theta_call_div}, theta del put = {theta_put_div}')

vega_div = europea_dividendos.vega()
print(f'vega = {vega_div}')

rho_call_div = europea_dividendos.rho('call')
rho_put_div = europea_dividendos.rho('put')
print(f'rho call = {rho_call_div}, rho put = {rho_put_div}')