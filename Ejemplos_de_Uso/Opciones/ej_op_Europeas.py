print('============================================\nOPCIONES EUROPEAS\n============================================')

from opciones.clases_opciones import OpcionEuropea

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