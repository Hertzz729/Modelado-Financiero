from opciones.precios import Black_Scholes
from modelado_financiero.volatilidad import estimacion_sigma_Newton

s0 = 80
k = 85
t = 0.05
r =  0.10
precio_call = Black_Scholes(s0, k, t, r, 0.22, 'call')  # precio "de mercado"
precio_put = Black_Scholes(s0, k, t, r, 0.22, 'put')

"""
Nota: en estimacion_sigma_Newton el argumento sigma_inicial pude pasarse como argumento o no como argumento. 
ej: estimacio_sigma_Newton(s0,k,t,r, precio_call, 0.22)
ej: estimacio_sigma_Newton(s0,k,t,r, precio_call)
"""

print('===========\nCALL\n==============')

sigma_call = estimacion_sigma_Newton(s0, k, t, r,precio_call, 0.22) # por defecto la funcion usa el tipo de opcion CALL (puede pasarse o no como parametro)
print(f"Sigma implícita (call): {sigma_call:.4f}")

print('===========\nPUT\n ==============')

sigma_put = estimacion_sigma_Newton(s0, k, t, r, precio_put, tipo='put')
print(f"Sigma implícita (put):  {sigma_put:.4f}")