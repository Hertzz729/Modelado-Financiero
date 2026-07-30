from Opciones.precios import Black_Scholes
from Volatilidad.volatilidad import estimacion_sigma_tangente

s0 = 100
k = 110
t = 1.0
r = 0.06
precio_call = Black_Scholes(s0, k, t, r, 0.35, 'call')
precio_put = Black_Scholes(s0, k, t, r, 0.35, 'put')

"""
Nota: en estimacion_sigma_Newton los argumentos sigma pude pasarse como argumento o no como argumento. 
ej: estimacion_sigma_tangente(s0, k, t, r, precio_call, sigma_0, sigma_1 )
ej: estimacion_sigma_tangente(s0,k,t,r, precio_call)
"""

print('===========\nCALL\n==============')
sigma_call = estimacion_sigma_tangente(s0, k, t, r, precio_call, 0.40, 0.30) # por defecto la funcion usa el tipo de opcion CALL (puede pasarse o no como parametro)
print(f"Sigma implícita (call): {sigma_call:.4f}")

print('===========\nPUT\n ==============')
sigma_put = estimacion_sigma_tangente(s0, k, t, r, precio_put, tipo='put')
print(f"Sigma implícita (put):  {sigma_put:.4f}")