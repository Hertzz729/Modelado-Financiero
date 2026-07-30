from Opciones.precios import Black_Scholes
from Volatilidad.volatilidad import estimacion_sigma_biseccion

s0 = 100
k = 100
t = 0.5
r = 0.05
precio_call = Black_Scholes(s0, k, t, r, 0.28)
precio_put = Black_Scholes(s0, k, t, r, 0.28, 'put')

print('===========\nCALL\n==============')
sigma_call = estimacion_sigma_biseccion(s0, k, t, r, precio_call, sigma_low=0.01, sigma_high=1.0, tipo='call') # por defecto la funcion usa el tipo de opcion CALL (puede pasarse o no como parametro)
print(f"Sigma implícita (call): {sigma_call:.4f}")

print('===========\nPUT\n ==============')
sigma_put = estimacion_sigma_biseccion(s0, k, t, r, precio_put, sigma_low=0.01, sigma_high=1.0, tipo='put')
print(f"Sigma implícita (put):  {sigma_put:.4f}")