import numpy as np
from opciones.precios import Black_Scholes
from modelado_financiero.volatilidad import sonrisa_volatilidad

s0, t, r = 100, 0.5, 0.05
strikes = np.array([80, 90, 100, 110, 120]) # lista de strikes
sigmas_reales = np.array([0.30, 0.24, 0.20, 0.23, 0.29])  # forma de "sonrisa"
precios_call = [Black_Scholes(s0, k, t, r, sig, 'call') for k, sig in zip(strikes, sigmas_reales)] # lista de precios del call
precios_put = [Black_Scholes(s0, k, t, r, sig, 'put') for k, sig in zip(strikes, sigmas_reales)] # lista de precios del put


print('===========\nCALL\n==============')
vols_call = sonrisa_volatilidad(s0, r, t, strikes, precios_call, tipo='call')
print(f"Sonrisa recuperada (call): {np.round(vols_call, 3)}")

print('===========\nPUT\n ==============')
vols_put = sonrisa_volatilidad(s0, r, t, strikes, precios_put, tipo='put')
print(f"Sonrisa recuperada (put):  {np.round(vols_put, 3)}")