print('============================================\nOPCIONES CON FORWARDS (FUTUROS)\n============================================')

from Opciones.clases_opciones import OpcionFuturos

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
print(f'precio call = {p_call_forward}, precio put = {p_put_forward}')

delta_c_fw = op_forward.delta('call')
delta_p_fw = op_forward.delta('put')
print(f'delta call = {delta_c_fw}, delta put = {delta_p_fw}')

gamma_fw = op_forward.gamma()
print(f'gamma = {gamma_fw}')

theta_c_fw = op_forward.theta('call')
theta_p_fw = op_forward.theta('put')
print(f'theta call = {theta_c_fw}, theta put = {theta_p_fw}')

vega_fw = op_forward.vega()
print(f'vega = {vega_fw}')

rho_c_fw = op_forward.rho('call')
rho_p_fw = op_forward.rho('put')
print(f'rho call = {rho_c_fw}, rho put = {rho_p_fw}')
