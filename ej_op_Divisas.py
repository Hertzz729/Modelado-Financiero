from Opciones.clases_opciones import OpcionForex, OpcionEuropea

s0 = 1.1
k = 1.1
t = 3/12
sigma =  0.10
rd = 0.05 # tasa de interes domestica
rf = 0.03 # tasa de interes extranjera

opcion = OpcionForex(s0,k,t,sigma,rd,rf)

p_call = opcion.precio('call')
p_put = opcion.precio('put')
print(f'precio del call = {p_call}, precio del put con = {p_put}')

d1,d2 = opcion.d1d2
print(f'El valor de d1 = {d1}, d2 = {d2}')
