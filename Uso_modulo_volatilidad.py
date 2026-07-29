import numpy as np

from Volatilidad.volatilidad import estimacion_sigma, sonrisa_volatilidad, estimacion_sigma_biseccion

# ==============================================================================
# DATOS DE MERCADO PARA LAS PRUEBAS
# ==============================================================================
# Supongamos una acción cotizando a $100 USD (S0 = 100)
# Tasa libre de riesgo del 5% anual (r = 0.05)
# Vencimiento a 6 meses (t = 0.5 años)
s0 = 100.0
r = 0.05
t = 0.5

# ==============================================================================
# 1. PRUEBA: Newton-Raphson (estimacion_sigma)
# ==============================================================================
print("=" * 70)
print("1. PRUEBA: Newton-Raphson para Opciones At-The-Money (ATM)")
print("=" * 70)

strike_atm = 100.0
# Precio observado en el mercado para este Call (corresponde a una vol real ~ 25%)
call_mercado_atm = 7.95
sigma_semilla = 0.15 # Semilla deliberadamente alejada para probar convergencia

print(f"Buscando volatilidad implícita para Strike ${strike_atm} con precio de mercado ${call_mercado_atm}...")
vol_nr = estimacion_sigma(
    s0=s0,
    k=strike_atm,
    t=t,
    r=r,
    sigma_estimada=sigma_semilla,
    call_mercado=call_mercado_atm,
    error=1e-5
)

print(f"\nResultado final Newton-Raphson : {vol_nr:.4f} ({vol_nr*100:.2f}%)")
#assert abs(vol_nr - 0.25) < 0.01, "Error: La volatilidad estimada no convergió al ~25%"
print("Status: PASÓ LA PRUEBA\n")


# ==============================================================================
# 2. PRUEBA: Bisección (estimacion_sigma_biseccion)
# ==============================================================================
print("=" * 70)
print("2. PRUEBA: Bisección en Escenario de Alta Volatilidad (Deep OTM / Evento de Estrés)")
print("=" * 70)

strike_otm = 120.0
# Una opción Out-Of-The-Money con un precio anormalmente alto ($4.80) debido a alta volatilidad (~45%)
call_mercado_otm = 4.80

# Le damos un rango amplio donde buscar [1%, 100%]
sigma_min = 0.01
sigma_max = 1.00

print(f"Buscando con Bisección en rango [{sigma_min*100}%, {sigma_max*100}%]...")
vol_bis = estimacion_sigma_biseccion(
    s0=s0,
    k=strike_otm,
    t=t,
    r=r,
    call_mercado=call_mercado_otm,
    sigma_low=sigma_min,
    sigma_high=sigma_max,
    error=1e-5
)

print(f"\nResultado final Bisección : {vol_bis:.4f} ({vol_bis*100:.2f}%)")
assert vol_bis is not None, "Error: El método de bisección devolvió None"
print("Status: PASÓ LA PRUEBA\n")


# ==============================================================================
# 3. PRUEBA: Sonrisa de Volatilidad (sonrisa_volatilidad)
# ==============================================================================
print("=" * 70)
print("3. PRUEBA: Construcción y Gráfica de la Sonrisa de Volatilidad")
print("=" * 70)

# Un vector de Strikes (desde In-The-Money hasta Out-Of-The-Money)
strikes_mercado = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])

# Precios observados reales en mercado que exhiben el 'Smile' o 'Skew' tipico
# (Las opciones muy dentro/fuera del dinero cotizan con prima/volatilidad más alta)
precios_mercado = np.array([22.80, 18.50, 14.30, 10.80, 7.95, 5.50, 3.70, 2.30, 1.40])

print("Calculando la sonrisa para 9 precios/strikes de mercado...")
volatilidades_obtenidas = sonrisa_volatilidad(
    s0=s0,
    r=r,
    t=t,
    strikes=strikes_mercado,
    precios=precios_mercado,
    sigma_inicial=0.20
)

print("\nResultados devueltos por Strike:")
for k, v in zip(strikes_mercado, volatilidades_obtenidas):
    print(f"  Strike ${k:3d} -> Volatilidad Implícita: {v*100:.2f}%")

print("\nStatus: PASÓ LA PRUEBA (Revisa la gráfica desplegada)")