<p align="center">
  <img src="imagenes/banner.png" alt="Banner" width="100%">
</p>

<h1 align="center">
📈 Librería de Finanzas Cuantitativas
</h1>

<p align="center">
Implementación en Python de modelos para la valuación de derivados financieros,
análisis de riesgo y herramientas de finanzas cuantitativas.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Estado-En%20desarrollo-orange?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Compatible-blue?style=for-the-badge)
![SciPy](https://img.shields.io/badge/SciPy-Compatible-blue?style=for-the-badge)

</p>

---

# 📖 Descripción

**Librería de Finanzas Cuantitativas** es un proyecto desarrollado en Python cuyo objetivo es proporcionar una colección de herramientas para la valuación de derivados financieros, el análisis de riesgo y la implementación de modelos clásicos utilizados en ingeniería financiera.

El proyecto está diseñado con una arquitectura modular y orientada a objetos, permitiendo utilizar los distintos modelos tanto con fines educativos como para aplicaciones prácticas de análisis cuantitativo.

Actualmente la librería incluye modelos de valuación para opciones europeas y americanas, cálculo de sensibilidades (Greeks), estimación de volatilidad implícita y métricas de riesgo como Value at Risk (VaR).

---

# ✨ Características

- ✅ Modelo Black-Scholes
- ✅ Modelo Black-76
- ✅ Opciones Europeas
- ✅ Opciones Americanas con dividendos discretos
- ✅ Letras Griegas (Delta, Gamma, Vega, Theta y Rho)
- ✅ Volatilidad implícita
- ✅ Sonrisa de volatilidad
- ✅ VaR Paramétrico
- ✅ VaR Histórico
- ✅ VaR Monte Carlo
- ✅ Arquitectura orientada a objetos
- ✅ Métodos numéricos mediante diferencias finitas

---

# 📂 Estructura del proyecto

```text
finanzas_cuantitativas/

│
├── opciones/
│   ├── clases.py
│   ├── precios.py
│   └── griegas.py
│
├── volatilidad/
│   └── volatilidad.py
│
├── riesgo/
│   └── riesgo.py
│
├── ejemplos/
│
├── imagenes/
│
└── README.md
```

---

# 📚 Modelos implementados

| Categoría | Implementación |
|------------|----------------|
| Opciones Europeas | Black-Scholes |
| Opciones Americanas | Aproximación de Black |
| Opciones sobre futuros | Black-76 |
| Dividendos discretos | ✔ |
| Greeks | Delta, Gamma, Vega, Theta y Rho |
| Riesgo | VaR Paramétrico, Histórico y Monte Carlo |
| Volatilidad | Implícita mediante Newton-Raphson y Bisección |

---

# 🚀 Ejemplo de uso

```python
from finanzas_cuantitativas.opciones import OpcionEuropea

call = OpcionEuropea(
    s0=100,
    k=95,
    t=6/12,
    r=0.08,
    sigma=0.25
)

print("Precio:", call.precio_bs())
print("Delta :", call.delta)
print("Gamma :", call.gamma)
print("Vega  :", call.vega)
print("Theta :", call.theta)
```

---

# 📊 Arquitectura

```mermaid
graph TD

A[Finanzas Cuantitativas]

A --> B[Opciones]

A --> C[Riesgo]

A --> D[Volatilidad]

B --> E[Black-Scholes]

B --> F[Black-76]

B --> G[Opciones Americanas]

B --> H[Greeks]

C --> I[VaR Paramétrico]

C --> J[VaR Histórico]

C --> K[VaR Monte Carlo]

D --> L[Volatilidad Implícita]

D --> M[Sonrisa de Volatilidad]
```

---

# 📈 Ejemplos

## Sonrisa de volatilidad

<p align="center">
<img src="imagenes/sonrisa_volatilidad.png" width="700">
</p>

---

## Sensibilidades (Greeks)

<p align="center">
<img src="imagenes/greeks.png" width="700">
</p>

---

# 🧠 Métodos Numéricos

La librería implementa tanto soluciones analíticas como métodos numéricos para la valuación y análisis de derivados.

Entre ellos se encuentran:

- Diferencias finitas
- Newton-Raphson
- Método de Bisección
- Simulación Monte Carlo

---

# 📌 Aplicaciones

La librería puede utilizarse para:

- Ingeniería Financiera.
- Finanzas Cuantitativas.
- Gestión de Riesgo.
- Modelado de Derivados.
- Cursos universitarios.
- Investigación.
- Desarrollo de estrategias de cobertura.

---

# 🛣 Roadmap

## ✔ Implementado

- [x] Black-Scholes
- [x] Black-76
- [x] Greeks
- [x] Dividendos discretos
- [x] VaR
- [x] Volatilidad Implícita
- [x] Sonrisa de Volatilidad

## 🚧 En desarrollo

- [ ] Árboles Binomiales (CRR)
- [ ] Barone-Adesi & Whaley
- [ ] Bjerksund-Stensland
- [ ] Monte Carlo para Opciones
- [ ] Opciones Exóticas
- [ ] Superficie de Volatilidad
- [ ] Optimización de Portafolios (Markowitz)
- [ ] Cálculo de volatilidad histórica

---

# 📄 Licencia

Este proyecto se distribuye bajo la licencia **MIT**.

Consulta el archivo **LICENSE** para más información.

---

# 👨‍💻 Autor

**Jerson Gallardo**

Matemático Algorítmico — Escuela Superior de Física y Matemáticas (ESFM-IPN)

Intereses:

- Finanzas Cuantitativas
- Machine Learning
- Deep Learning
- Modelado Matemático
- Derivados Financieros

GitHub:

https://github.com/TU_USUARIO

LinkedIn:

https://www.linkedin.com/in/TU_LINKEDIN/

---

<p align="center">

⭐ Si este proyecto te resulta útil, considera darle una estrella al repositorio.

</p>