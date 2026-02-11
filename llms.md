# llms.md — Flow-Lenia (ISAL 2025) — Fórmulas verificadas + guía de implementación

Fuente: *Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist: Discovering Diverse Ecosystem Dynamics* (Michel et al., ISAL 2025). fileciteturn1file10

> Nota importante: este paper **define** las ecuaciones (1)–(7) (afinidad, campo de flujo, localización de parámetros, mixing y mutación) pero **no** detalla la forma concreta de los kernels `K_i` ni de las growth functions `G_i` (eso viene del trabajo base Flow-Lenia 2023 citado por el paper). Aquí dejo la parte implementable **tal cual** aparece en ISAL 2025, con notación consistente y aclaraciones mínimas. fileciteturn1file3

---

## 1) Espacios y símbolos

- Grid discreto: \(L\) (por ejemplo \(256\times256\)). fileciteturn1file4  
- Canales: \(C\). Estado:
  \[
  A^t : L \to \mathbb{R}_{\ge 0}^{C}
  \]
  interpretado como **densidad de materia** (no acotada a \([0,1]\)). fileciteturn1file3  
- Kernels de convolución:
  \[
  K=\{K_i: L\to[0,1]\ \mid\ i=1,\dots,|K|\}
  \]
  fileciteturn1file3  
- Growth functions:
  \[
  G=\{G_i:[0,1]\to[-1,1]\ \mid\ i=1,\dots,|K|\}
  \]
  fileciteturn1file3  
- Para cada par kernel–growth \(i\), el paper usa dos canales:
  - \(c_i^0\): **source channel** para la convolución \(K_i * A^t_{c_i^0}\)
  - \(c_i^1\): **target channel** al que contribuye la afinidad

- Iverson bracket: \([c_i^1 = j]\) vale 1 si \(c_i^1=j\), si no 0. fileciteturn1file3  

---

## 2) Ecuación de afinidad (Flow-Lenia “base”)

**(1) Afinidad por canal** \(j\):
\[
U^t_j(x)=\sum_{i=1}^{|K|} h_i\cdot G_i\!\left( (K_i * A^t_{c_i^0})(x)\right)\cdot [c_i^1=j]
\tag{1}
\]
donde \(h_i\in\mathbb{R}\) pondera cada par kernel–growth. fileciteturn1file3  

**Implementación:**
- Precompute convoluciones \((K_i * A^t_{c_i^0})\) por \(i\) y evalúa \(G_i(\cdot)\).
- Acumula en \(U^t_j\) sólo si \([c_i^1=j]=1\).

---

## 3) Campo de flujo (atracción + difusión dependiente de concentración)

El paper define un campo de flujo por canal \(i\) (aquí \(i\) indexa canales del estado, no kernels):

\[
\begin{cases}
F^t_i = (1-\alpha^t)\,\nabla U^t_i \;-\; \alpha^t\,\nabla A^t_{\Sigma} \\
\alpha^t(p) = \left[\left(\dfrac{A^t_{\Sigma}(p)}{\theta_A}\right)^n\right]^1_0
\end{cases}
\tag{2}
\]

donde:
- \(A^t_{\Sigma}(p)\) es la **masa total** (suma sobre canales) en la celda \(p\),
- \(\theta_A\) controla la intensidad de la “difusión anti-concentración”,
- \(n\) es un exponente,
- \([\cdot]^1_0\) es **clamp** a \([0,1]\). fileciteturn1file3  

**Lectura práctica:**
- Si hay poca masa, \(\alpha\approx0\) ⇒ flujo ~ \(\nabla U\) (atracción).
- Si hay mucha masa, \(\alpha\to1\) ⇒ domina \(-\nabla A_\Sigma\) (difusión/repulsión por gradiente de concentración).

---

## 4) Conservación de masa: “reintegration tracking”

Para mover materia según \(F^t\) preservando masa, usan **reintegration tracking** (Moroz 2020), descrito como un sistema particle-in-grid donde cada celda “envía” materia y, si cae entre celdas, se distribuye proporcionalmente. fileciteturn1file3  

> El paper ISAL 2025 no da fórmula cerrada de la reintegración, sólo el principio. Para implementarlo fielmente necesitas el algoritmo de Moroz (2020) citado. fileciteturn1file14  

---

## 5) Localización de parámetros (genes) que viajan con la materia

Definen un mapa de parámetros:
\[
P : L \to \Theta
\]
y en su implementación “embeben” el vector de pesos kernel \(h\in\mathbb{R}^{|K|}\) como parámetros locales \(P^t_i(x)\). Entonces la afinidad se vuelve:

\[
U^t_j(x)=\sum_{i=1}^{|K|} P^t_i(x)\cdot G_i\!\left( (K_i * A^t_{c_i^0})(x)\right)\cdot [c_i^1=j]
\tag{3}
\]
fileciteturn1file1  

**Interpretación:** distintos “trozos” de materia pueden portar distintos \(P\) ⇒ coexisten reglas distintas en la misma grilla. fileciteturn1file0  

---

## 6) Mixing rules (resolver conflictos cuando llega materia con distintos parámetros)

Como múltiples fuentes pueden contribuir a una misma celda destino \(x_{dest}\), definen una regla de mezcla:

\[
P^{t+dt}(x_{dest}) = M\Big(\big(A^t(x_{src}),\,P^t(x_{src}),\,I(x_{src},x_{dest})\big)\ \big|\ x_{src}\in L\Big)
\tag{4}
\]
donde \(I(x_{src},x_{dest})\) denota la proporción de materia que fluye de \(x_{src}\) a \(x_{dest}\). fileciteturn1file1  

### 6.1 Negotiation rule (la que introduce este paper)

La “negotiation rule” hace selección estocástica de la fuente cuyos parámetros hereda el destino, ponderando:
- masa de la fuente \(A^t(x_{src})\),
- cantidad que llega \(I(x_{src},x_{dest})\),
- “afinidad de mezcla” \(V^t(x_{src})\),
- presión selectiva \(\beta\) (inverse temperature).

La probabilidad queda:

\[
\mathbb{P}\Big(P^{t+dt}(x_{dest}) = P^t(x_{src})\Big)
=
\frac{\exp\Big(\beta\,A^t(x_{src})\,I(x_{src},x_{dest})\,V^t(x_{src})\Big)}
{\sum_{x\in L}\exp\Big(\beta\,A^t(x)\,I(x,x_{dest})\,V^t(x)\Big)}
\tag{5}
\]
fileciteturn1file0  

y definen \(V^t\) como:

\[
V^t(x_{src})=\sum_{j=1}^{C}\sum_{i=1}^{|K|}
Q^t_i(x_{src})\cdot G_i\!\left((K_i * A^t_{c_i^0})(x_{src})\right)\cdot [c_i^1=j]
\tag{6}
\]
donde \(Q^t_i(x)\) es un conjunto **separado** de parámetros usado **sólo** para computar la afinidad de mezcla. fileciteturn1file2  

**Detalles implementables:**
- Para cada \(x_{dest}\), sólo consideras \(x_{src}\) con \(I(x_{src},x_{dest})>0\) (soporte escaso).
- Calculas logits \( \ell(x_{src}) = \beta A^t(x_{src}) I(x_{src},x_{dest}) V^t(x_{src}) \).
- Softmax sobre esos logits y sampleas una fuente.
- Asignas \(P^{t+dt}(x_{dest}) \leftarrow P^t(x_{src^\*})\).

---

## 7) Mutación de parámetros locales

A una frecuencia definida por el experimento, seleccionan áreas aleatorias del grid y aplican ruido gaussiano multivariante:

\[
P^{t+dt}(x)=P^t(x)+\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,\Sigma)
\tag{7}
\]
fileciteturn1file2  

---

## 8) Pseudocódigo mínimo de un step (orientativo)

1. **Compute affinity** \(U^t\) usando (1) o (3) si hay parámetros localizados.
2. **Compute mass sum** \(A^t_\Sigma\) y \(\alpha^t\) (clamp) según (2).
3. **Compute flow field** \(F^t\) según (2) (gradientes discretos).
4. **Advect/reintegrate** materia (y “arrastrar” \(P\) con la materia) con reintegration tracking. fileciteturn1file3  
5. **Mix parameters** en celdas destino con (4) + negotiation rule (5)–(6).
6. **Mutate** \(P\) en regiones (7) cuando toque.

---

## 9) Qué NO está totalmente especificado en este PDF (y necesitas del paper base)

- Forma paramétrica exacta de \(K_i\) y \(G_i\) (y normalizaciones).
- Discretización exacta de \(\nabla\) y de la reintegración (Moroz 2020 / Flow-Lenia 2023).
- Definición exacta de \(I(x_{src},x_{dest})\) (sale naturalmente del paso de reintegración/advección). fileciteturn1file3  

Si vas a implementar “Flow-Lenia” fiel al ecosistema del paper, lo normal es partir del código o especificación del trabajo Flow-Lenia 2023 y aplicar encima las ecuaciones (3)–(7) y la instrumentación de métricas.

---
