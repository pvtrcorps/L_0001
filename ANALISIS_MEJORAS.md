# Análisis técnico del juego (Flow-Lenia en Godot)

## Diagnóstico general

El proyecto ya implementa una base sólida: simulación en GPU con compute shaders, evolución genética localizada, visualización en tiempo real y herramientas de inspección de especies. La arquitectura separa razonablemente UI (`Main.gd`), simulación (`LeniaSimulation.gd`), tracking biológico (`SpeciesTracker.gd`) y shaders.

Aun así, hay varios puntos a mejorar para robustez, fidelidad científica y rendimiento.

## Mejoras y arreglos prioritarios

### 1) Coherencia de métricas UI
- **Problema detectado**: el % de `Coverage` estaba hardcodeado para 1024×1024, rompiéndose al cambiar resolución.
- **Impacto**: telemetría engañosa en 2048/4096.
- **Acción aplicada**: cálculo dinámico con `res_x * res_y`.

### 2) Métrica de diversidad inexistente
- **Problema detectado**: la etiqueta mostraba `Diversity: (Calc...)` sin cálculo real.
- **Impacto**: falta de feedback evolutivo para experimentación.
- **Acción aplicada**: se añadió una diversidad normalizada por entropía de Shannon sobre histogramas génicos.

### 3) Fuga de recursos al cambiar resolución
- **Problema detectado**: `change_resolution()` no liberaba/recreaba pipelines separables `conv_h` y `conv_v`; `_free_resources()` tampoco liberaba `tex_conv_intermediate`.
- **Impacto**: riesgo de leaks de VRAM, inconsistencias de pipeline y degradación tras varios cambios de resolución.
- **Acción aplicada**: liberación y recreación explícita de esos RIDs.

## Mejoras recomendadas (siguientes iteraciones)

### A) Fidelidad Flow-Lenia (paper ISAL 2025)
1. Exponer en UI el control de frecuencia/intensidad de mutación (ecuación 7).
2. Instrumentar explícitamente la variable `I(x_src,x_dest)` para depuración de mixing/negotiation.
3. Añadir modo de experimento con presets reproducibles (seed fija + snapshot de parámetros).

### B) Rendimiento
1. Reducir presión de CPU al parsear `analysis_buffer` (posible submuestreo adaptativo dinámico según FPS).
2. Reusar estructuras temporales en `get_species_info_at` para minimizar conversiones repetidas de byte arrays.
3. Trazas de frame-time por etapa (`signal`, `conv`, `flow`, `normalize`) para detectar cuellos.

### C) UX científica
1. Añadir panel de métricas ecológicas: riqueza de especies, equidad (Pielou), persistencia temporal.
2. Historial temporal (sparklines) para masa total/cobertura/diversidad.
3. Exportación de snapshot (parámetros + semillas + histogramas) para reproducibilidad entre corridas.

### D) Robustez
1. Validación de rangos de min/max génico en UI (evitar `min > max`).
2. Manejo explícito de valores NaN/Inf en shaders críticos (con flags de depuración).
3. Pruebas automatizadas de humo para `change_resolution()` repetido (100 iteraciones).

## Resultado esperado tras los cambios aplicados

- Métricas UI más confiables para experimentación multi-resolución.
- Visibilidad inmediata del estado de diversidad genética.
- Menor riesgo de degradación por recursos no liberados al reconfigurar la simulación.
