# Esquemas de Datos Mock

## `companies.csv`

- `company_id` (int): Identificador único de empresa.
- `empresa` (string): Nombre comercial.
- `segmento_foco` (string): `bovino`, `porcino` o `mixto`.
- `tamano_empresa` (string): `grande`, `mediana`, `pequena`.

## `flows.csv`

- `flow_id` (int): Identificador único del flujo.
- `company_id` (int): FK hacia empresa.
- `empresa` (string): Nombre de empresa.
- `segmento` (string): `bovino` o `porcino`.
- `ruta_logistica` (string): Cruce fronterizo.
- `origen` / `destino` (string): Ubicaciones.
- `origin_lat` / `origin_lon` / `dest_lat` / `dest_lon` (float): Coordenadas.
- `distance_km` (float): Distancia logística.
- `transport_time_hrs` (float): Tiempo estimado de transporte.
- `congestion_base` (float): Nivel base de congestión (0-1).
- `climate_variability_base` (float): Variabilidad climática base (0-1).
- `sensitivity_temperature` (float): Sensibilidad térmica de la carga (0-1).
- `incidences_proxy` (float): Proxy histórico de incidencias (0-1).

## `trade_monthly.csv`

- `month` (date): Mes del registro.
- `flow_id`, `company_id`, `empresa`, `segmento`, `ruta_logistica`.
- `volume_tons` (float): Volumen mensual en toneladas.
- `value_usd` (float): Valor económico mensual USD.
- `shipment_count` (int): Número de envíos.
- `distance_km`, `transport_time_hrs`.
- `congestion_border` (float): Congestión mensual frontera (0-1).
- `climate_variability` (float): Variabilidad climática mensual (0-1).
- `sensitivity_temperature` (float): Sensibilidad térmica de la carga.
- `incidences_proxy` (float): Incidencias proxy.
- `incidence_probability` (float): Probabilidad de incidencia estimada.
- `incidence_flag` (int): Evento binario de incidencia (0/1).

