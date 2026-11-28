import numpy as np # type: ignore

class EnergiaSmartHome:
    def __init__(self, gamma, seed):
        self.rng = np.random.default_rng(seed)
        self.gamma = gamma

        self.niveles_bateria = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.n_niveles_bateria = len(self.niveles_bateria)
        self.n_bloques_horario = 6
        self.n_estados = self.n_bloques_horario * self.n_niveles_bateria
        
        #0: Usar energía solar/batería para la casa y almacenar excedentes
        #1: Comprar energía de la red eléctrica directamente 
        #2: Descargar la batería para cubrir picos de consumo
        self.n_acciones = 3

        #Parámetros económicos y energéticos por bloque horario
        self.tarifa = np.array([0.12, 0.15, 0.14, 0.18, 0.28, 0.25])
        self.generacion_solar = np.array([0.0, 1.2, 2.8, 2.0, 0.4, 0.0])
        self.demanda = np.array([0.8, 1.2, 1.0, 1.3, 2.2, 2.0])

        # modelo explícito P[s, a, s'] y R_model[s, a, s'] para programación dinámica
        self.P, self.R_model = self._modelo_transicion()

    def codificar_estado(self, i_bateria, i_bloque):
        return i_bateria * self.n_bloques_horario + i_bloque
    def decodificar_estado(self, estado):
        return divmod(estado, self.n_bloques_horario)

    def transicion(self, bat_idx, bloque_idx, accion):
        sgte_bloque = (bloque_idx + 1) % self.n_bloques_horario
        terminado = (bloque_idx == self.n_bloques_horario - 1)

        solar = self.generacion_solar[bloque_idx]
        demanda = self.demanda[bloque_idx]
        price = self.tarifa[bloque_idx]

        e_neta = solar - demanda

        cambio_bateria = 0
        energia_a_comprar = 0.0
        penalizacion_extra = 0.0

        if accion == 0:
            # Acción 0: Usar energía solar/batería para la casa y almacenar excedentes
            if e_neta >= 0.5:
                cambio_bateria = 1
            elif e_neta < 0.0:
                deficit = -e_neta
                aporte_bateria = 0.5 if bat_idx > 0 else 0.0
                if aporte_bateria > 0.0:
                    cambio_bateria = -1
                restante = deficit - aporte_bateria
                if restante > 0.0:
                    energia_a_comprar = restante
                    penalizacion_extra = 0.15 + 0.10 * restante
        elif accion == 1:
            # Acción 1: Comprar energía de la red eléctrica directamente 
            if e_neta >= 0.5:
                cambio_bateria = 1
            elif e_neta < 0.0:
                energia_a_comprar = max(0.0, -e_neta)
        elif accion == 2:
            # Descargar la batería para cubrir picos de consumo
            if e_neta < -1.0 and bat_idx >= 2:
                cambio_bateria = -2
                energia_a_comprar = max(0.0, -e_neta - 2.0)
            elif bat_idx > 0:
                cambio_bateria = -1
                energia_a_comprar = max(0.0, -e_neta - 1.0)
            else:
                energia_a_comprar = max(0.0, -e_neta) + 0.3
                penalizacion_extra = 0.10
        else:
            raise ValueError("Acción inválida")

        bat_tmas1 = int(np.clip(bat_idx + cambio_bateria, 0, self.n_bat - 1))

        costo = energia_a_comprar * price + penalizacion_extra

        # Transiciones estocásticas 
        s1 = self.codificar_estado(bat_tmas1, sgte_bloque)
        r1 = -costo

        if bat_tmas1 > bat_idx:
            bat_var = bat_idx
        else:
            bat_var = bat_tmas1

        s2 = self.codificar_estado(bat_var, sgte_bloque)
        r2 = -(costo + 0.05 * price)

        return [
            (0.85, s1, r1, terminado),
            (0.15, s2, r2, terminado),
        ]

    def _modelo_transicion(self):
        P = np.zeros((self.n_estados, self.n_acciones, self.n_estados), dtype=float)
        R_model = np.zeros((self.n_estados, self.n_acciones, self.n_estados), dtype=float)

        for s in range(self.n_estados):
            bat_idx, bloque_idx = self.decodificar_estado(s)
            for a in range(self.n_acciones):
                for prob, s_sgte, recompensa, _ in self.transicion(bat_idx, bloque_idx, a):
                    P[s, a, s_sgte] += prob
                    # Promedio ponderado si hubiera múltiples ramas hacia el mismo s_next
                    R_model[s, a, s_sgte] = recompensa
        return P, R_model

    def reset(self):
        bat_idx = int(self.rng.integers(0, self.n_niveles_bateria))
        return self.codificar_estado(bat_idx, 0)

    def step(self, estado, accion):
        bat_idx, bloque_idx = self.decodificar_estado(estado)
        transiciones = self.transicion(bat_idx, bloque_idx, accion)
        probs = [t[0] for t in transiciones]
        eleccion = int(self.rng.choice(len(transiciones), p=probs))
        _, next_estado, recompensa, terminado = transiciones[eleccion]
        return next_estado, recompensa, terminado 