#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:06:11 2026

@author: ndahmen
"""

# Modules- on charge les fonctions du module d'assimilation de données
import numpy as np
from module_assim import SimulationConfig, PhysicalModel, ObservationSystem, AssimEngine, Visu

# Config de simu
# On fixe la configuration de simulation ex: durée, pas de temps du model, frequence d'assimilation, si on assimile des données d'orbite
SimConfig = SimulationConfig('20150101', '20150102', obs_on_sat = False,  freq_assim = 3)

# Modèle physique
# Initialisation du modele Salammbô 1D puis on fait un run deterministe pour créer une réalité synthétique (truth)
PhyModel = PhysicalModel(SimConfig)
t, dyn = PhyModel.run_deterministic()
t, dyn_ens = PhyModel.run_ensemble()

# Observation
# Initialisation puis génération des observations bruitées à partir de la réalité synthétique sur des points fixes du maillage ou sur des orbites

ObsSys = ObservationSystem(SimConfig, PhyModel.cell_centers, PhyModel.Ec_on_cells)
obs = ObsSys.generate_observations(t, dyn)
# obs = ObsSys.generate_observations(t, dyn, np.array([0, 10, 15, 26, 29, 31, 32]))
# obs = ObsSys.generate_synthetic_mesh_observations(t, dyn, np.array([0, 10, 20, 31]))
# obs = ObsSys.generate_synthetic_satellite_observations(t, dyn)

# Assimilation
# On initialise puis on réalise l'expérience d'assimilation 
engine = AssimEngine(SimConfig, PhyModel, ObsSys)
t_b, xb, std_b, t_a, xa, std_a = engine.run_assim(obs)

# Visualisation des résultats
Viz = Visu(SimConfig, PhyModel, obs, t, dyn, t_b, xb, std_b, t_a, xa, std_a)
Viz.plot_all()

