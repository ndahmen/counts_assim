#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
module_assim.py — Module principal d'assimilation de données pour le modèle 
de ceintures de radiation (diffusion 1D radiale + EnKF).

Contient :
    - SimulationConfig   : paramètres de simulation et d'assimilation
    - PhysicalModel      : maillage, discrétisation et évolution du modèle 1D
    - ObservationSystem  : opérateur d'observation H et génération d'obs. synthétiques
    - AssimEngine        : moteur d'assimilation EnKF (prédiction + analyse)
    - Visualization      : fonctions de visualisation 
"""


import numpy as np

import os

import warnings

import pandas as pd

import datetime as datetime

from tqdm import tqdm

from scipy.sparse.linalg import spsolve
from scipy.sparse import dia_matrix

from scipy.interpolate import interp1d

import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# --- imports --- 
############################# NE PAS TOUCHER #############################
class SimpleOrbit:
    def __init__(self, per, app, incl = 0, GM=1.541397e-6):
        self.e = (app - per)/(app+per)
        self.a = (app+per)/2
        self.incl = incl*np.pi/180
        self.period = 2*np.pi*np.sqrt(self.a**3/GM)

    def mean_anomaly(self, t):
        return t*2*np.pi/self.period

    def eccentric_anomaly(self, M=None, t=None):
        if M is None:
            M = self.mean_anomaly(t)
        #M = E - e sin(E)
        E = M.copy()
        dE = E
        eps = 1e-8
        while np.linalg.norm(dE) > eps:
            dE = (M + self.e * np.sin(E) - E)/(1-self.e*np.cos(E))
            E += dE
        return E

    def true_anomaly(self, E=None,M=None,t=None):
        if E is None:
            E = self.eccentric_anomaly(M,t)
        E = E % (2*np.pi)
        R = (1+self.e) * np.tan(E/2)**2 / (1-self.e)
        θ = 2*np.arctan(np.sqrt(R))
        θ[E>np.pi] = 2*np.pi - θ[E>np.pi]
        return θ

    def radius(self,E=None,M=None,t=None):
        if E is None:
            E = self.eccentric_anomaly(M,t)
        return self.a * (1-self.e*np.cos(E))

    def latitude(self,E=None,M=None,t=None):
        if E is None:
            E = self.eccentric_anomaly(M,t)
        r = self.radius(E)
        θ = self.true_anomaly(E)
        y = np.sin(θ)
        z = y * np.sin(self.incl)
        return np.arcsin(z)

    def L(self,E=None,M=None,t=None):
        if E is None:
            E = self.eccentric_anomaly(M,t)
        r = self.radius(E)
        lat = self.latitude(E)
        return r/np.sin(np.pi/2-lat)**2

def M2E(M,y,L,Eo,Bo):
    return -Eo + np.sqrt( Eo**2 + 2.*Eo*Bo*M/y**2/L**3)

def uni2psd(uni, Eo, c, Ec):
    clum = c * 100.
    imp2 = Ec * (Ec + 2 * Eo) / clum**2
    if uni.shape[0] == Ec.shape[0]:
        psd = uni / imp2[:, None]
    else:
        psd = uni / imp2[None, :]
    return(psd)

def psd2uni(psd, Eo, c, Ec):
    clum = c * 100.
    imp2 = Ec * (Ec + 2 * Eo) / clum**2
    if psd.shape[0] == Ec.shape[0]:
        uni = psd * imp2[:, None]
    else:
        uni = psd * imp2[None, :]
    return(uni)

def Err2(f, fref, cell_sizes):
    num = np.sqrt((cell_sizes*(f-fref)**2).sum(axis = -1))
    den = np.sqrt((cell_sizes*(fref)**2).sum(axis = -1))
    return(100 * num / den)


############################# NE PAS TOUCHER #############################


class SimulationConfig:
    
    """
    Paramètres de simulation et d'assimilation.
    Contient la configuration temporelle, physique, EnKF et d'observation.
    """
    # --- constantes physiques de classe ---
    
    mo = 9.109*1e-31 # electron mass
    clum = 2.997925*1e8 # speed of light
    MeV2J = 1.6021*1e-13 
    J2MeV = 1. / MeV2J
    Re = 6371200 # Earth's radius in m
    Eo = mo*clum**2.*J2MeV # electron rest energy
    b0 = 29867.
    mu = 500.
    
    flag_int = int(9999)
    
    def __init__(
        self,
        date_start,
        date_end,
        dt_model = 300.0,
        Ne = 100,
        sigmaB = 0.3,
        sigmaR = 0.2,
        seed = 1234,
        constant_proxies = True,
        obs_on_sat = False,
        freq_assim = 1,
        verbose = True):
        
        # --- Time ---
        self.date_start = date_start
        self.date_end = date_end
        self.dt_model = dt_model
                
        # --- Physique ---
        self.Ec_at_L4 = M2E(M = SimulationConfig.mu,
                            y = 1.0,
                            L = 4.0,
                            Eo = SimulationConfig.Eo,
                            Bo = 1e-5 * SimulationConfig.b0)
        
        # --- Assim and computing options ---
        self.Ne = int(Ne)
        self.sigmaB = sigmaB
        self.sigmaR = sigmaR
        self.sigmaKp = sigmaB
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
        self.warmup_duration = 0.0

        # --- Observations ---
        self.obs_on_sat = obs_on_sat
        self.freq_assim = freq_assim
                
        # --- Validation ---
        self._validate()
        
        # --- build time variables ---
        self._build_time_var()
        
        # --- geoidx and drivers---
        self._load_geoidx(constant_proxies)
        self._sample_drivers()
        self.apply_noise_on_drivers()
        
        # --- print config ---
        if verbose:
            self.print_config()
        
    def _build_time_var(self):
        
        """Construit les grilles temporelles (heures, dt, observations) et les compteurs (Nt, Niter, No)"""
        
        start = self.date_start
        end = self.date_end

        # build hourly grid
        Epoch_h = pd.date_range(start, end, freq = datetime.timedelta(hours = 1))
        self.Epoch_h = Epoch_h.to_pydatetime()
        
        # build dt epochs
        # sample according to dt
        Epoch_dt = pd.date_range(start, end, freq = datetime.timedelta(seconds = self.dt_model))
        self.Epoch_dt = Epoch_dt.to_pydatetime()
        
        # Number of time steps
        self.Nt = len(Epoch_dt)
        # Number of iter : size of the dt time grid removed from first (initial state) and last entry (no computation) 
        self.Niter = len(Epoch_dt) - 2

        self.dt_assim = self.dt_model * self.freq_assim
        Epoch_obs = pd.date_range(start, end, freq = datetime.timedelta(seconds = self.dt_assim))
        self.Epoch_obs = Epoch_obs.to_pydatetime()

        self.Nc = len(Epoch_obs)
    
    def _load_geoidx(self, constant_proxies):
        
        """Charge les drivers Kp/Lpp : constants (valeur fixe) ou depuis fichiers .npy"""
        
        if constant_proxies:
            Kp = 3.0
            Lmp = 7.0
            self.Kp_dt = Kp * np.ones(self.Nt)
            self.Lpp_dt = (5.6 - 0.46 * Kp) * np.ones(self.Nt)
            self.Lmp_dt = Lmp * np.ones(self.Nt)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))

            self.Kp_dt = np.load(os.path.join(script_dir, "kp.npy"))
            self.Lpp_dt = np.load(os.path.join(script_dir, "Lpp.npy"))
            self.Lmp_dt = np.load(os.path.join(script_dir, "Lmp.npy"))

            
    def _sample_drivers(self):
        
        """Échantillonne les perturbations Kp/Lpp pour les membres de l'ensemble (bruit gaussien)"""
        
        self.sample_Kp = self.sigmaKp * self.rng.standard_normal(self.Ne)
        self.sample_Lpp = 0.0 * self.sample_Kp
        self.sample_Lmp = 0.0 * self.sample_Kp
        
    def apply_noise_on_drivers(self):
        
        """Ajoute les perturbations aux drivers pour former les membres de l'ensemble"""
        
        Kp_dt_ens = self.Kp_dt[:, None] + self.sample_Kp[None, :]
        Lpp_dt_ens = self.Lpp_dt[:, None] + self.sample_Lpp[None, :]
        Lmp_dt_ens = self.Lmp_dt[:, None] + self.sample_Lpp[None, :]
        
        self.Kp_dt_ens = Kp_dt_ens
        self.Lpp_dt_ens = Lpp_dt_ens
        self.Lmp_dt_ens = Lmp_dt_ens
    
    def _validate(self):
        """ """
        if self.dt_model <= 0:
            raise ValueError("dt_model must be positif")
        if self.Ne <= 0:
            raise ValueError("Ne must be positif")
        if type(self.freq_assim) != int or self.freq_assim < 1 :
            warnings.warn("problem of assim freq -> change to 1 ", UserWarning)
            self.freq_assim = int(1)
        
    def copy(self):
        
        """Retourne une copie profonde de la configuration"""
        
        import copy
        return copy.deepcopy(self)
    
    def print_config(self):
        
        """Affiche la configuration de manière structurée"""
        
        print("=" * 60)
        print(" SIMULATION CONFIGURATION ".center(60, "="))
        print("=" * 60)
                
        # --- Time ---
        # print("\n TIME params")
        print(" TIME params ".center(60, "*"))
        print(f"  Start Date     : {self.date_start}")
        print(f"  End Date       : {self.date_end}")
        print(f"  Model dt       : {self.dt_model} s ({self.dt_model/60:.1f} min)")
        print(f"  Assimilation dt: {self.dt_assim} s ({self.dt_assim/60:.1f} min)")
        print(f"  Niter (time steps): {self.Nt - 2}")
                
        # --- Physics ---
        # print("\n PHYSICS params")
        print(" PHYSICS params ".center(60, "*"))
        print(f"  Mu        : {self.mu}")
        print(f"  Ec at L=4          : {self.Ec_at_L4:.2e} MeV")
        
        # --- EnKF ---
        print(" ENSEMBLE KALMAN FILTER params ".center(60, "*"))
        print(f"  Number of Ensembles: {self.Ne}")
        print(f"  Sigma B (Background): {self.sigmaB:.3f}")
        print(f"  Sigma R (Obs Error) : {self.sigmaR:.3f}")
        print(f"  Sigma Kp (driver sampling) : {self.sigmaKp:.3f}")

        # --- Observations ---
        print(" OBSERVATIONS params ".center(60, "*"))
        print(f"  Obs on Satellites  : {self.obs_on_sat}")
        
        # --- params ---
        print(" MISC params ".center(60, "*"))
        print(f"  Random Seed        : {self.seed}")
                        
        print("\n" + "=" * 60)



class PhysicalModel:
    """
    Modèle physique 1D radial (diffusion) sur maillage en volumes finis.
    Contient le maillage, la discrétisation spatiale et l'évolution temporelle
    (déterministe ou par ensemble).
    """
    
    def __init__(self, config: SimulationConfig):
    
        self.config = config.copy()

        self._load_mesh()

        self.Ec_on_cells = M2E(M = self.config.mu, 
                y = 1.0,
                L = self.cell_centers,
                Eo = self.config.Eo,
                Bo = 1e-5 * self.config.b0)

        self.Ec_at_Lmax = M2E(M = self.config.mu, 
                y = 1.0,
                L = self.L_all[-1],
                Eo = self.config.Eo,
                Bo = 1e-5 * self.config.b0)

    def _load_mesh(self):
        
        """
        Construit le maillage en volumes finis :
            - x_all : grille complète (faces + centres entrelacés)
            - cell_centers : centres des cellules (Nx points) - nombres d'inconnus
            - cell_faces : faces des cellules
            - J1 : 1/x² (jacobien en L)
            - capa : taille des cellules × J1 (coefficients de volume)
        """
        
        ox = float(1.0) 
        lx = float(8.0) 
        N  = 34
                            
        # finite volume mesh         
        self.L_all = np.linspace(ox, lx, 2*N+1)
        self.cell_centers = self.L_all[1:-1:2]
        self.NL = len(self.cell_centers)
        self.cell_faces = self.L_all[::2]
        self.cell_sizes = self.cell_faces[1:] - self.cell_faces[:-1]
        
        # pre compute jacobians
        self.J1 = 1.0 / self.L_all**2
        self.capa = self.cell_sizes * self.J1[1:-1:2]
    
    def build_mat(self, Dll):
        """
        Construit la matrice de diffusion en discrétisation volumes finis.
        Entrée : Dll (coefficient de diffusion sur la grille complète)
        Sortie : (diag_-1, diag_0, diag_+1) pour la matrice tridiagonale        
        """
        Dxx = self.J1 * Dll
        L_all = self.L_all
        diag0 = []
        diag1 = []
        diag_1 = []

        NL = self.NL

        h2_all = np.array([L_all[2*i+1]-L_all[2*i-1] for i in range(1,NL,1)]+[L_all[-1]-L_all[-2]])
        h3_all = np.array([L_all[1]-L_all[0]]+[L_all[2*i-1]-L_all[2*i-3] for i in range(2,NL+1,1)])

        h2 = 0.0
        h3 = 0.0

        for i in range(1,NL+1,1):
            h2 = h2_all[i-1]
            h3 = h3_all[i-1]
            A0 = (-Dxx[2*i])/(h2)
            B0 = (-Dxx[2*i-2])/(h3)

            diag0.append(A0+B0)
            if i<NL:
                A = (Dxx[2*i])/(h2)
                diag1.append(A)
            if i>1:
                B = (Dxx[2*i-2])/(h3)
                diag_1.append(B)

        diago0 = np.array(diag0)
        diago1 = np.array([0]+diag1)
        diago_1 = np.array(diag_1+[0])

        return (diago_1,diago0,diago1)

    def build_bc(self, Dll, bc_l, bc_r):
        """
        Construit le vecteur de conditions aux limites.
        Entrée : Dll, bc_l (condition gauche), bc_r (condition droite)
        Sortie : S1 (vecteur source aux limites)
        """
        
        NL = self.NL
        Dxx = self.J1 * Dll
        S1 = np.zeros(NL)

        h3 = self.L_all[1]-self.L_all[0]                                
        S1[0] = bc_l*Dxx[0]/h3

        h2 = self.L_all[-1]-self.L_all[-2]
        S1[-1] = bc_r*Dxx[-1]/h2

        return S1
    
    def eval_dll(self, L_all, Kp):
        
        if np.ndim(Kp) == 0: # Deterministe
            
            dll = 10**(0.45 * Kp - 8.985) * L_all**(10.2) / 86400.0
            
        else: # Ensembliste
                            
            dll = 10**(0.45 * Kp[None, :] - 8.985) * L_all[:, None]**(10.2) / 86400.0
            
        return(dll)
    
    def eval_tau(self, cell_centers, Kp, Lpp, Lmp):
        
        tau_in = 1e5 * np.ones(len(cell_centers))
        tau_out = 1e6 * np.ones(len(cell_centers))
        tau_drop = 1e3 * np.ones(len(cell_centers))
        
        # tau_in = 1e4
        # tau_out = 1e4
        # tau_drop = 1e4
                        
        if np.ndim(Kp) == 0: # Deterministe
            tau = 1e30 * np.ones(len(cell_centers))
            
            tau[cell_centers < Lpp] = tau_in[cell_centers < Lpp]
            tau[cell_centers >= Lpp] = tau_out[cell_centers >= Lpp]
            tau[cell_centers >= Lmp] = tau_drop[cell_centers >= Lmp]
            
        else: # Ensembliste
            tau = 1e30 * np.ones((len(cell_centers), len(Kp)))
            for m in range(len(Kp)):
                tau[cell_centers < Lpp[m], m] = tau_in[cell_centers < Lpp[m]]
                tau[cell_centers >= Lpp[m], m] = tau_out[cell_centers >= Lpp[m]]
                tau[cell_centers >= Lmp[m], m] = tau_drop[cell_centers >= Lmp[m]]

        return(tau)
    
    def eval_bc(self, Ec, Kp):
        
        if np.ndim(Kp) == 0: # Deterministe
            
            bc = 1e30
            
        else: # Ensembliste
                    
            bc = 1e30 * np.ones(len(Kp))
            
        return(bc)


    def step(self, state, dt, dll, tau, bc_l, bc_r):
        """
        Un pas de temps implicite du modèle.
        Entrée : state (état actuel), dt (pas de temps), dll (diffusion),
                 tau (temps de perte), bc_l, bc_r (conditions aux limites)
        Sortie : state_new (état après un pas)
        """
        NL = self.NL
        dm1, d0, dp1 = self.build_mat(dll)
        
        itau = 1.0 / tau 
    
        (dm1, d0, dp1) = self.build_mat(dll)
        itau = 1/tau if (tau is not None) else 0
        data = [-dt*dm1, self.capa-dt*d0 + self.capa*itau*dt, -dt*dp1]
        offsets = [-1,0,1]
        E = dia_matrix((data, offsets), shape=(NL, NL))
        S1 = self.build_bc(dll, bc_l, bc_r)
        RHS = dt*S1 + self.capa * state
        state_new = spsolve(E,RHS)

        
        return state_new
    
    def step_ensemble(self, states, dt, dll_ens, tau_ens, bc_l, bc_r_ens):
        """
        Un pas de temps pour l'ensemble des membres (mode séquentiel).
        Entrée : states (Nx, Ne), dll_ens (Nx, Ne), tau_ens (Ne,)
        Sortie : states_new (Nx, Ne)
        """
        res = []
        
        Ne = self.config.Ne
                
        # Serial computing
        for member in range(Ne):
            state = self.step(
                states[:, member], 
                dt, 
                dll_ens[:, member], 
                tau_ens[:, member], 
                bc_l, 
                bc_r_ens[member], 
            )
            res.append(state)
        
        # Prallel computing
        # args_list = [
        #     (states.T[m], dt, dll_ens.T[m], tau, bc_l, bc_r)
        #     for m in range(Ne)
        # ]
            
        # with Pool(processes=40) as pool:
        #     res = pool.starmap(self.step, args_list)
                    
        return np.array(res).T 
    
    def run_deterministic(self, initial_state = None):
        """
        Lance la propagation déterministe du modèle sur toute la période.
        Sortie : (time_dyn, state_dyn) — temps et états à chaque pas
        """
        
        Nt = self.config.Nt
        
        dt = self.config.dt_model
        
        Kp_dt = self.config.Kp_dt
        Lpp_dt = self.config.Lpp_dt
        Lmp_dt = self.config.Lmp_dt

        state_dyn = np.ones(((Nt - 1), self.NL))
        time_dyn = []
        
        # Define empty initial state or filled
        if initial_state is None:
            initial_state = np.ones(self.NL)
            
        # Save initial state
        time_dyn.append(self.config.Epoch_dt[0])
        state_dyn[0] = initial_state
        
        for s in tqdm(range(1, Nt - 1 , 1), desc = ' [MODEL] : RUN - determinstic'):
            
            Kp = Kp_dt[s+1]
            Lpp = Lpp_dt[s+1]
            Lmp = Lmp_dt[s+1]
            
            dll = self.eval_dll(self.L_all, Kp)
            tau = self.eval_tau(self.cell_centers, Kp, Lpp, Lmp)
            bc = self.eval_bc(self.Ec_at_Lmax, Kp)
                        
            time_dyn.append(self.config.Epoch_dt[s])
            state_dyn[s] = self.step(state_dyn[s-1], dt, dll, tau, 1.0, bc)
        
        return np.array(time_dyn), state_dyn

    
    def run_ensemble(self, initial_state = None):
        """
        Lance la propagation ensebmliste du modèle sur toute la période.
        Sortie : (time_dyn, state_dyn_ens) — temps et états à chaque pas
        """
        
        Nt = self.config.Nt
        
        dt = self.config.dt_model
        
        Kp_dt_ens = self.config.Kp_dt_ens
        Lpp_dt_ens = self.config.Lpp_dt_ens
        Lmp_dt_ens = self.config.Lmp_dt_ens
    
        state_dyn_ens = np.ones(((Nt - 1), self.NL, self.config.Ne))
        time_dyn = []
        
        # Define empty initial state or filled
        if initial_state is None:
            initial_state = np.ones((self.NL, self.config.Ne))
            
        # Save initial state
        time_dyn.append(self.config.Epoch_dt[0])
        state_dyn_ens[0] = initial_state
        
        for s in tqdm(range(1, Nt - 1 , 1), desc = ' [MODEL] : RUN - ensemble'):
            
            Kp_ens = Kp_dt_ens[s+1]
            Lpp_ens = Lpp_dt_ens[s+1]
            Lmp_ens = Lmp_dt_ens[s+1]
            
            dll_ens = self.eval_dll(self.L_all, Kp_ens)
            tau_ens = self.eval_tau(self.cell_centers, Kp_ens, Lpp_ens, Lmp_ens)
            bc_ens = self.eval_bc(self.Ec_at_Lmax, Kp_ens)
                        
            time_dyn.append(self.config.Epoch_dt[s])
            state_dyn_ens[s] = self.step_ensemble(state_dyn_ens[s-1], dt, dll_ens, tau_ens, 1.0, bc_ens)
        
        return np.array(time_dyn), state_dyn_ens

class ObservationSystem:
    """
    Système d'observation : opérateur d'observation H et génération
    d'observations synthétiques (maillage ou satellite).
    """
    
    def __init__(self, config: SimulationConfig, cell_centers: np.ndarray, Ec_on_cells : np.ndarray):
        """
        Init obs system
        """
        
        self.config = config.copy()
        self.cell_centers = cell_centers
        self.Ec_on_cells = Ec_on_cells
        self.NL = len(cell_centers)
        
    def generate_observations(self, time_dyn, state_dyn, obs_indexes = None):
        """
        Génère les observations synthétiques.
        Appelle generate_synthetic_mesh_observations ou
        generate_synthetic_satellite_observations selon obs_on_sat.
        obs_indexes (optionnel) : indices des points du maillage à observer.
        """
        if self.config.obs_on_sat:
            return self.generate_synthetic_satellite_observations(time_dyn, state_dyn)
        else:
            return self.generate_synthetic_mesh_observations(time_dyn, state_dyn, obs_indexes)
                
                        
    # --- Observation Operator ---
    
    def H(self, state):
        """
        Opérateur d'observation : transforme l'état du modèle (psd) en observation (psd, flux)
        À modifier pour intégrer les comptages.
        """
        # in case of identity
        # observation = state     
        # in case of flux
        observation = psd2uni(state, self.config.Eo, self.config.clum, self.Ec_on_cells)
        return observation

    def H_1(self, obs):
        """
        Opérateur inverse : espace d'observation vers état du modèle.
        """    
        # in case of identity
        # state = obs
        # in case of flux
        state = uni2psd(obs, self.config.Eo, self.config.clum, self.Ec_on_cells)
        return state

    def generate_synthetic_mesh_observations(self, time_dyn, state_dyn, input_obs_indexes):
        """
        Observations sur un sous-ensemble du maillage (tout le maillage par défaut) 
        à chaque temps d'observation.
        """
        
        Epoch_obs = self.config.Epoch_obs
        Nc = self.config.Nc
        sigmaR = self.config.sigmaR
        rng = self.config.rng
        flag_int = self.config.flag_int
        
        # L grid node indexes (p) and values to assimilate on

        if input_obs_indexes is None:
            input_obs_indexes = np.arange(self.NL)            
        
        # Number of observation locations on mesh
        No = len(input_obs_indexes)
        input_obs_locations = self.cell_centers[input_obs_indexes]
        
        # We suppose observations are evaluated on fixed mesh points (no time dependence)
        # Repeat cell centers according to time
        obs_indexes = np.repeat(input_obs_indexes[np.newaxis, :], Nc, axis = 0)
        obs_locations = np.repeat(input_obs_locations[np.newaxis, :], Nc, axis = 0)
        
        obs_values_xarr = xr.DataArray(
            self.H(state_dyn),
            coords={"time": pd.to_datetime(time_dyn), 'L' : self.cell_centers},
            dims=["time", 'L'])
        
        # Interpolate on time
        print('\n [OBS] : GEN SYNTH MESH')
        obs_truth = obs_values_xarr.interp(time = Epoch_obs,
                                                      L = obs_locations[0],
                                                      method = "nearest",).to_numpy()
        
        # Apply noise on truth observations to generate "real" noisy obs + apply observation operator
        y = np.zeros_like(obs_truth)
        for k in range(Nc):
            # A log10 gaussian error is added to mimic sensor error
            y[k] = np.power(10, np.log10(obs_truth[k]) + sigmaR * rng.standard_normal(No))
                        
        return {'Epoch_obs' : Epoch_obs,
                'obs_indexes' : obs_indexes,
                'obs_positions' : obs_locations,
                'obs_truth' : obs_truth,
                'obs_noisy' : y}
    
    def generate_synthetic_satellite_observations(self, time_dyn, state_dyn):
        """
        Observations satellitaires sur orbites (geo, gps, heo).
        Nécessite SimpleOrbit (module externe).
        """

        Epoch_obs = self.config.Epoch_obs
        Nc = self.config.Nc
        sigmaR = self.config.sigmaR
        rng = self.config.rng
        
        flag_int = self.config.flag_int
        
        obs_list = [{'orbit' : 'geo'}, {'orbit' : 'gps'}, {'orbit' : 'heo'}]

        t_obs = np.array([(Epoch_obs[k] - Epoch_obs[0]).total_seconds() for k in range(Nc)])

        # generate satellite L trace

        geo = SimpleOrbit(6.61,6.61,0.)
        heo = SimpleOrbit(1.09,5.79,10.)
        gps = SimpleOrbit(4.2,4.21,55.)

        obs_positions = np.zeros((Nc, len(obs_list)))

        for s, obs in enumerate(obs_list):
            if obs['orbit'] == 'geo':
                obs_positions[:, s] = geo.L(t = t_obs)
            elif obs['orbit'] == 'gps':
                obs_positions[:, s] = gps.L(t = t_obs)
            elif obs['orbit'] == 'heo':
                obs_positions[:, s] = heo.L(t = t_obs)
            else:
                raise ValueError(f"unknown {obs['orbit']} sat provided")
        
        # filter positions outside the model's grid
        obs_positions[ obs_positions < self.cell_centers[0]] = np.nan
        obs_positions[ obs_positions > self.cell_centers[-1]] = np.nan
        
        obs_dyn_xarr = xr.DataArray(
        self.H(state_dyn),
        coords={"time": pd.to_datetime(time_dyn), 'L' : self.cell_centers},
        dims=["time", 'L'])
        
        # Interpolate on time
        obs_values_xarr_interp = obs_dyn_xarr.interp(time = Epoch_obs, method = 'nearest').to_numpy()
        
        obs_indexes = np.full_like((obs_positions), flag_int, dtype = int)
        obs_truth = np.full_like((obs_positions), np.nan, dtype = float)
        y = np.full_like((obs_truth), np.nan, dtype = float)
        
        # find closest point on grid and interpolate
        for k in tqdm(range(Nc), ' [OBS] : GEN SYNTH SAT OBS'):
            f = interp1d(
                self.cell_centers,
                obs_values_xarr_interp[k],
                kind="nearest",
                bounds_error=False,
                fill_value=np.nan
            )
            
            # Locate closest point for each satellite
            for s, pos_sat in enumerate(obs_positions[k]):
                if np.isnan(pos_sat):
                    obs_indexes[k, s] = int(flag_int)

                else:
                    # find closest sat position
                    index_sat_mesh = np.argmin(abs(self.cell_centers - pos_sat))
                    flux_sat_mesh  = f(self.cell_centers[index_sat_mesh])
                    
                    if np.isnan(flux_sat_mesh):
                        obs_positions[k, s] = int(flag_int)
                    else:
                        obs_indexes[k, s] = index_sat_mesh
                        obs_truth[k, s] = flux_sat_mesh
                        y[k, s] = np.power(10, np.log10(obs_truth[k, s]) + sigmaR * rng.standard_normal(1))
                
        return {'Epoch_obs' : Epoch_obs,
                'obs_indexes' : obs_indexes,
                'obs_positions' : obs_positions,
                'obs_truth' : obs_truth,
                'obs_noisy' : y}

class AssimEngine:
    
    """
    Moteur d'assimilation EnKF (Ensemble Kalman Filter).
    """
    
    def __init__(self,
                 config : SimulationConfig,
                 model : PhysicalModel,
                 obs_sys : ObservationSystem):
        

        # Copy objects
        self.config = config.copy()
        self.model = model
        self.obs_sys = obs_sys
        
        # Time vars
        self.Nt = self.config.Nt
        self.Epoch_dt = self.config.Epoch_dt
        self.Epoch_obs = self.config.Epoch_obs
        self.Nc = self.config.Nc
        self.has_obs = np.isin(self.Epoch_dt, self.Epoch_obs)

        # Obs operator
        self.H = self.obs_sys.H
        self.H_1 = self.obs_sys.H_1
        
        # Assim vars and options
        self.Ne = config.Ne
        self.sigmaB = config.sigmaB
        self.sigmaR = config.sigmaR
        self.sigmaKp = config.sigmaKp
        self.seed = config.seed
                        
        self.NL = model.NL
        
        self.Nstate = self.NL
        self.flag_int = self.config.flag_int
                
        self.xb_history = []
        self.xa_history = []
        
        self.xb_std_history = []
        self.xa_std_history = []
        
    def init_background(self, initial_state = None):
        """
        Initialisation de l'ensemble de propagation à l'état initial + bruit
        Eb contient l'état propagé échantilloné sur l'ensemble - exprimé dans l'espace log10
        """
        
        Ne = self.Ne
        NL = self.NL
        Nstate = self.NL
        rng = self.config.rng
        
        un = np.ones(Ne)
        
        if initial_state is None:
            initial_state = np.ones(NL)

        initial_homog_log10 = np.log10(initial_state)
        
        Eb =  np.zeros((Nstate, Ne))
        
        # Init homogenous part - uniform
        # Eb = np.outer(log_initial_homog, un)
        
        # Init homog part - with log gaussian noise
        Eb = np.outer(initial_homog_log10, un) + np.outer(self.sigmaB, 
                                                                 un)* rng.standard_normal((self.NL,
                                                                                                self.Ne))
        return Eb
    
    def forecast(self, Eb, dt, dll_ens, tau_ens, bc_l, bc_r_ens):
        """
        Étape de prédiction : propagation de l'ensemble d'état.
        Sortie : Eb (NL, Ne) après un pas de modèle
        """
        # Retour vers l'espace lin
        states = np.power(10, Eb)
                
        states_new = self.model.step_ensemble(
            states = states,
            dt=dt,
            dll_ens=dll_ens,
            tau_ens=tau_ens,
            bc_l=1.0,
            bc_r_ens=bc_r_ens,
        )
        
        # Aller vers l'espace log10
        Eb = np.log10(states_new)
                
        return Eb            
    
    def analysis(self, Eb, obs_at_epoch):
        """
        Étape d'analyse : mise à jour de l'ensemble par les observations.
        Assimilation sur le log10 de l'etat
        Algorithme EnKF standard :
            - Calcul des anomalies Ab (modèle) et Ao (observations)
            - Gain de Kalman K = BHt @ inv(HBHt + sigmaR²·I)
            - Observations perturbées Y
            - Mise à jour : Ea = Eb + K @ (Y - Z)
        Entrée : Eb (NL, Ne), obs_at_epoch = (obs_position, obs_values)
        Sortie : Ea (NL, Ne) après analyse
        """
                
        Ne = self.Ne
        sigmaR = self.sigmaR
        rng = self.config.rng

        un = np.ones(Ne)
                
        xb = np.mean(np.power(10, Eb), axis=1)
        
        Ab = Eb - np.outer(np.log10(xb), un)
        
        obs_position_at_epoch = obs_at_epoch[0]
        obs_values_at_epoch = obs_at_epoch[1]
        
        mask = obs_position_at_epoch != self.flag_int 
        obs_position_at_epoch = obs_position_at_epoch[mask]
        obs_values_at_epoch = obs_values_at_epoch[mask]
        
        No = len(obs_position_at_epoch)
        I = np.eye(No)

        Z = np.log10(self.H(np.power(10, Eb)))[obs_position_at_epoch]
    
        # Compute xo the mean of Z and Ao its anomaly (Ao = [Z_1 - xo, Z_2 - xo, ..., Z_Ne - xo])
        xo = np.mean(Z, axis=1)
        Ao = Z - np.outer(xo, un)
            
        BHt = Ab @ Ao.T / (Ne-1.)  # Use empirical covariance estimation with anomalies matrixes Ab and Ao (see tutorial notes)
        HBHt = Ao @ Ao.T / (Ne-1.) # Use empirical covariance estimation with anomalies matrixes
        
        K =  BHt @ np.linalg.inv(HBHt + sigmaR**2*I)
        
        y = obs_values_at_epoch
    
        # Produce an ensemble of perturbed observations
        Y = np.outer(np.log10(y), un) + sigmaR * rng.standard_normal((No, Ne)) # Same method as the background initial ensemble sampling
    
        # Compute the analysis of each ensemble member    
        Ea = Eb + K @ (Y - Z)  # Analysis update (see EnKF Algorithm)
        
        return(Ea)

        
    def run_assim(self, obs_dict, initial_state = None):
        """
        Boucle principale d'assimilation sur toute la période.
        Pour chaque pas : forecast puis (analysis si observations disponibles).
        """

        Epoch_dt = self.Epoch_dt
        Epoch_obs = self.Epoch_obs
        
        dt = self.config.dt_model
        
        Nt = self.Nt
        
        Nc = self.Nc
        
        NL = self.NL

        t_b = []
        xb = np.zeros((Nt - 1, NL))
        std_b = np.zeros((Nt - 1, NL))

        t_a = []
        xa = []
        std_a = []
                                
        Kp_dt_ens = self.config.Kp_dt_ens
        Lpp_dt_ens = self.config.Lpp_dt_ens
        Lmp_dt_ens = self.config.Lmp_dt_ens
                
        # Initialization
        Eb = self.init_background(initial_state)
        
        t_b.append(Epoch_dt[0])
        xb[0] = np.median(np.power(10, Eb), axis = 1)
        std_b[0] = np.std(Eb, axis = 1)

        t_a.append(Epoch_obs[0])
        xa.append(np.median(np.power(10, Eb), axis = 1))
        std_a.append(np.std(Eb, axis = 1))
                
        for k_time in tqdm(range(1, Nt - 1 , 1), desc = ' [AssimEngine] : RUN - Assimilation'):
            # remove initial epoch and last epoch
            
            # Eval phy ens
            Kp_ens = Kp_dt_ens[k_time + 1]
            Lpp_ens = Lpp_dt_ens[k_time + 1]
            Lmp_ens = Lmp_dt_ens[k_time + 1]
            
            dll_ens = self.model.eval_dll(self.model.L_all, Kp_ens)
            tau_ens = self.model.eval_tau(self.model.cell_centers, Kp_ens, Lpp_ens, Lmp_ens)
            bc_ens = self.model.eval_bc(self.model.Ec_at_Lmax, Kp_ens)
            
            # Forecast
            Eb = self.forecast(Eb, dt, dll_ens, tau_ens, 1.0, bc_ens)

            t_b.append(Epoch_dt[k_time])
            xb[k_time] = np.median(np.power(10, Eb), axis = 1)
            std_b[k_time] = np.std(Eb, axis=1)
            
            # Analysis if obs avail
            if self.has_obs[k_time]:

                k_assim = list(Epoch_obs).index(Epoch_dt[k_time])
                t_a.append(Epoch_obs[k_assim])
                
                obs_at_epoch = (obs_dict['obs_indexes'][k_assim], obs_dict['obs_noisy'][k_assim])
                Ea = self.analysis(Eb, obs_at_epoch)

                xa.append(np.median(10**Ea, axis = 1))
                std_a.append(np.std(Ea, axis=1))
                
                Eb = np.copy(Ea)

        return t_b, xb, std_b, t_a, np.array(xa), np.array(std_a)


class Visu:
    
    """
    Class for visualisation
    """
    
    def __init__(self,
                 config : SimulationConfig,
                 model : PhysicalModel, obs,
                 t_dyn , dyn,
                 t_b, xb, std_b,
                 t_a, xa, std_a):
        
        self.config = config.copy()
        self.model = model
        self.obs = obs
        self.t_dyn = t_dyn
        self.dyn = dyn
        self.t_b = t_b
        self.t_a = t_a
        self.xb = xb
        self.xa = xa
        self.std_b = std_b
        self.std_a = std_a
    
    def Ltime_scatter(self, time, Ls, data, title = None, ax = None, vmin = None, vmax = None):
        
        if ax is None:
            fig, ax = plt.subplots()

        if vmax is None:
            vmax = data.max()
        if vmin is None:
            vmin = data.max()/1e5
        
        for p in range(len(Ls[0])):        
            c = ax.scatter(x = time, y = Ls[:, p], c = data[:, p], norm = LogNorm(vmin = vmin, vmax = vmax) , cmap = 'jet')

        plt.colorbar(c)
        ax.set_title(title)
        ax.set_xlim(self.config.Epoch_dt[0], self.config.Epoch_dt[-1])
    
    def Ltime(self, time, Ls, data, title = None, ax = None, isLogNorm = True, overplot = False, isratio = False, vmin = None, vmax = None):
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if isLogNorm:
            
            if isratio:
                cmap = 'bwr'
                if vmax is None:
                    vmax = 1e1
                if vmin is None:
                    vmin = 1e-1

            else:
                cmap = 'jet'
                if vmax is None:
                    vmax = data.max()
                if vmin is None:
                    vmin = data.max()/1e5
            
            norm = LogNorm(vmin = vmin, vmax = vmax)
            
        else:
            cmap = 'viridis'
            if vmax is None:
                vmax = 1
            if vmin is None:
                vmin = 0
                
            norm = Normalize(vmin = vmin, vmax = vmax)
                
        c = ax.pcolormesh(time, Ls, data, norm = norm, cmap = cmap)
        
        if overplot:
            ax.plot(self.config.Epoch_dt, self.config.Lpp_dt, '--w')
            ax.plot(self.config.Epoch_dt, self.config.Lmp_dt, '-w')
       
        plt.colorbar(c)
        ax.set_title(title)
        ax.set_xlim(self.config.Epoch_dt[0], self.config.Epoch_dt[-1])
        ax.set_ylim(Ls[0], Ls[-1])

        
    def plottime(self, time, f, label = None, ax = None, isLog = True):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(time, f, '-o', label = label)
        if isLog:
            ax.set_yscale('log')
    
    def proxiestime(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        self.plottime(self.config.Epoch_dt, self.config.Kp_dt, label = 'Kp', ax = ax, isLog = False)
        ax.set_title('proxies')
        ax.legend()
    
    def xaxbobstime_at_L(self, index, ax = None):
        
        NL = self.model.NL
        
        if ax is None:
            fig, ax = plt.subplots()

        if index <= NL - 1:
            isLog = True
            mask = (self.obs['obs_indexes'] == index)
            first_occ = np.argmax(mask, axis=1)
            first_occ[~mask.any(axis=1)] = -1

            Epob = []
            xob = []
            for t in range(len(self.obs['obs_indexes'])):
                if first_occ[t] == -1:
                    continue
                else:
                    Epob.append(self.config.Epoch_obs[t])
                    xob.append(self.obs['obs_noisy'][t, first_occ[t]])
            
            self.plottime(Epob, xob, label = 'obs', ax = ax)

            ax.set_title(f'xa VS xb VS obs at L = {self.model.cell_centers[index]:.2f}')
        else:
            isLog = False
            ax.set_title(f'xa VS xb for aug param = {list(self.config.aug_params.keys())[index - NL]}')
            
        self.plottime(self.t_b, self.xb[:, index], label = 'xb', ax = ax, isLog=isLog)
        self.plottime(self.t_a, self.xa[:, index], label = 'xa', ax = ax, isLog=isLog)
        
        ax.legend()

    def stdtime_at_L(self, index_L, ax = None):
        
        if ax is None:
            fig, ax = plt.subplots()
        self.plottime(self.t_b, self.std_b[:, index_L], label = 'std_xb', ax = ax, isLog = False)
        self.plottime(self.t_a, self.std_a[:, index_L], label = 'std_xa', ax = ax, isLog = False)
        ax.set_title(f'log std_b VS log std_a at L = {self.model.cell_centers[index_L]:.2f}')
        ax.legend()
    
    def plot_error_global(self, ax = None):
        if ax is None:
            fig, ax = plt.subplots()
        
        # plot err xb
        self.plottime(self.t_b, Err2(np.log10(self.xb), np.log10(self.dyn), self.model.cell_sizes), label = 'xb', ax = ax)
        # plot err xa
        self.plottime(self.t_a, Err2(np.log10(self.xa), np.log10(self.dyn[::self.config.freq_assim]), self.model.cell_sizes), label = 'xa', ax = ax)
        # plot err obs
        
        if not(self.config.obs_on_sat):
            self.plottime(self.t_dyn[::self.config.freq_assim], Err2(np.log10(self.obs['obs_noisy'][:-1]), np.log10(self.dyn[::self.config.freq_assim]), self.model.cell_sizes), label = 'obs', ax = ax)
        
        ax.set_title('Err2 on log10')
        ax.legend()
        
    def plot_ratios(self):
        
        fig, ax = plt.subplots(3, 1)
                
        # Plot ratio xb/dyn
        self.Ltime(self.t_b, self.model.cell_centers, (self.xb/self.dyn).T, title = 'xb/dyn', ax = ax[0], isratio=True)
        ax[0].set_xticks([])
        self.Ltime(self.t_a, self.model.cell_centers, (self.xa/self.dyn[::self.config.freq_assim]).T, title = 'xa/dyn', ax = ax[1], isratio=True)
        ax[1].set_xticks([])
        if not(self.config.obs_on_sat):
            self.Ltime(self.t_dyn[::self.config.freq_assim], self.model.cell_centers, (self.obs['obs_noisy'][:-1]/self.dyn[::self.config.freq_assim]).T, title = 'obs/dyn', ax = ax[2], isratio=True)
            ax[2].set_xticks([])
        
        fig.autofmt_xdate()
        
    
    def plot_all(self):
        
        fig, ax = plt.subplots(3, 2)
        
        # Plot xb - xb std
        self.Ltime(self.t_b, self.model.cell_centers, self.xb.T, title = 'forecast', ax = ax[0, 0], overplot = True)
        ax[0, 0].set_xticks([])
        self.Ltime(self.t_b, self.model.cell_centers, self.std_b.T, title = 'logstd - forecast', ax = ax[0, 1], isLogNorm=False)
        ax[0, 1].set_xticks([])

        # plot xa - xa std
        self.Ltime(self.t_a, self.model.cell_centers, self.xa.T, title = 'analysis - corrected state', ax = ax[1, 0])
        ax[1, 0].set_xticks([])
        self.Ltime(self.t_a, self.model.cell_centers, self.std_a.T, title = 'logstd - corrected state', ax = ax[1, 1],isLogNorm=False)
        ax[1, 1].set_xticks([])

        # plot obs
        self.Ltime_scatter(self.obs['Epoch_obs'], self.obs['obs_positions'], self.obs['obs_noisy'], title = 'obs noisy', ax = ax[2, 0], vmin = 1e2, vmax = 1e8)
        
        # plot ratio xa/dyn
        self.Ltime(self.t_a, self.model.cell_centers, (self.xa/self.dyn[::self.config.freq_assim]).T, title = 'ratio a/truth', ax = ax[2, 1], isratio = True,)
        
        fig.autofmt_xdate()