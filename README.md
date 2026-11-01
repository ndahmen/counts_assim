## Description

Module pour implémenter une expérience d'assimilation de données sur un modèle
simplifié de ceintures de radiation (diffusion 1D radiale) avec un filtre de
Kalman d'ensemble (EnKF).
## Structure

**`experience.py`**  : pour designer et lancer l'expérience
**`module_assim.py`**  : petite librairie de fonctions et d'objets pour réaliser l'expérience

| Classe              | Rôle                                                                |
| ------------------- | ------------------------------------------------------------------- |
| `SimulationConfig`  | Paramètres de simulation (dates, pas de temps, EnKF, observations)  |
| `PhysicalModel`     | Modèle physique 1D (diffusion) sur maillage en volumes finis        |
| `ObservationSystem` | Opérateur d'observation H et génération d'observations synthétiques |
| `AssimEngine`       | Moteur EnKF : forecast, analysis, boucle d'assimilation             |
| `Vizualisation`     | Fonctions pour affichage des résultats                              |

## Exemple d'utilisation rapide

```python
from module_assim import SimulationConfig, PhysicalModel, ObservationSystem, AssimEngine

# 1. Configuration
config = SimulationConfig(
    date_start="20150301",
    date_end="20150305",
    dt_model=300.0,    # pas de modèle : 5 min
    Ne=100,            # taille de l'ensemble
    sigmaB=0.2,        # écart-type du bruit du modèle
    sigmaR=0.1,        # écart-type des erreurs d'observation
    freq_assim=1,      # fréquence d'assimilation (en pas de modèle)
    obs_on_sat=False,  # observations sur maillage (True = satellites)
)

# 2. Modèle physique
model = PhysicalModel(config)

# 3. Observations
obs_sys = ObservationSystem(config, model.cell_centers)

# 4. Moteur d'assimilation
engine = AssimEngine(config, model, obs_sys)

# 5. Lancer l'expérience
xb, xa = engine.run_assim(obs_dict)
```

## Concepts clés

### Modèle physique
Modèle 1D d'équation de diffusion 1D en coordonnée L :  
∂f/∂t = (1/L²) ∂/∂L (L² D ∂f/∂L) − f/τ
avec : 
- f la fonction de distribution appelée aussi psd, proportionnelle au flux
- D (noté aussi DLL) = D(Kp, L) - coefficient de diffusion pilotée par Kp, un proxy temporel
- τ (temps de perte).

### Opérateur d'observation H
- On assimile dans l'espace log10 (on suppose que l'état des ceintures (la psd) est log gaussien).
- On suppose qu'on observe le log10 de la psd
- L'opérateur d'observation permet donc de passer de l'état du modèle à l'état des observations : `H(state) = log10(state)`
- ==C'est le point de départ des modifications pour implémenter un opérateur personnalisé permettant d'assimiler des comptages (il faut passer par le flux).==
### Erreurs d'observations
- Pour l'instant, les observations sont supposés être bruitées par un bruit gaussien avec une matrice de covariance diagonale (sigmaR)
- ==On cherchera ici à étudier l'implémentation de bruits d'observations plus complexes (matrice de covariance avec corrélations spatiales en L)==

