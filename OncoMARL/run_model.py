import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ============================================================
# 1. Définition du modèle (doit correspondre à train_dqn_v2.py)
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)

# ============================================================
# 2. Configuration et Chargement
# ============================================================
# Noms des actions pour un affichage clair
ACTIONS = {
    0: "CONTINUE_CURRENT_PATHWAY",
    1: "REASSESS_CURRENT_PATHWAY",
    2: "SWITCH_PATHWAY",
    3: "MULTIDISCIPLINARY_REASSESSMENT",
}

# Définition des chemins
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "oncomarl_v2" / "double_dqn_v2.pt"

print("=" * 60)
print("TEST DU MODÈLE ONCOMARL V2")
print("=" * 60)

# Vérification de l'existence du modèle
if not MODEL_PATH.exists():
    print(f"ERREUR : Le modèle n'a pas été trouvé au chemin {MODEL_PATH}")
    print("Veuillez d'abord exécuter 'train_dqn_v2.py' pour générer le modèle.")
    exit(1)

# Chargement du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)

state_dim = checkpoint["state_dim"]
action_dim = checkpoint["action_dim"]

model = QNetwork(state_dim, action_dim).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Mode évaluation (désactive le dropout, etc.)

print(f"Modèle chargé avec succès depuis : {MODEL_PATH}")
print(f"Appareil utilisé : {device}\n")

# ============================================================
# 3. Définition de 10 exemples de test (États Synthétiques)
# Format : [tumor_burden, toxicity, uncertainty, progression, visit_progress]
# ============================================================
examples = [
    # Ex 1 : Début de traitement idéal (Faible tumeur, faible toxicité)
    {"desc": "Patient stable au début", "state": [0.2, 0.1, 0.1, 0.0, 0.1]},
    
    # Ex 2 : Grosse tumeur, mais sans toxicité
    {"desc": "Forte charge tumorale, tolérance OK", "state": [0.8, 0.1, 0.2, 0.0, 0.2]},
    
    # Ex 3 : Toxicité sévère nécessitant une attention
    {"desc": "Forte toxicité, faible tumeur", "state": [0.3, 0.8, 0.2, 0.0, 0.3]},
    
    # Ex 4 : Forte incertitude médicale
    {"desc": "Forte incertitude clinique", "state": [0.5, 0.2, 0.9, 0.0, 0.4]},
    
    # Ex 5 : Progression confirmée de la maladie (progression = 1.0)
    {"desc": "Progression de la maladie observée", "state": [0.7, 0.4, 0.3, 1.0, 0.5]},
    
    # Ex 6 : Fin de parcours de soins
    {"desc": "Dernières visites, état moyen", "state": [0.6, 0.5, 0.4, 0.0, 0.9]},
    
    # Ex 7 : Cas critique (Tout est élevé)
    {"desc": "Cas critique (Tumeur+, Toxicité+, Prog+)", "state": [0.9, 0.9, 0.8, 1.0, 0.8]},
    
    # Ex 8 : Fin de traitement avec toxicité élevée
    {"desc": "Fin de traitement, toxicité sévère", "state": [0.2, 0.8, 0.2, 0.0, 0.9]},
    
    # Ex 9 : Cas complètement incertain en début de traitement
    {"desc": "Début de traitement, très incertain", "state": [0.4, 0.1, 0.95, 0.0, 0.0]},
    
    # Ex 10 : Situation intermédiaire classique
    {"desc": "Situation intermédiaire", "state": [0.5, 0.5, 0.5, 0.0, 0.5]},
]

# ============================================================
# 4. Évaluation et Affichage des Résultats
# ============================================================
with torch.no_grad():
    for i, ex in enumerate(examples, start=1):
        state_array = np.array(ex["state"], dtype=np.float32)
        state_tensor = torch.tensor(state_array, device=device).unsqueeze(0)
        
        # Passage dans le réseau pour obtenir les Q-values
        q_values = model(state_tensor).cpu().numpy()[0]
        
        # Meilleure action (celle avec la plus haute valeur Q)
        best_action_idx = np.argmax(q_values)
        best_action_name = ACTIONS[best_action_idx]
        
        print(f"--- Exemple {i} : {ex['desc']} ---")
        print(f"État en entrée : [Tumeur: {state_array[0]:.2f}, Toxicité: {state_array[1]:.2f}, "
              f"Incertitude: {state_array[2]:.2f}, Progression: {state_array[3]}, Visite: {state_array[4]:.2f}]")
        print("Valeurs Q estimées :")
        for action_idx, action_name in ACTIONS.items():
            # Afficher une étoile à côté de l'action choisie
            marker = "★" if action_idx == best_action_idx else " "
            print(f"  {marker} {action_name.ljust(30)} : {q_values[action_idx]:.4f}")
        
        print(f">> ACTION CHOISIE : {best_action_name}")
        print("-" * 60)


        