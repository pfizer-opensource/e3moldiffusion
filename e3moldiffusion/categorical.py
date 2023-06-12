import torch

def get_uniform_transition(alpha_bar_t, K):
  stay_prob = torch.eye(K) * alpha_bar_t
  diffuse_prob = (1.0 - alpha_bar_t) / K * torch.ones(K**2).view(K, K)
  Q_t = stay_prob + diffuse_prob
  return Q_t

def get_terminal_transition(alpha_bar_t, K, e):
  terminal_state = torch.zeros(K)
  terminal_state[e] = 1.0
  stay_prob = torch.eye(K) * alpha_bar_t
  diffuse_prob = (1.0 - alpha_bar_t) * ( torch.ones(K, 1) * (terminal_state.unsqueeze(0)) )
  Q_t = stay_prob + diffuse_prob
  return Q_t


drugs_with_h = {
    'atom_encoder': {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
    'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15},
    'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Br', 'I', 'Hg', 'Bi']
}

qm9_with_h = {
    "atom_encoder": {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4},
    "atom_decoder": ["H", "C", "N", "O", "F"],
}

bonds = [
        'None'
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC'
    ]

NUM_ATOMS_QM9 = len(qm9_with_h.get("atom_encoder"))
NUM_ATOMS_DRUGS = len(drugs_with_h.get("atom_encoder"))
NUM_BONDS = len(bonds)

MASK_BOND_IDX = NUM_BONDS + 1
MASK_ATOM_IDX_QM9 = NUM_ATOMS_QM9 + 1
MASK_ATOM_IDX_DRUGS = NUM_ATOMS_DRUGS + 1