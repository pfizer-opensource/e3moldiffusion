import torch
from e3moldiffusion.sde import get_beta_schedule

DEFAULT_BETAS = get_beta_schedule(kind="cosine", num_diffusion_timesteps=300)
DEFAULT_ALPHAS = 1.0 - DEFAULT_BETAS
ALPHAS_BAR = torch.cumprod(DEFAULT_ALPHAS, dim=0)


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


class CategoricalKernel(torch.nn.Module):
  def __init__(self,
               num_classes: int = 6,
               alphas_bar: torch.Tensor = DEFAULT_ALPHAS, 
               timesteps: int = 300):
    super().__init__()
    Qs = [get_terminal_transition(a, K=num_classes, e=num_classes-1) for a in alphas_bar]
    Qs_prev = torch.eye(num_classes)
    Qs_bar = []
    for i in range(timesteps):
      Qsb = Qs_prev @ Qs[i]
      Qs_bar.append(Qsb)
      Qs_prev = Qsb
    
    Qs = torch.stack(Qs)
    Qs_bar = torch.stack(Qs_bar)
    Qs_bar_prev = Qs_bar[:-1]
    Qs_prev_pad = torch.eye(num_classes)
    Qs_bar_prev = torch.concat([Qs_prev_pad.unsqueeze(0), Qs_bar_prev], dim=0)
    self.register_buffer('alphas_bar', alphas_bar)
    self.register_buffer('Qt', Qs)
    self.register_buffer('Qt_bar', Qs_bar)
    self.register_buffer('Qt_bar_prev', Qs_bar_prev)
    
  def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
    probs = torch.einsum('nj, nji -> ni', [x0, self.Qt_bar[t]])
    return probs
    
  def reverse_posterior(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
    a = torch.einsum('nj, nji -> ni', [xt, self.Qt[t].transpose(-2, -1)])
    b = torch.einsum('nj, nji -> ni', [xt, self.Qt_bar_prev[t]])
    p0 = a * b
    p1 = torch.einsum('nj, nji -> ni', [x0, self.Qt_bar[t]])
    p1 = (p1 * xt).sum(-1, keepdims=True)
    probs = p0 / p1 
    return probs
  

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

