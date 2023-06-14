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


class CategoricalTerminalKernel(torch.nn.Module):
  def __init__(self,
               terminal_state,
               num_classes: int = 5,
               alphas: torch.Tensor = DEFAULT_ALPHAS, 
               timesteps: int = 300,
               ):
    super().__init__()
    
    self.num_classes = num_classes
    self.terminal_state = terminal_state
    Qs = [get_terminal_transition(a, K=num_classes, e=terminal_state) for a in alphas]
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
    self.register_buffer('alphas', alphas)
    self.register_buffer('Qt', Qs)
    self.register_buffer('Qt_bar', Qs_bar)
    self.register_buffer('Qt_bar_prev', Qs_bar_prev)
    
  def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
    """_summary_
    Computes the forward categorical posterior q(xt | x0) ~ Cat(xt, p = x0_j . Qt_bar_ji)
    Args:
        x0 (torch.Tensor): _description_ one-hot vectors of shape (n, k)
        t (torch.Tensor): _description_ time variable of shape (n,)

    Returns:
        _type_: _description_
    """
    probs = torch.einsum('nj, nji -> ni', [x0, self.Qt_bar[t]])
    return probs
  
  def reverse_posterior_for_every_x0(self, xt: torch.Tensor, t: torch.Tensor):
    """_summary_
    Computes the reverse posterior q(x_{t-1} | xt, x0) as described in Austin et al. (2021) https://arxiv.org/abs/2107.03006 in Eq.3
    but for every possible value of x0
    Args:
        xt (torch.Tensor): _description_ a perturbed (noisy) one-hot vector of shape (n, k)
        t (torch.Tensor): _description_ time variable of shape (n,)
    Returns:
        _type_: _description_
    """
      
    # xt: (n, k)
    #x0 = torch.eye(self.num_classes, device=xt.device, dtype=xt.dtype).unsqueeze(0)
    #x0 = x0.repeat((xt.size(0), 1, 1))
    # (n, k, k)
    a = torch.einsum('nj, nji -> ni', [xt, self.Qt[t].transpose(-2, -1)])
    # (n, k)
    a = a.unsqueeze(1)
    # (n, 1, k)
    #b = torch.einsum('naj, nji -> nai', [x0, self.Qt_bar_prev[t]])
    b = self.Qt_bar_prev[t]
    # (n, k, k)
    p0 = a * b
    # (n, k, k)
    # p1 = torch.einsum('naj, nji -> nai', [x0, self.Qt_bar[t]])
    p1 = self.Qt_bar[t]
    # (n, k, k)
    
    xt = xt.unsqueeze(1)
    # (n, 1, k)
  
    p1 = (p1 * xt).sum(-1, keepdims=True)
    # (n, k, 1)
    m = (p1==0.0).float() * 1e-4
    probs = p0 / (p1 + m)
    # (n, k, k)    
    return probs
  
  def reverse_posterior(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
    """_summary_
    Computes the reverse posterior q(x_{t-1} | xt, x0) as described in Austin et al. (2021) https://arxiv.org/abs/2107.03006 in Eq.3
    Args:
        x0 (torch.Tensor): _description_ one specific one-hot vector of shape (n, k)
        xt (torch.Tensor): _description_ a perturbed (noisy) one-hot vector of shape (n, k)
        t (torch.Tensor): _description_ time variable of shape (n,)
    Returns:
        _type_: _description_
    """
    a = torch.einsum('nj, nji -> ni', [xt, self.Qt[t].transpose(-2, -1)])
    b = torch.einsum('nj, nji -> ni', [x0, self.Qt_bar_prev[t]])
    p0 = a * b
    # (n, k)
    p1 = torch.einsum('nj, nji -> ni', [x0, self.Qt_bar[t]])
    p1 = (p1 * xt).sum(-1, keepdims=True)
    # (n, 1)
    
    probs = p0 / p1 
    return probs



class CategoricalUniformKernel(torch.nn.Module):
  def __init__(self,
               num_classes: int = 5,
               alphas: torch.Tensor = DEFAULT_ALPHAS, 
               timesteps: int = 300, **kwargs
               ):
    super().__init__()
    
    self.num_classes = num_classes
    Qs = [get_uniform_transition(a, K=num_classes) for a in alphas]
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
    self.register_buffer('alphas', alphas)
    self.register_buffer('Qt', Qs)
    self.register_buffer('Qt_bar', Qs_bar)
    self.register_buffer('Qt_bar_prev', Qs_bar_prev)
    
  def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
    """_summary_
    Computes the forward categorical posterior q(xt | x0) ~ Cat(xt, p = x0_j . Qt_bar_ji)
    Args:
        x0 (torch.Tensor): _description_ one-hot vectors of shape (n, k)
        t (torch.Tensor): _description_ time variable of shape (n,)

    Returns:
        _type_: _description_
    """
    probs = torch.einsum('nj, nji -> ni', [x0, self.Qt_bar[t]])
    return probs
  
  def reverse_posterior_for_every_x0(self, xt: torch.Tensor, t: torch.Tensor):
    """_summary_
    Computes the reverse posterior q(x_{t-1} | xt, x0) as described in Austin et al. (2021) https://arxiv.org/abs/2107.03006 in Eq.3
    but for every possible value of x0
    Args:
        xt (torch.Tensor): _description_ a perturbed (noisy) one-hot vector of shape (n, k)
        t (torch.Tensor): _description_ time variable of shape (n,)
    Returns:
        _type_: _description_
    """
      
    # xt: (n, k)
    #x0 = torch.eye(self.num_classes, device=xt.device, dtype=xt.dtype).unsqueeze(0)
    #x0 = x0.repeat((xt.size(0), 1, 1))
    # (n, k, k)
    a = torch.einsum('nj, nji -> ni', [xt, self.Qt[t].transpose(-2, -1)])
    # (n, k)
    a = a.unsqueeze(1)
    # (n, 1, k)
    #b = torch.einsum('naj, nji -> nai', [x0, self.Qt_bar_prev[t]])
    b = self.Qt_bar_prev[t]
    # (n, k, k)
    p0 = a * b
    # (n, k, k)
    # p1 = torch.einsum('naj, nji -> nai', [x0, self.Qt_bar[t]])
    p1 = self.Qt_bar[t]
    # (n, k, k)
    
    xt = xt.unsqueeze(1)
    # (n, 1, k)
  
    p1 = (p1 * xt).sum(-1, keepdims=True)
    # (n, k, 1)
    m = (p1==0.0).float() * 1e-4
    probs = p0 / (p1 + m)
    # (n, k, k)    
    return probs
  
  def reverse_posterior(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
    """_summary_
    Computes the reverse posterior q(x_{t-1} | xt, x0) as described in Austin et al. (2021) https://arxiv.org/abs/2107.03006 in Eq.3
    Args:
        x0 (torch.Tensor): _description_ one specific one-hot vector of shape (n, k)
        xt (torch.Tensor): _description_ a perturbed (noisy) one-hot vector of shape (n, k)
        t (torch.Tensor): _description_ time variable of shape (n,)
    Returns:
        _type_: _description_
    """
    a = torch.einsum('nj, nji -> ni', [xt, self.Qt[t].transpose(-2, -1)])
    b = torch.einsum('nj, nji -> ni', [x0, self.Qt_bar_prev[t]])
    p0 = a * b
    # (n, k)
    p1 = torch.einsum('nj, nji -> ni', [x0, self.Qt_bar[t]])
    p1 = (p1 * xt).sum(-1, keepdims=True)
    # (n, 1)
    
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

