This file is a merged representation of the entire codebase, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
__init__.py
autoattack.py
autopgd_base.py
checks.py
other_utils.py
state.py
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path="__init__.py">
try:
    # prefer a class already named AutoAttack
    from .autoattack import AutoAttack
except ImportError:
    # fallback: export AutoAttackL1 under the public name AutoAttack
    from .autoattack import AutoAttackL1 as AutoAttack
</file>

<file path="autoattack.py">
# autoattack_l1.py
import math
import time
import numpy as np
import torch

from .other_utils import Logger
from . import checks
from .state import EvaluationState
from .autopgd_base import APGDAttack  # only dependency we need for attacks


class AutoAttackL1:
    """
    Minimal AutoAttack-style orchestrator that ONLY supports:
      - APGD on Cross-Entropy loss  ('apgd-ce')
      - APGD on DLR loss            ('apgd-dlr')
    Norm is fixed to 'L1'. No FAB / Square / targeted variants.

    Versions:
      - 'standard': runs apgd-ce then apgd-dlr, APGD n_restarts=5 (L1 default)
      - 'plus':     same attacks, but you can bump iterations/restarts if you want
      - 'rand':     same attacks with eot_iter=20 (randomized defenses)
      - 'custom':   respects attacks_to_run you pass in (subset of the two above)
    """

    def __init__(
        self,
        model,
        eps,
        seed=None,
        verbose=True,
        attacks_to_run=None,
        version="standard",
        device="cuda",
        log_path=None,
        n_iter=100,
        n_restarts=None,      # if None we'll pick good defaults below
        eot_iter=1,
        rho=0.75
    ):
        self.model = model
        self.norm = "L1"
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.version = version
        self.device = device
        self.logger = Logger(log_path)

        # attacks we allow
        self._ALL = ["apgd-ce", "apgd-dlr"]
        self.attacks_to_run = list(attacks_to_run) if attacks_to_run else []

        # APGD (single instance; we switch the loss between 'ce' and 'dlr')
        self.apgd = APGDAttack(
            self.model,
            n_restarts=5 if n_restarts is None else n_restarts,
            n_iter=n_iter,
            verbose=False,
            eps=self.epsilon,
            norm=self.norm,
            eot_iter=eot_iter,
            rho=rho,
            seed=self.seed,
            device=self.device,
            logger=self.logger,
        )

        # Map version presets (and validate attacks_to_run)
        self._configure_version()

    # ------------------------ helpers ------------------------

    def get_logits(self, x):
        return self.model(x)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def _configure_version(self):
        if self.version in ["standard", "plus", "rand"] and self.attacks_to_run:
            raise ValueError(
                "attacks_to_run will be overridden unless you use version='custom'"
            )

        if self.version == "standard":
            # Classic ordering
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            # APGD defaults for L1
            self.apgd.n_restarts = 5
            self.apgd.eot_iter = 1

        elif self.version == "plus":
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            self.apgd.n_restarts = 5  # bump to your taste
            self.apgd.eot_iter = 1

        elif self.version == "rand":
            self.attacks_to_run = ["apgd-ce", "apgd-dlr"]
            self.apgd.n_restarts = 1
            self.apgd.eot_iter = 20  # stochastic defenses

        elif self.version == "custom":
            # Keep only supported items, preserve user order
            if not self.attacks_to_run:
                raise ValueError("With version='custom', provide attacks_to_run.")
            unsupported = [a for a in self.attacks_to_run if a not in self._ALL]
            if unsupported:
                raise ValueError(f"Unsupported attacks in attacks_to_run: {unsupported}")
        else:
            raise ValueError(f"Unknown version: {self.version}")

        if self.verbose:
            self.logger.log(
                f"Configured version '{self.version}' with attacks: {', '.join(self.attacks_to_run)}"
            )

    # ------------------------ public API ------------------------

    @torch.no_grad()
    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = math.ceil(x_orig.shape[0] / bs)
        acc = 0.0
        for counter in range(n_batches):
            x = x_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs : min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output.max(1)[1] == y).float().sum()

        if self.verbose:
            self.logger.log(f"clean accuracy: {acc / x_orig.shape[0]:.2%}")
        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation(
        self,
        x_orig,
        y_orig,
        bs=250,
        return_labels=False,
        state_path=None,
    ):
        """
        Runs APGD-CE then APGD-DLR (or whatever is in attacks_to_run), only on
        currently robust points, like AutoAttack's choreography. Supports resuming via state.
        """
        # ----- state (optional resume) -----
        if state_path is not None and state_path.exists():
            state = EvaluationState.from_disk(state_path)
            if set(self.attacks_to_run) != state.attacks_to_run:
                raise ValueError(
                    "The state was created with a different set of attacks to run."
                )
            if self.verbose:
                self.logger.log(f"Restored state from {state_path}")
                self.logger.log(
                    "Since the state has been restored, only adversarials from the current run are returned."
                )
        else:
            state = EvaluationState(set(self.attacks_to_run), path=state_path)
            state.to_disk()
            if self.verbose and state_path is not None:
                self.logger.log(f"Created state in {state_path}")

        attacks_to_run = [a for a in self.attacks_to_run if a not in state.run_attacks]
        if self.verbose:
            self.logger.log(
                f"using {self.version} version including {', '.join(attacks_to_run)}."
            )
            if state.run_attacks:
                self.logger.log(f"{', '.join(state.run_attacks)} was/were already run.")

        # ----- pre-checks -----
        if self.version != "rand":
            checks.check_randomized(
                self.get_logits, x_orig[:bs].to(self.device), y_orig[:bs].to(self.device), bs=bs, logger=self.logger
            )
        n_cls = checks.check_range_output(self.get_logits, x_orig[:bs].to(self.device), logger=self.logger)
        checks.check_dynamic(self.model, x_orig[:bs].to(self.device), is_tf_model=False, logger=self.logger)
        # keep a minimal class-count sanity (no targeted logic here)
        if n_cls < 2:
            raise ValueError("Model appears to have < 2 classes from output range check.")

        # ----- initial clean evaluation -----
        with torch.no_grad():
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            if state.robust_flags is None:
                robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
                y_adv = torch.empty_like(y_orig)
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])
                    x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x).max(dim=1)[1]
                    y_adv[start_idx:end_idx] = output
                    robust_flags[start_idx:end_idx] = y.eq(output)

                state.robust_flags = robust_flags
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": robust_accuracy}
                state.clean_accuracy = robust_accuracy
                if self.verbose:
                    self.logger.log(f"initial accuracy: {robust_accuracy:.2%}")
            else:
                robust_flags = state.robust_flags.to(x_orig.device)
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict = {"clean": state.clean_accuracy}
                if self.verbose:
                    self.logger.log(f"initial clean accuracy: {state.clean_accuracy:.2%}")
                    self.logger.log(
                        f"robust accuracy at the time of restoring the state: {robust_accuracy:.2%}"
                    )

            x_adv = x_orig.clone().detach()
            startt = time.time()

            # ----- main loop over attacks -----
            for attack in attacks_to_run:
                num_robust = int(torch.sum(robust_flags).item())
                if num_robust == 0:
                    break

                n_batches = int(np.ceil(num_robust / bs))
                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)
                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)

                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)
                    if len(x.shape) == 3:
                        x = x.unsqueeze(0)

                    # run APGD with desired loss
                    if attack == "apgd-ce":
                        self.apgd.loss = "ce"
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    elif attack == "apgd-dlr":
                        self.apgd.loss = "dlr"
                        self.apgd.seed = self.get_seed()
                        adv_curr = self.apgd.perturb(x, y)
                    else:
                        raise ValueError(f"Attack not supported: {attack}")

                    output = self.get_logits(adv_curr).max(dim=1)[1]
                    false_batch = ~y.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    state.robust_flags = robust_flags

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                    # store adversarial labels (optional; mirrors original behavior)
                    y_adv[non_robust_lin_idcs] = output[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = int(torch.sum(false_batch).item())
                        self.logger.log(
                            f"{attack} - {batch_idx + 1}/{n_batches} - {num_non_robust_batch} out of {x.shape[0]} successfully perturbed"
                        )

                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                robust_accuracy_dict[attack] = robust_accuracy
                state.add_run_attack(attack)
                if self.verbose:
                    self.logger.log(
                        f"robust accuracy after {attack.upper()}: {robust_accuracy:.2%} (total time {time.time() - startt:.1f} s)"
                    )

            # ----- final checks and state save -----
            state.to_disk(force=True)
            if self.verbose:
                # L1 radius report
                res = (x_adv - x_orig).abs().reshape(x_orig.shape[0], -1).sum(dim=-1)
                self.logger.log(
                    f"max {self.norm} perturbation: {res.max():.5f}, "
                    f"nan in tensor: {(x_adv != x_adv).sum()}, max: {x_adv.max():.5f}, min: {x_adv.min():.5f}"
                )
                final_ra = torch.sum(robust_flags).item() / x_orig.shape[0]
                self.logger.log(f"robust accuracy: {final_ra:.2%}")

        return (x_adv, y_adv) if return_labels else x_adv

    # Convenience API mirroring upstream behavior
    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250, return_labels=False):
        if self.verbose:
            self.logger.log(f"using {self.version} version including {', '.join(self.attacks_to_run)}")

        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False
        for attack in self.attacks_to_run:
            startt = time.time()
            x_adv, y_adv = self.run_standard_evaluation(x_orig, y_orig, bs=bs, return_labels=True)
            adv[attack] = (x_adv, y_adv) if return_labels else x_adv
            if verbose_indiv:
                acc_indiv = self.clean_accuracy(x_adv, y_orig, bs=bs)
                self.logger.log(
                    f"robust accuracy by {attack.upper()}\t {acc_indiv:.2%}\t (time attack: {time.time() - startt:.1f} s)"
                )
        return adv
</file>

<file path="autopgd_base.py">
## L1 separated version from original AutoAttack

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .other_utils import L0_norm, L1_norm
from .checks import check_zero_gradients


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''
    x = x2.clone().float().view(x2.shape[0], -1) #original image x2 to 1D
    y = y2.clone().float().view(y2.shape[0], -1) #current perturbation y2 to 1D , perturbation vector d =u-x 
    sigma = y.clone().sign() # sign (=/-)of y vector s
    u = torch.min(1 - x - y, x + y) # compute y_i (gamma),   u=x+y? 
    u = torch.min(torch.zeros_like(y), u) #-u is exactly min(|u_i-x_i|, y_i)
    l = -torch.clone(y).abs() # -|y| corresponds to the point where the shrink lamda equals the perturbation magnitude |di|. #negative absolute current perturbation
    d = u.clone()
    #line 7
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1) # concatenates all breakpoints (u and l) into a single long vector and sorts them (bs). indbs are their indices
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1) #bs2 is bs shifted left with a zero appended
 
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1) #pre-calculates the slope of the piecewise linear function between each of the sorted breakpoints.

    #checking of easy case
    s1 = -u.sum(dim=1) # line 5

    c = eps1 - y.clone().abs().sum(dim=1) #remaining budget when starting from current perturbation y, c represents how much extra L1 we can still add to |y|_1 before hitting eps
    c5 = s1 + c < 0   #checks if this is over the budget (c5), S > eps (line 6), if exceeds budget need lamda_star
    c2 = c5.nonzero().squeeze(1) #check for hard case? 

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1) #This vector pre-computes the value of the function f(lamda) at every breakpoint. S array is what will compare to eps to find where S crosses eps.

    if c2.nelement != 0:   # The if S > eps check is equivalent to it

      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)

      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
       
      #implemented by concatenating breakpoints u and l, sorting them, building cumulative arrays, 
      #then binary-searching (lb/ub) to find the interval where the sum crosses eps, followed by computing alpha = lamda_star.
      while counter < nitermax:  # binary search to find the lamda_star  line 10-29 
        counter4 = torch.floor((lb + ub) / 2.)  # dividing the middle
        counter2 = counter4.type(torch.LongTensor)

        c8 = s[c2, counter2] + c[c2] < 0   #The c8 = s ... < 0 is the check that decides which half of the search space to discard.
        ind3 = c8.nonzero().squeeze(1)     #logic to update lb, ub 
        ind32 = (~c8).nonzero().squeeze(1)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]

        counter += 1

      lb2 = lb.long()  
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]  # Once the binary search finds the correct interval, it finds the exact value of lambda star using linear interpolation.
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2]) # This line uses the alpha to compute the final, corrected magnitudes for the perturbation for each pixel. 

    return (sigma * d).view(x2.shape)     #uses alpha and clamps to yield final d, which equals zi in magnitude.


class APGDAttack():
    """
    AutoPGD (L1-only, CE & DLR)
    https://arxiv.org/abs/2003.01690

    Kept function & method names/signatures identical to the upstream file,
    but restricted to L1 norm and untargeted CE / DLR losses.
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='L1',
            n_restarts=1,
            eps=None,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch (L1 + CE/DLR only).
        Names/args kept to preserve external API compatibility.
        """
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        self.use_largereps = use_largereps
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0. if eps is not None else None
        self.is_tf_model = is_tf_model  # kept
        self.y_target = None
        self.logger = logger

        # L1 only
        assert self.norm in ['L1'], 'This minimal build supports only L1.'
        assert self.eps is not None
        if self.is_tf_model:
            raise ValueError('TF models are not supported in this minimal L1 build.')

        # parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    def init_hyperparam(self, x):
        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        # L1 normalize to avoid changing names
        try:
            t = x.abs().view(x.shape[0], -1).sum(dim=-1)
        except:
            t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

   # def dlr_loss(self, x, y):
        #x_sorted, ind_sorted = x.sort(dim=1)
        #ind = (ind_sorted[:, -1] == y).float()
        #u = torch.arange(x.shape[0], device=x.device)
    def dlr_loss(self, x, y):
        """
        DLR loss with binary-safe 
        - C >= 3 (original AutoAttack):  -(z_y - z_k) / (z_(1) - z_(3) + eps)
        - C = 2 (binary):                -(z_y - z_k) 
        Returns: [B] tensor
        """
        x_sorted, ind_sorted = x.sort(dim=1)     # ascending: [:, -1]=max=z_(1), [:, -2]=z_(2), [:, -3]=z_(3)
        B, C = x.shape
        u = torch.arange(B, device=x.device)

        if C >= 3:
            # --- ORIGINAL (unchanged) ---
            ind = (ind_sorted[:, -1] == y).float()  # is y the argmax?
            return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

        elif C == 2:
            # --- BINARY-SAFE --- aligned with paper
            # y must be 0/1; pick the other class logit directly
            other = y ^ 1  # same as (1 - y), but explicit for {0,1}
            zy = x[u, y]
            zk = x[u, other]
            return -(zy - zk)  
            #zmax = x_sorted[:, -1]                      # max logit
            #zmin = x_sorted[:,  0]                      # min logit
            #is_top = (ind_sorted[:, -1] == y)           # y is argmax?
            #zk = torch.where(is_top, zmin, zmax)        # second max if y is max; else max
            #zy = x[u, y]
            #return -(zy - zk) / (zmax - zmin + 1e-12)

        else:
            # C == 1 (degenerate): no meaningful margin
            return torch.zeros(B, device=x.device, dtype=x.dtype)




    def attack_single_run(self, x, y, x_init=None):
        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            
        if x_init is not None:
            x_adv = x_init.clone()
            if self.verbose:
                print('[custom init] L1 perturbation {:.5f}'.format(
                    (x_adv - x).abs().view(x.shape[0], -1).sum(1).max()))
        
        elif getattr(self, "use_rs", True):
        # random L1 init (projection onto L1-ball  [0,1])
            t = torch.randn(x.shape, device=self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = (x + t + delta)
        else:
        # deterministic start: exactly the clean image
            x_adv = x.clone()        

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]], device=self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]], device=self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        # CE / DLR only (PyTorch)
        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknown loss (allowed: ce, dlr)')

        #  gradient at init  line 3–4: compute VL(x(i)) 
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        if self.loss in ['dlr']:
            check_zero_gradients(grad, logger=self.logger)

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        #alpha = 1.0  # L1 uses alpha=1 in upstream
        alpha = 0.1 # for voting dataset some cases
        step_size = alpha * self.eps * torch.ones([x.shape[0], *([1] * self.ndims)], device=self.device).detach()
        x_adv_old = x_adv.clone()
        k = max(int(.04 * self.n_iter), 1)
        n_fts = math.prod(self.orig_dim)

        # init sparsity schedule
        if x_init is None:
            topk = .2 * torch.ones([x.shape[0]], device=self.device)
            sp_old = n_fts * torch.ones_like(topk)
        else:
            topk = L0_norm(x_adv - x) / n_fts / 1.5
            sp_old = L0_norm(x_adv - x)

        adasp_redstep = 1.5
        adasp_minstep = 10.
        counter3 = 0
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter): #Loop here line 4
            # gradient step (L1 sparse update + projection)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0] #line 6 (compute sparse sign of gradient with k.d active coordinates) 
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1] * (len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10) #line 14 step to u, sparsegrad.sign() --S(VL,k.d) 

                delta_u = x_adv_1 - x  #line 15 -- proposed perturbation relative to clean x
                delta_p = L1_projection(x, delta_u, self.eps)
                #x_adv = (x + delta_u + delta_p)
                x_adv = (x_adv_1 + delta_p)
                #x_adv   = x + delta_p.clamp(0., 1.)
            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
            if self.verbose:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(step_size.mean(), topk.mean() * n_fts)
                print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
                    i, loss_best.sum(), acc.float().mean(), str_stats))

            # step-size & sparsity adaptation (L1)
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1 + 0
                ind = (y1 > loss_best).nonzero().squeeze() #track best loss/point 
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1
                if counter3 == k:  
                    sp_curr = L0_norm(x_best - x)  #  line 7 -- update k 
                    fl_redtopk = (sp_curr / sp_old) < .95 # did the sparsity change enough? 
                    topk = sp_curr / n_fts / 1.5   #equation 9 
                    step_size[fl_redtopk] = alpha * self.eps   # line 8 -- update n eta 
                    step_size[~fl_redtopk] /= adasp_redstep    # reduce as per equation 10 
                    step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)  #keep within eps and eps/10 
                    sp_old = sp_curr.clone()

                    x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                    grad[fl_redtopk] = grad_best[fl_redtopk].clone()

                    counter3 = 0

        return (x_best, acc, loss_best, x_best_adv)

    def perturb(self, x, y=None, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True return points with highest loss (not used by wrapper)
        :param x_init:      optional custom initialization
        """
        assert self.loss in ['ce', 'dlr']
        if y is not None and len(y.shape) == 0:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        y_pred = self.model(x).max(1)[1]
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)  # not taken, y passed
        else:
            y = y.detach().clone().long().to(self.device)  #it is executed

        adv = x.clone()  # fill with adversarials
        acc = (y_pred == y)  # starts with correctly classified in this batch
        loss = -1e10 * torch.ones_like(acc).float()

        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))

        if self.use_largereps: # not executed, default is false
            epss = [3. * self.eps_orig, 2. * self.eps_orig, 1. * self.eps_orig]
            iters = [.3 * self.n_iter_orig, .3 * self.n_iter_orig, .4 * self.n_iter_orig]
            iters = [math.ceil(c) for c in iters]
            iters[-1] = self.n_iter_orig - sum(iters[:-1])

        if not best_loss: # True, as best loss pass yet (first run)
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts): # n_restart= 1 or 5 any value
                ind_to_fool = acc.nonzero().squeeze()
                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()

                    if not self.use_largereps: # True (as use_largereps is default False)
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool, x_init=x_init) #x_init = NONE here, it calls attack_single_run() the main algorithm 1
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)  # not executed
                    best_curr, acc_curr, loss_curr, adv_curr = res_curr #do
                    ind_curr = (acc_curr == 0).nonzero().squeeze()  #do

                    acc[ind_to_fool[ind_curr]] = 0  # write back fooled examples to the correct original indices 
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()))

            return adv # return this adv
        else: # not executed now, but executed when it is best loss
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]], device=self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y, x_init=x_init)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum()))
            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, float(epss[0]))
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            if x_init is not None:
                x_init += L1_projection(x, x_init - x, float(eps))
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)
        return (x_init, acc, loss, x_adv)


# Note: Linf/L2 branches have been removed intentionally
# to keep this file focused on L1 + CE/DLR while preserving the API functional.
</file>

<file path="checks.py">
import torch
import warnings
import math
import sys

from .other_utils import L2_norm

funcs = {'grad': 0,
    'backward': 0,
    '_make_grads': 0,
    }

checks_doc_path = 'flags_doc.md'


def check_randomized(model, x, y, bs=250, n=5, alpha=1e-4, logger=None):
    acc = []
    corrcl = []
    outputs = []
    with torch.no_grad():
        for _ in range(n):
            output = model(x)
            corrcl_curr = (output.max(1)[1] == y).sum().item()
            corrcl.append(corrcl_curr)
            outputs.append(output / (L2_norm(output, keepdim=True) + 1e-10))
    acc = [c != corrcl_curr for c in corrcl]
    max_diff = 0.
    for c in range(n - 1):
        for e in range(c + 1, n):
            diff = L2_norm(outputs[c] - outputs[e])
            max_diff = max(max_diff, diff.max().item())
    if any(acc) or max_diff > alpha:
        msg = 'it seems to be a randomized defense! Please use version="rand".' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_range_output(model, x, alpha=1e-5, logger=None):
    with torch.no_grad():
        output = model(x)
    fl = [output.max() < 1. + alpha, output.min() >  -alpha,
        ((output.sum(-1) - 1.).abs() < alpha).all()]
    if all(fl):
        msg = 'it seems that the output is a probability distribution,' +            ' please be sure that the logits are used!' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
    return output.shape[-1]


def check_zero_gradients(grad, logger=None):
    z = grad.view(grad.shape[0], -1).abs().sum(-1)
    if (z == 0).any():
        msg = f'there are {(z == 0).sum()} points with zero gradient!' +             ' This might lead to unreliable evaluation with gradient-based attacks.' +             f' See {checks_doc_path} for details.'
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def tracefunc(frame, event, args):
    if event == 'call' and frame.f_code.co_name in funcs.keys():
        funcs[frame.f_code.co_name] += 1


def check_dynamic(model, x, is_tf_model=False, logger=None):
    if is_tf_model:
        msg = 'the check for dynamic defenses is not currently supported'
    else:
        msg = None
        sys.settrace(tracefunc)
        model(x)
        sys.settrace(None)
        if any([c > 0 for c in funcs.values()]):
            msg = 'it seems to be a dynamic defense! The evaluation' +                 ' with AutoAttack might be insufficient.' +                 f' See {checks_doc_path} for details.'
    if msg is not None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')


def check_n_classes(n_cls, attacks_to_run, apgd_targets=None, fab_targets=None, logger=None):
    """
    Minimal version:
      - Only sanity-checks DLR feasibility (n_cls > 2) when 'apgd-dlr' is requested.
      - Keeps the original signature for compatibility. Ignores targeted/FAB args.
    """
    msg = None
    if 'apgd-dlr' in attacks_to_run:
        if n_cls <= 2:
            msg = f'with only {n_cls} classes it is not possible to use the DLR loss!'
    if msg is not None:
        if logger is None:
            warnings.warn(Warning(msg))
        else:
            logger.log(f'Warning: {msg}')
</file>

<file path="other_utils.py">
import os
import collections.abc as container_abcs

import torch

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path
        
    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
                f.flush()
            
def check_imgs(adv, x, norm):
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == 'Linf':
        res = delta.abs().max(dim=1)[0]
    elif norm == 'L2':
        res = (delta ** 2).sum(dim=1).sqrt()
    elif norm == 'L1':
        res = delta.abs().sum(dim=1)

    str_det = 'max {} pert: {:.5f}, nan in imgs: {}, max in imgs: {:.5f}, min in imgs: {:.5f}'.format(
        norm, res.max(), (adv != adv).sum(), adv.max(), adv.min())
    print(str_det)
    
    return str_det

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
</file>

<file path="state.py">
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Set
import warnings

import torch


@dataclass
class EvaluationState:
    _attacks_to_run: Set[str]
    path: Optional[Path] = None
    _run_attacks: Set[str] = field(default_factory=set)
    _robust_flags: Optional[torch.Tensor] = None
    _last_saved: datetime = datetime(1, 1, 1)
    _SAVE_TIMEOUT: int = 60
    _clean_accuracy: float = float("nan")

    def to_disk(self, force: bool = False) -> None:
        seconds_since_last_save = (datetime.now() -
                                   self._last_saved).total_seconds()
        if self.path is None or (seconds_since_last_save < self._SAVE_TIMEOUT
                                 and not force):
            return
        self._last_saved = datetime.now()
        d = asdict(self)
        if self.robust_flags is not None:
            d["_robust_flags"] = d["_robust_flags"].cpu().tolist()
        d["_run_attacks"] = list(self._run_attacks)
        with self.path.open("w", ) as f:
            json.dump(d, f, default=str)

    @classmethod
    def from_disk(cls, path: Path) -> "EvaluationState":
        with path.open("r") as f:
            d = json.load(f)
        d["_robust_flags"] = torch.tensor(d["_robust_flags"], dtype=torch.bool)
        d["path"] = Path(d["path"])
        if path != d["path"]:
            warnings.warn(
                UserWarning(
                    "The given path is different from the one found in the state file."
                ))
        d["_last_saved"] = datetime.fromisoformat(d["_last_saved"])
        return cls(**d)

    @property
    def robust_flags(self) -> Optional[torch.Tensor]:
        return self._robust_flags

    @robust_flags.setter
    def robust_flags(self, robust_flags: torch.Tensor) -> None:
        self._robust_flags = robust_flags
        self.to_disk(force=True)

    @property
    def run_attacks(self) -> Set[str]:
        return self._run_attacks

    def add_run_attack(self, attack: str) -> None:
        self._run_attacks.add(attack)
        self.to_disk()
        
    @property
    def attacks_to_run(self) -> Set[str]:
        return self._attacks_to_run
    
    @attacks_to_run.setter
    def attacks_to_run(self, _: Set[str]) -> None:
        raise ValueError("attacks_to_run cannot be set outside of the constructor")

    @property
    def clean_accuracy(self) -> float:
        return self._clean_accuracy

    @clean_accuracy.setter
    def clean_accuracy(self, accuracy) -> None:
        self._clean_accuracy = accuracy
        self.to_disk(force=True)

    @property
    def robust_accuracy(self) -> float:
        if self.robust_flags is None:
            raise ValueError("robust_flags is not set yet. Start the attack first.")
        if self.attacks_to_run - self.run_attacks:
            warnings.warn("You are checking `robust_accuracy` before all the attacks"
                          " have been run.")
        return self.robust_flags.float().mean().item()
</file>

</files>
