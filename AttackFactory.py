import torch
import utils

# Import all attacks
from attacks.linf_attack.APGD_Linf import DLR_AutoAttackPytorchMatGPUWrapper as APGD_Linf_Attack
from attacks.l1_attack.autoattack import AutoAttackL1 as AutoAttack
from attacks.l2_attack import autopgd_base
from attacks.l0_attack.L0_PGD import L0_PGD_AttackWrapper
from attacks.l0_attack.L0_Linf_PGD import L0_Linf_PGD_AttackWrapper
from attacks.l0_attack.L0_Sigma_PGD import L0_Sigma_PGD_AttackWrapper


# =========================================================
# Attack 1: APGD-Linf
# =========================================================
def APGD_Linf(device, model, correctLoader, eps):
    """Run APGD-Linf attack and return adversarial loader."""
    clipMin, clipMax = 0.0, 1.0
    numSteps = 500
    etaStart = 2.0 * eps

    print(f"\n--- Running APGD-Linf (DLR) | eps={eps:.6f} ---")
    
    advLoader = APGD_Linf_Attack(
        device, correctLoader, model, eps, etaStart, numSteps, clipMin, clipMax
    )
    
    return advLoader


# =========================================================
# Attack 2: APGD-L1
# =========================================================
def APGD_L1(device, model, correctLoader, eps):
    """Run APGD-L1 attack and return adversarial loader."""
    
    # Convert dataloader to tensors
    x_list, y_list = [], []
    for x, y in correctLoader:
        x_list.append(x)
        y_list.append(y)
    x_data = torch.cat(x_list, dim=0).float()
    y_data = torch.cat(y_list, dim=0).long()

    print(f"\n--- Running APGD-L1 (DLR) | eps={eps} ---")
    
    adv = AutoAttack(
        model,
        eps=eps,
        version="custom",
        attacks_to_run=["apgd-dlr"],
        device=device
    )

    adv.apgd.n_iter = 500
    adv.apgd.n_restarts = 1
    adv.apgd.use_rs = False
    adv.seed = 0

    x_adv = adv.run_standard_evaluation(x_data.to(device), y_data.to(device), bs=64)

    advLoader = utils.TensorToDataLoader(x_adv.detach().cpu(), y_data.cpu(), batchSize=64)
    
    return advLoader


# =========================================================
# Attack 3: APGD-L2
# =========================================================
def APGD_L2(device, model, correctLoader, eps_max):
    """Run APGD-L2 attack and return adversarial loader."""
    num_steps = 200
    loss_name = "dlr"

    print(f"\n--- Running APGD-L2 (DLR) | eps_max={eps_max} ---")

    attackObject = autopgd_base.APGDAttack(
        predict=model,
        n_iter=num_steps,
        norm="L2",
        n_restarts=0,
        eps=eps_max,
        seed=0,
        loss=loss_name,
        eot_iter=1,
        rho=0.75,
        topk=None,
        verbose=False,
        device=device,
        use_largereps=False,
        is_tf_model=False,
        logger=None
    )

    # Initialize hyperparameters
    dummy_x = torch.zeros((1, 1, 40, 50), device=device)
    attackObject.init_hyperparam(dummy_x)

    advLoader = attackObject.APGDCroceAttackWrapper(device, correctLoader)
    
    return advLoader

# Attack 4: L0-PGD
def L0_PGD(device, model, correctLoader, sparsity):
    """Run L0-PGD attack and return adversarial loader."""
    n_restarts = 10
    num_steps = 30
    step_size = 20
    random_start = False

    print(f"\n--- Running L0-PGD | sparsity={sparsity} ---")

    advLoader = L0_PGD_AttackWrapper(
        model=model,
        device=device,
        dataLoader=correctLoader,
        n_restarts=n_restarts,
        num_steps=num_steps,
        step_size=step_size,
        sparsity=sparsity,
        random_start=random_start
    )
    
    return advLoader


# --------- Attack 5: L0 + Linf PGD ------------
def L0_Linf_PGD(device, model, correctLoader, eps):
    """Run L0+Linf PGD attack and return adversarial loader."""
    k = 20
    n_restarts = 10
    step_size = 20
    num_steps = 30
    random_start = False

    print(f"\n--- Running L0+Linf PGD | eps={eps:.6f} ---")

    advLoader = L0_Linf_PGD_AttackWrapper(
        model=model,
        device=device,
        dataLoader=correctLoader,
        n_restarts=n_restarts,
        num_steps=num_steps,
        step_size=step_size,
        sparsity=k,
        epsilon=eps,
        random_start=random_start
    )
    
    return advLoader


# ------ Attack 6: L0 + Sigma-map PGD
def L0_Sigma_PGD(device, model, correctLoader, sparsity):
    """Run L0+Sigma PGD attack and return adversarial loader."""
    n_restarts = 10
    num_steps = 75
    step_size = 15
    kappa = 10
    random_start = False

    print(f"\n--- Running L0+Sigma PGD | sparsity={sparsity} ---")

    advLoader = L0_Sigma_PGD_AttackWrapper(
        model=model,
        device=device,
        dataLoader=correctLoader,
        n_restarts=n_restarts,
        num_steps=num_steps,
        step_size=step_size,
        sparsity=sparsity,
        kappa=kappa,
        random_start=random_start
    )
    
    return advLoader