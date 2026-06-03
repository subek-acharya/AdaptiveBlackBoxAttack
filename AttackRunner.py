# AttackRunner.py
# Runs all attacks with multiple parameters, evaluates, saves samples, and writes results

import torch
import os
import utils
import AttackFactory

# Import configurations from config.py
from config import (
    APGD_LINF_PARAMS,
    APGD_L1_PARAMS,
    APGD_L2_PARAMS,
    L0_PGD_PARAMS,
    L0_LINF_PARAMS,
    L0_SIGMA_PARAMS,
    SAVE_CONFIG,
)


# =========================================================
# UTILITY FUNCTIONS
# =========================================================

def evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes):
    """Evaluate adversarial samples on both models."""
    # Synthetic model
    syn_acc = utils.validateD(adv_loader, synthetic_model, device)
    print(f"[SYNTHETIC MODEL] Adversarial Accuracy: {syn_acc:.4f}")
    utils.calculateClasswiseAccuracy(adv_loader, synthetic_model, device, num_classes)
    
    # Oracle model
    oracle_acc = utils.validateD(adv_loader, oracle, device)
    print(f"[ORACLE MODEL] Adversarial Accuracy: {oracle_acc:.4f}")
    utils.calculateClasswiseAccuracy(adv_loader, oracle, device, num_classes)
    
    return syn_acc, oracle_acc


def save_adv_samples(adv_loader, attack_name, param_name, param_value, n_save=None):
    """Save adversarial samples to organized folder structure."""
    if n_save is None:
        n_save = SAVE_CONFIG["n_save"]
    
    base_dir = os.path.join(SAVE_CONFIG["base_dir"], attack_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # Create filename
    if isinstance(param_value, float) and param_value <= 1:
        filename = f"{param_name}_{int(param_value * 255)}by255.pt"
    else:
        filename = f"{param_name}_{param_value}.pt"
    
    save_path = os.path.join(base_dir, filename)
    
    imgs, lbls = [], []
    count = 0
    for x, y in adv_loader:
        imgs.append(x.cpu())
        lbls.append(y.cpu())
        count += x.size(0)
        if count >= n_save:
            break
    
    final_imgs = torch.cat(imgs)[:n_save]
    final_lbls = torch.cat(lbls)[:n_save]
    torch.save({"images": final_imgs, "labels": final_lbls}, save_path)
    print(f"Saved {final_imgs.shape[0]} samples to {save_path}")
    
    return save_path


def write_attack_results_to_file(results_file, attack_name, param_name, results_summary):
    """Write results for one attack type to file."""
    results_file.write(f"\n\n{'='*90}\n")
    results_file.write(f"{attack_name} RESULTS\n")
    results_file.write(f"{'='*90}\n")
    results_file.write(f"{param_name:<15} {'Value':<15} {'Syn Acc':<12} {'Oracle Acc':<12}\n")
    results_file.write(f"{'-'*90}\n")
    
    for name, results in results_summary.items():
        val = results['param_value']
        val_str = f"{val:.6f}" if isinstance(val, float) and val < 10 else f"{val}"
        
        results_file.write(
            f"{name:<15} {val_str:<15} {results['syn_acc']:<12.4f} {results['oracle_acc']:<12.4f}\n"
        )
    
    results_file.write(f"{'-'*90}\n")
    results_file.flush()


# =========================================================
# MAIN ATTACK RUNNER
# =========================================================

def run_all_attacks(device, oracle, synthetic_model, correct_loader, num_classes, results_file):
    """Run all attacks with all parameter values."""
    
    all_results = {}
    
    # =====================================================
    # ATTACK 1: APGD-Linf
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: APGD-Linf")
    print("#"*70)
    
    apgd_linf_results = {}
    for param_name, eps in APGD_LINF_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"APGD-Linf | eps = {eps}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.APGD_Linf(device, synthetic_model, correct_loader, eps)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "APGD_Linf", "eps", eps)
        
        # Store results
        apgd_linf_results[param_name] = {
            "param_value": eps,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "APGD-Linf", "Epsilon", apgd_linf_results)
    all_results["apgd_linf"] = apgd_linf_results
    
    # =====================================================
    # ATTACK 2: APGD-L1
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: APGD-L1")
    print("#"*70)
    
    apgd_l1_results = {}
    for param_name, eps in APGD_L1_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"APGD-L1 | eps = {eps}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.APGD_L1(device, synthetic_model, correct_loader, eps)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "APGD_L1", "eps", eps)
        
        # Store results
        apgd_l1_results[param_name] = {
            "param_value": eps,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "APGD-L1", "L1 Epsilon", apgd_l1_results)
    all_results["apgd_l1"] = apgd_l1_results
    
    # =====================================================
    # ATTACK 3: APGD-L2
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: APGD-L2")
    print("#"*70)
    
    apgd_l2_results = {}
    for param_name, eps_max in APGD_L2_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"APGD-L2 | eps_max = {eps_max}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.APGD_L2(device, synthetic_model, correct_loader, eps_max)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "APGD_L2", "eps", eps_max)
        
        # Store results
        apgd_l2_results[param_name] = {
            "param_value": eps_max,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "APGD-L2", "L2 Epsilon", apgd_l2_results)
    all_results["apgd_l2"] = apgd_l2_results
    
    # =====================================================
    # ATTACK 4: L0-PGD
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: L0-PGD")
    print("#"*70)
    
    l0_pgd_results = {}
    for param_name, sparsity in L0_PGD_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"L0-PGD | sparsity = {sparsity}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.L0_PGD(device, synthetic_model, correct_loader, sparsity)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "L0_PGD", "k", sparsity)
        
        # Store results
        l0_pgd_results[param_name] = {
            "param_value": sparsity,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "L0-PGD", "Sparsity(k)", l0_pgd_results)
    all_results["l0_pgd"] = l0_pgd_results
    
    # =====================================================
    # ATTACK 5: L0+Linf PGD
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: L0+Linf PGD")
    print("#"*70)
    
    l0_linf_results = {}
    for param_name, eps in L0_LINF_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"L0+Linf PGD | eps = {eps}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.L0_Linf_PGD(device, synthetic_model, correct_loader, eps)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "L0_Linf_PGD", "eps", eps)
        
        # Store results
        l0_linf_results[param_name] = {
            "param_value": eps,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "L0+Linf PGD", "Epsilon", l0_linf_results)
    all_results["l0_linf"] = l0_linf_results
    
    # =====================================================
    # ATTACK 6: L0+Sigma PGD
    # =====================================================
    print("\n" + "#"*70)
    print("# RUNNING: L0+Sigma PGD")
    print("#"*70)
    
    l0_sigma_results = {}
    for param_name, sparsity in L0_SIGMA_PARAMS.items():
        print(f"\n{'='*60}")
        print(f"L0+Sigma PGD | sparsity = {sparsity}")
        print(f"{'='*60}")
        
        # Run attack
        adv_loader = AttackFactory.L0_Sigma_PGD(device, synthetic_model, correct_loader, sparsity)
        
        # Evaluate
        syn_acc, oracle_acc = evaluate_attack(adv_loader, synthetic_model, oracle, device, num_classes)
        
        # Save samples
        save_adv_samples(adv_loader, "L0_Sigma_PGD", "k", sparsity)
        
        # Store results
        l0_sigma_results[param_name] = {
            "param_value": sparsity,
            "syn_acc": syn_acc,
            "oracle_acc": oracle_acc,
        }
    
    write_attack_results_to_file(results_file, "L0+Sigma PGD", "Sparsity(k)", l0_sigma_results)
    all_results["l0_sigma"] = l0_sigma_results
    
    return all_results