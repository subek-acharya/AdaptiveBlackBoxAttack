try:
    # prefer a class already named AutoAttack
    from .autoattack import AutoAttack
except ImportError:
    # fallback: export AutoAttackL1 under the public name AutoAttack
    from .autoattack import AutoAttackL1 as AutoAttack
