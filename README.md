# penetration_test

This repository is an **extended implementation** of our WEBIST work and codebase:

- **Base (WEBIST paper) repository:** https://github.com/cxjuan/penetration_test

The original repository accompanies the WEBIST paper implementation and experiments.  
This repository extends the WEBIST codebase with additional algorithms, training/evaluation utilities, and implementation details aimed at more systematic large-scale experiments.

Built on the open-source penetration testing benchmark **NetworkAttackSimulator (NASim)**:  
https://github.com/Jjschwartz/NetworkAttackSimulator

---

## What is added in this extended repository (vs. the WEBIST repo)

Compared with the WEBIST repository above, this extension studies **the impact of Lagrangian multiplier update rules** beyond the paper’s baseline constraint-handling configuration, including:

- **Scalar multiplier updates** (projected Lagrangian updates)
- **Hybrid-neural multiplier updates** (neural update with explicit structure/constraints)
- **Pure-neural multiplier updates** (fully learned multiplier dynamics)

These additions enable controlled studies of **effectiveness–efficiency trade-offs** under different multiplier-update rules.

> For strict reproducibility, keep the WEBIST repo as the “minimal reference implementation,”  
> and treat this repo as the “research/experimental extension.”

---

## Associated Publication

This repository supports the findings of the following papers:

> **Multi-Objective Policy Optimization for Effective and Cost-Conscious Penetration Testing**  
> Authors: Xiaojuan Cai, Lulu Zhu, Zhuo Li, Hiroshi Koide  
> Citation: Cai, X.; Zhu, L.; Li, Z.; Koide, H. (2025). *Multi-Objective Policy Optimization for Effective and Cost-Conscious Penetration Testing.* In Proceedings of the 21st International Conference on Web Information Systems and Technologies, pp. 488–499.

> **Effective and Cost-Aware Penetration Testing via Lagrangian Multi-Objective Reinforcement Learning: Scalar, Hybrid, and Pure Neural Multiplier Updates**  
> Authors: Xiaojuan Cai, Lulu Zhu, Zhuo Li, Hiroshi Koide
