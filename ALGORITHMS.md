# Robot-learning algorithm programme

Status: starter objectives and policy-network primitives, not trained algorithms or reported comparisons. Human approval on 2026-10-25 expanded exploration beyond PPO; E6/E7 training still follows the MVR, metric-review, and compute gates in SCOPE.md.

## Implemented foundations
- REINFORCE/A2C/PPO: policy-gradient and clipped-surrogate objectives with detached returns/advantages/old policies.
- Robot DPO-style objective: preferred vs rejected trajectory log-likelihood ratios relative to a frozen reference policy. Pairs must share the same initial/context specification; sum the actual robot-action log-probabilities over each trajectory. This is offline preference optimisation, not an online exploration algorithm.
- Robot GRPO-style objective: grouped rollouts from the same task context, group-relative rewards, valid-action masks, clipped ratios, and reference-policy KL estimation. Equal-reward groups have zero reward advantages, not NaNs.
- Behaviour cloning: an explicit demonstrator-labelled offline baseline.
- IQL expectile value loss: a building block only; replay, advantage-weighted actor fitting, and the complete IQL trainer are not implemented.
- Categorical robot actor-critic: a small network for discrete bounded turn-rate commands. This is untrained source code, not a learned controller used in the starter demo.

See src/lcert/rl/objectives.py, policies.py, and catalog.py. Numerical smoke checks use synthetic inputs and gradients only; they are not model/policy training or performance evidence.

## Registered next algorithms
SAC and TD3 for continuous-control/off-policy comparisons; IQL and CQL for offline data; MAPPO and independent PPO for multi-agent comparisons. Their catalogue entries describe required data and critic contracts, and explicitly say when a full trainer is absent. Do not substitute one algorithm for another or claim a trainer exists because its name is registered.

## Experiment contract
- Use matched rewards, observation/action definitions, training budgets, independent seeds, and frozen evaluation scenarios.
- Keep action-command likelihoods distinct from shield-executed actions. On-policy, off-policy, preference, and group-relative datasets are not interchangeable.
- Preference labels must state whether they came from humans, a scripted evaluator, or synthetic reward comparisons; never call synthetic labels human feedback.
- Keep the reference policy frozen within a declared update/run, and record its checkpoint hash.
- Require grouped-context IDs for GRPO and paired-context IDs for DPO; do not normalise rewards across unrelated tasks.
- Keep data collection on training seeds/distributions. Final test access and any reportable rollout metrics need the reviewed evaluator and ledger logging.
- Compare the same frozen policy under no shield, nominal, uncertainty-aware, and labelled oracle shields before adding separately trained shield-aware policies.
- DPO/GRPO here are proposed robot-action adaptations of methods introduced for language models. No language-model training is introduced, and no safety/sample-efficiency improvement is assumed.

## Primary references
- PPO: https://arxiv.org/abs/1707.06347
- DPO: https://arxiv.org/abs/2305.18290
- GRPO introduction in DeepSeekMath: https://arxiv.org/abs/2402.03300
- SAC: https://arxiv.org/abs/1812.05905
- TD3: https://arxiv.org/abs/1802.09477
- CQL: https://arxiv.org/abs/2006.04779
- IQL: https://arxiv.org/abs/2110.06169
- Multi-agent PPO: https://arxiv.org/abs/2103.01955

These are method references, not evidence that the adaptations work for this project's robots. The full training/evaluation harness remains gated and must be verified before any claimed result.
