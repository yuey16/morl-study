# Implementation sources

- CAPQL, MORL/D + MOSAC, and GPI-LS use
  [MORL-Baselines](https://github.com/LucasAlegre/morl-baselines), pinned in
  `requirements.txt`. Its upstream MIT license and notices remain with the
  installed dependency.
- Momba uses [the public Momba implementation](https://github.com/adamstafa/momba).
  `prepare.py` downloads the exact upstream files listed in `sources.json`;
  those files are not bundled in this source archive. The local code adds
  objective-specific critic branches and a small PyTorch training loop.
- MO-MPO is a TD(0) implementation of the per-objective E-step and Gaussian
  policy projection described in
  [Abdolmaleki et al., ICML 2020](https://proceedings.mlr.press/v119/abdolmaleki20a.html).
- Environment implementations come from
  [MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium).

Dependency licenses are not replaced by this package.
