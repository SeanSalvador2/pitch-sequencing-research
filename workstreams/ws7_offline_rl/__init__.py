"""Workstream 7 -- conservative offline RL + OPE + exploitability (the capstone; SPEC 12.7, 10).

Reads the shared :mod:`pitchseq` foundation and WS3 artifacts (behavior propensities + q-hat;
decision D33), builds a conservative fitted-Q-iteration policy with a behavior-support pessimism
penalty (decision D48), evaluates it through the shared OPE gate with the full SPEC 9 battery
(behavior-recovery first, step-wise DR + refit-bootstrap FQE, D24 agreement), cross-checks against
WS5's tabular MDP, and caps the study with the SPEC 10 predictability / exploitability frontier
(decision D49). It re-implements no evaluation -- every value flows through :mod:`pitchseq.eval`.
"""
