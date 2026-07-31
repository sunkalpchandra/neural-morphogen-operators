# Reference papers

35 papers surveyed to calibrate the writing, structure and figure conventions of
`paper/neurips_2026.tex`. Downloaded from arXiv; see `_downloaded.json` for the
arXiv IDs and `_analysis.json` for the extracted structure (page counts, figure
and table counts, abstract lengths, section headings).

| File | Why it was read |
|---|---|
| `message_passing_neural_pde_solvers.pdf` | PDE/operator learning |
| `learning_system_params_turing_patterns.pdf` | Turing patterns + learning |
| `ml_biochemical_spatial_patterns.pdf` | ML for spatiotemporal reaction-diffusion |
| `position_biology_challenge_piml.pdf` | position: PIML for biology |
| `diffusion_to_reaction_diffusion_oversmoothing.pdf` | RD view of oversmoothing |
| `lattice_graph_ssl_spatial_omics.pdf` | spatial omics graph SSL |
| `semanticst_spatial_graph_learning.pdf` | spatial transcriptomics clustering |
| `hypergraph_nn_spatial_domains.pdf` | hypergraph NN spatial domains |
| `feast_attention_spatial_transcriptomics.pdf` | attention for ST |
| `neural_operator_discovery_trajectories.pdf` | neural operator discovery |
| `fno_universal_approximation_error_bounds.pdf` | FNO approximation theory |
| `fno_discretization_error.pdf` | discretisation error of FNO |
| `operator_learning_theory_tour.pdf` | convergence rates for operator learning |
| `cnn_operator_approximation_bounds.pdf` | operator approximation bounds |
| `generalization_multi_input_operator.pdf` | generalisation for operator learning |
| `modified_strang_splitting_parabolic.pdf` | Strang splitting, semilinear parabolic |
| `strang_unconditional_energy_dissipation.pdf` | unconditional dissipation of Strang splitting |
| `pattern_formation_anisotropic_diffusion.pdf` | Turing patterns under anisotropic diffusion |
| `wellposedness_inverse_identification_nonlocal.pdf` | inverse identification, well-posedness |
| `determining_nonlinear_balance_laws.pdf` | identifiability of nonlinear PDE terms |
| `gradient_stability_nonlinear_pde_inference.pdf` | inference in nonlinear PDE models |
| `gread_graph_reaction_diffusion.pdf` | reaction-diffusion GNNs |
| `grand_graph_neural_diffusion.pdf` | graph neural diffusion |
| `oversmoothing_diffusion_gnn.pdf` | oversmoothing in diffusion GNNs |
| `trajectorynet_optimal_transport_cells.pdf` | dynamic OT for cell trajectories |
| `wasserstein_lagrangian_flows.pdf` | Wasserstein Lagrangian flows |
| `joint_velocity_growth_flow_matching.pdf` | single-cell dynamics via flow matching |
| `stabilized_neural_odes_longtime.pdf` | stability of neural ODEs |
| `semi_implicit_neural_odes.pdf` | semi-implicit neural ODEs |
| `multigroup_gaussian_processes.pdf` | multi-group GPs for expression |
| `cell_cell_communication_inference.pdf` | spatial cell-cell communication |
| `range_of_cell_cell_communication.pdf` | tuning communication range |
| `multimodal_singlecell_foundation.pdf` | single-cell foundation models |
| `dimino_dimension_informed_operator.pdf` | dimension-informed neural operators |
| `operator_learning_domain_decomposition.pdf` | geometry generalisation in operator learning |

## What the survey changed in the manuscript

* **Register.** The sampled papers are uniformly impersonal. The manuscript now
  contains **zero** first-person pronouns in either main text or appendices.
* **Abstract length.** Sample range 149--260 words (median ~190); the abstract is
  243 words, down from 305.
* **Figure density.** Sample carries 3--14 figures; the main text carries six,
  spread over pages 2--7, with three more in the appendices.
* **Formal apparatus.** The theory papers surveyed (`fno_universal_approximation`,
  `strang_unconditional_energy_dissipation`, `determining_nonlinear_balance_laws`)
  average >20 numbered environments each. The manuscript now states 1 theorem,
  14 propositions, 1 lemma, 1 corollary, 2 definitions and 14 proofs across
  13 displayed equations, with the substantive derivations in
  Appendices A--H.
* **Verification.** `scripts/verify_theory.py` checks every analytical claim
  against the implementation under adversarially perturbed parameters and emits
  Table 3; no proposition appears without a corresponding executable check.
