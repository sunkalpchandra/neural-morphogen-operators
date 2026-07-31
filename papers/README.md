# Reference papers

Ten papers surveyed to calibrate the writing, structure and figure conventions of
`paper/neurips_2026.tex`. Downloaded from arXiv; see `_downloaded.json` for the
arXiv IDs and `_analysis.json` for the extracted structure (page counts, figure
and table counts, abstract lengths, section headings).

| File | Why it was read |
|---|---|
| `message_passing_neural_pde_solvers.pdf` | canonical learned-PDE-solver paper; framing of stability and generalisation |
| `neural_operator_discovery_trajectories.pdf` | operator discovery from data; how operator claims are stated |
| `learning_system_params_turing_patterns.pdf` | inferring reaction--diffusion parameters from patterns |
| `ml_biochemical_spatial_patterns.pdf` | ML for spatiotemporal reaction--diffusion; inverse problems in Turing systems |
| `diffusion_to_reaction_diffusion_oversmoothing.pdf` | reaction--diffusion view of over-smoothing (directly relevant to our Moran's I result) |
| `position_biology_challenge_piml.pdf` | position paper on why physics-informed ML underdelivers in biology |
| `lattice_graph_ssl_spatial_omics.pdf` | graph self-supervised learning on spatial omics |
| `semanticst_spatial_graph_learning.pdf` | spatial transcriptomics graph clustering |
| `hypergraph_nn_spatial_domains.pdf` | hypergraph GNN for spatial domain identification |
| `feast_attention_spatial_transcriptomics.pdf` | attention architectures for spatial transcriptomics |

## What the survey changed in our paper

* **Abstract length.** The sample runs 149--260 words (median ~190). Ours was 305
  and discursive; it is now 243 and leads with the method and numbers.
* **Figure density.** These papers carry 3--14 figures. Ours had *one* in the main
  text; it now has seven, spread over pages 2--8, including a dataset-overview
  panel of the kind this literature opens with.
* **Register.** Declarative and compact rather than essayistic. Editorial asides
  ("the more useful half of the paper", "it fired on our own model") were cut or
  rewritten as plain statements of result.
* **Negative results.** `position_biology_challenge_piml` and
  `ml_biochemical_spatial_patterns` both foreground where the method does not work;
  that licensed keeping our four negatives in the abstract rather than burying them.
