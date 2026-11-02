# CREMA-D 2D Representation Visualizations (t-SNE)

## Methodological Disclaimer & Interpretation Guide

1. **Illustration Only**: These 2D t-SNE projections are generated strictly for qualitative visualization and exploratory illustration of high-dimensional feature and latent geometry.
2. **Distances & Density Caveats**: Non-linear manifold projections (t-SNE) do not preserve global pairwise Euclidean distances or relative cluster densities. Cluster sizes, cluster separations, and spatial distances across different regions are not directly interpretable as quantitative emotion proximity metrics.
3. **No Optimization on Visualizations**: No hyperparameters, architecture decisions, rebalancing strategies, or tuning procedures were guided, adjusted, or optimized based on these plots. All model selection was conducted strictly by validation Macro-F1 prior to visualization.
4. **Held-Out Test Display**: Test set embeddings are displayed strictly post-evaluation for qualitative inspection. No test data guided representation learning or sampling decisions.
5. **Deterministic Pipeline**: All panels utilize an identical, standardized decomposition pipeline:
   - Initial dimensionality reduction: `sklearn.decomposition.PCA(n_components=50, random_state=42)`
   - Manifold embedding: `sklearn.manifold.TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)`
   - Fixed emotion color mapping from `data/label_map.json`.

---

## Index of Figures

| Figure | Filename (PNG / PDF) | Split | Representation / Setting | Description |
| :--- | :--- | :---: | :--- | :--- |
| **Figure 1** | `fig1_settings_tsne` | Test | Fusion V2 across 4 Imbalance Settings | 2x2 grid comparing test latent geometry across balanced ($ho pprox 1.2$), $ho=10$, $ho=20$, and $ho=50$. |
| **Figure 2** | `fig2_modalities_tsne` | Test | Audio, Video, Concat, and Fused Latents | 4-panel comparison of modality geometry with original-space silhouette scores in panel titles. |
| **Figure 3** | `fig3_rcs_synthesis_tsne` | Train | Real vs. RCS Synthetic Latents ($ho=10, 50$) | Joint t-SNE embedding comparing real training points (circles) to synthetic minority samples (triangles). |
| **Figure 4** | `fig4_actor_check_tsne` | Test | Balanced Fusion V2 Latents | Side-by-side comparison colored by emotion class (6 emotions) vs. actor identity (13 test speakers). |
| **Figure 5** | `fig5_errors_rho50_tsne` | Test | $ho=50$ Test Predictions across Methods | Identical test embedding showing classification errors (red) vs. correct predictions (grey) for `no_rebalance`, `balanced_softmax`, and `RCS-Optimal`. |
