"""
Eval Category Cluster Map — Semantic Embeddings Version

Uses sentence-transformers (all-mpnet-base-v2) to embed eval set descriptions,
then clusters and visualizes them. This captures meaning-level similarity
rather than just lexical overlap.
"""

import csv
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgba
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    import plotly.express as px
except Exception:
    px = None

# ── Load data ─────────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "Official Eval Categories + Sets-import to sheets.csv"

rows = []
with open(csv_path, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eval_set = row["Eval Set"].strip()
        category = row["Eval Category"].strip()
        if eval_set and category and category.lower() != "none":
            rows.append({"eval_set": eval_set, "category": category})

print(f"Loaded {len(rows)} eval sets across {len(set(r['category'] for r in rows))} categories.\n")

descriptions = [r["eval_set"] for r in rows]
categories   = [r["category"] for r in rows]

# ── Embeddings (semantic with robust fallback) ───────────────────────────
embedding_source = "Sentence Embeddings (all-mpnet-base-v2)"
try:
    print("Loading sentence-transformers model (all-mpnet-base-v2)...")
    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(descriptions, show_progress_bar=True, normalize_embeddings=True)
except Exception as exc:
    print(f"Could not load semantic model, falling back to TF-IDF embeddings: {exc}")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500,
        min_df=1,
    )
    embeddings = vectorizer.fit_transform(descriptions).toarray()
    embedding_source = "TF-IDF Fallback"

print(f"Embedding matrix shape: {embeddings.shape}")
print(f"Embedding source: {embedding_source}\n")

# ── Cosine similarity ────────────────────────────────────────────────────
cos_sim = cosine_similarity(embeddings)

# ── Hierarchical clustering ──────────────────────────────────────────────
condensed_dist = pdist(embeddings, metric="cosine")
condensed_dist = np.clip(condensed_dist, 0, None)
Z = linkage(condensed_dist, method="ward")

# ── t-SNE for 2D projection ─────────────────────────────────────────────
perplexity = min(30, len(descriptions) - 1)
tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    random_state=42,
    metric="cosine",
    init="random",
    max_iter=2000,
)
coords_2d = tsne.fit_transform(embeddings)

# ── Color map ─────────────────────────────────────────────────────────────
unique_cats = sorted(set(categories))
cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_cats))

# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 1: t-SNE scatter — convex hulls + centroid labels
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(22, 16))
ax.set_facecolor("#fafafa")
fig.set_facecolor("white")

centroid_positions = {}

for idx, cat in enumerate(unique_cats):
    mask = [i for i, c in enumerate(categories) if c == cat]
    pts = coords_2d[mask]
    color = cmap(idx)

    # Convex hull shading
    if len(pts) >= 3:
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                    color=to_rgba(color, 0.12), zorder=0)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color=to_rgba(color, 0.35), linewidth=1, linestyle="--", zorder=1)
        except Exception:
            pass

    ax.scatter(
        pts[:, 0], pts[:, 1],
        c=[color],
        label=f"{cat} ({len(mask)})",
        s=60,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.6,
        zorder=2,
    )

    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    centroid_positions[cat] = (cx, cy, len(mask))

# Centroid labels
for cat, (cx, cy, cnt) in centroid_positions.items():
    idx = cat_to_idx[cat]
    color = cmap(idx)
    wrapped = "\n".join(textwrap.wrap(cat, width=18))
    ax.annotate(
        wrapped,
        (cx, cy),
        fontsize=7,
        fontweight="bold",
        color=to_rgba(color, 1.0),
        ha="center", va="center",
        zorder=3,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=to_rgba(color, 0.5),
            alpha=0.85,
            linewidth=0.8,
        ),
    )

ax.set_title(f"Eval Categories — Cluster Map ({embedding_source}, t-SNE)",
             fontsize=18, fontweight="bold", pad=20)
ax.set_xlabel("t-SNE 1", fontsize=11)
ax.set_ylabel("t-SNE 2", fontsize=11)

handles, labels = ax.get_legend_handles_labels()
sorted_pairs = sorted(zip(labels, handles), key=lambda x: -int(x[0].split("(")[-1].rstrip(")")))
sorted_labels, sorted_handles = zip(*sorted_pairs)
ax.legend(
    sorted_handles, sorted_labels,
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    fontsize=7.5,
    title="Eval Category (count)",
    title_fontsize=10,
    framealpha=0.9,
)

ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
fig.savefig("eval_semantic_tsne.png", dpi=200, bbox_inches="tight")
print("Saved: eval_semantic_tsne.png")
plt.close()

# ── Interactive Plotly t-SNE map ──────────────────────────────────────────
if px is not None:
    cat_counts = Counter(categories)
    category_colors = {cat: to_hex(cmap(cat_to_idx[cat])) for cat in unique_cats}
    df_plot = pd.DataFrame(
        {
            "tsne_x": coords_2d[:, 0],
            "tsne_y": coords_2d[:, 1],
            "eval_set": descriptions,
            "category": categories,
            "category_count": [cat_counts[c] for c in categories],
        }
    )

    fig_interactive = px.scatter(
        df_plot,
        x="tsne_x",
        y="tsne_y",
        color="category",
        color_discrete_map=category_colors,
        hover_name="eval_set",
        hover_data={
            "category": True,
            "category_count": True,
            "tsne_x": ":.2f",
            "tsne_y": ":.2f",
        },
        title=f"Eval Categories — {embedding_source} t-SNE (Interactive)",
    )
    fig_interactive.update_traces(
        marker=dict(size=10, opacity=0.86, line=dict(width=0.6, color="white"))
    )
    fig_interactive.update_layout(
        width=1500,
        height=950,
        template="plotly_white",
        legend_title_text="Eval Category",
    )
    fig_interactive.update_xaxes(visible=False, title=None)
    fig_interactive.update_yaxes(visible=False, title=None, scaleanchor="x", scaleratio=1)

    html_out = Path("eval_semantic_tsne_interactive.html")
    fig_interactive.write_html(
        html_out,
        include_plotlyjs="inline",
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {"filename": "eval_semantic_tsne_interactive"},
        },
    )
    print(f"Saved: {html_out}")
else:
    print("Skipping interactive Plotly map: plotly is not available in this environment.")

# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 2: Dendrogram
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(26, 10))
short_labels = [d[:60] + "…" if len(d) > 60 else d for d in descriptions]
dn = dendrogram(
    Z,
    labels=short_labels,
    leaf_rotation=90,
    leaf_font_size=4.5,
    ax=ax,
    color_threshold=0.7 * max(Z[:, 2]),
)
ax.set_title("Eval Categories — Semantic Hierarchical Clustering", fontsize=16, fontweight="bold")
ax.set_ylabel("Ward Distance (cosine)")
plt.tight_layout()
fig.savefig("eval_semantic_dendrogram.png", dpi=200, bbox_inches="tight")
print("Saved: eval_semantic_dendrogram.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 3: Inter-category heatmap — hierarchically clustered (seaborn clustermap)
# ═══════════════════════════════════════════════════════════════════════════
import seaborn as sns

cat_sim = np.zeros((len(unique_cats), len(unique_cats)))
for i, cat_a in enumerate(unique_cats):
    idxs_a = [j for j, c in enumerate(categories) if c == cat_a]
    for k, cat_b in enumerate(unique_cats):
        idxs_b = [j for j, c in enumerate(categories) if c == cat_b]
        sims = cos_sim[np.ix_(idxs_a, idxs_b)]
        if i == k:
            mask_diag = ~np.eye(sims.shape[0], dtype=bool) if sims.shape[0] > 1 else np.ones_like(sims, dtype=bool)
            cat_sim[i, k] = sims[mask_diag].mean() if mask_diag.any() else 1.0
        else:
            cat_sim[i, k] = sims.mean()

df_sim = pd.DataFrame(cat_sim, index=unique_cats, columns=unique_cats)

g = sns.clustermap(
    df_sim,
    method="ward",
    metric="euclidean",
    cmap="YlOrRd",
    vmin=0,
    vmax=df_sim.values.max(),
    annot=True,
    fmt=".3f",
    annot_kws={"size": 6},
    linewidths=0.5,
    linecolor="white",
    figsize=(16, 14),
    dendrogram_ratio=(0.12, 0.12),
    cbar_kws={"label": "Avg Cosine Similarity", "shrink": 0.6},
)

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=45, ha="right")
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=8, rotation=0)
g.figure.suptitle("Inter-Category Semantic Similarity (Hierarchically Clustered)",
                   fontsize=15, fontweight="bold", y=1.02)
g.figure.savefig("eval_semantic_heatmap.png", dpi=200, bbox_inches="tight")
print("Saved: eval_semantic_heatmap.png")
plt.close()

# ── Interactive Plotly heatmap ────────────────────────────────────────────
if px is not None:
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import leaves_list

    # Reorder rows/cols by dendrogram leaf order (same as seaborn clustermap)
    row_linkage = linkage(pdist(df_sim.values, metric="euclidean"), method="ward")
    leaf_order = leaves_list(row_linkage)
    ordered_cats = [unique_cats[i] for i in leaf_order]
    df_ordered = df_sim.loc[ordered_cats, ordered_cats]

    # Build hover text: "Row X vs Col Y: 0.XXX"
    hover_text = []
    for i, row_cat in enumerate(ordered_cats):
        hover_row = []
        for j, col_cat in enumerate(ordered_cats):
            val = df_ordered.iloc[i, j]
            hover_row.append(
                f"<b>{row_cat}</b> vs <b>{col_cat}</b><br>"
                f"Cosine Similarity: <b>{val:.4f}</b>"
            )
        hover_text.append(hover_row)

    fig_hm = go.Figure(data=go.Heatmap(
        z=df_ordered.values,
        x=ordered_cats,
        y=ordered_cats,
        colorscale="YlOrRd",
        zmin=0,
        zmax=float(df_ordered.values.max()),
        text=[[f"{v:.3f}" for v in row] for row in df_ordered.values],
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertext=hover_text,
        hoverinfo="text",
        colorbar=dict(title=dict(text="Avg Cosine<br>Similarity", side="right")),
    ))

    fig_hm.update_layout(
        title=dict(
            text="Inter-Category Semantic Similarity (Hierarchically Ordered)",
            font=dict(size=18),
        ),
        width=1000,
        height=900,
        template="plotly_white",
        xaxis=dict(
            tickfont=dict(size=10),
            tickangle=45,
            side="bottom",
        ),
        yaxis=dict(
            tickfont=dict(size=10),
            autorange="reversed",
        ),
        margin=dict(l=200, b=200, t=80, r=80),
    )

    html_hm = Path("eval_semantic_heatmap_interactive.html")
    fig_hm.write_html(
        html_hm,
        include_plotlyjs="inline",
        full_html=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {"filename": "eval_semantic_heatmap"},
        },
    )
    print(f"Saved: {html_hm}")

# ═══════════════════════════════════════════════════════════════════════════
#  GAP ANALYSIS PLOTS (all interactive Plotly)
# ═══════════════════════════════════════════════════════════════════════════
if px is not None:
    import plotly.graph_objects as go
    from scipy.stats import gaussian_kde
    import itertools

    PLOTLY_CONFIG = {
        "displaylogo": False,
        "scrollZoom": True,
    }
    cat_counts_map = Counter(categories)
    category_colors = {cat: to_hex(cmap(cat_to_idx[cat])) for cat in unique_cats}

    # ── PLOT G1: Coverage Density Cluster Map ─────────────────────────────
    # KDE contour over t-SNE with scatter overlay. Cold zones = gaps.
    kde = gaussian_kde(coords_2d.T, bw_method=0.3)
    pad = 5
    xgrid = np.linspace(coords_2d[:, 0].min() - pad, coords_2d[:, 0].max() + pad, 200)
    ygrid = np.linspace(coords_2d[:, 1].min() - pad, coords_2d[:, 1].max() + pad, 200)
    Xg, Yg = np.meshgrid(xgrid, ygrid)
    Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)

    fig_density = go.Figure()

    # Contour background (reversed so low density = warm/visible)
    fig_density.add_trace(go.Contour(
        x=xgrid, y=ygrid, z=Zg,
        colorscale="YlOrRd",
        reversescale=False,
        opacity=0.6,
        contours=dict(showlabels=False),
        colorbar=dict(title=dict(text="Density", side="right"),
                      x=-0.08, len=0.5, y=0.5),
        hoverinfo="skip",
        name="Density",
        showlegend=False,
    ))

    # Scatter points by category
    for cat in unique_cats:
        mask = [i for i, c in enumerate(categories) if c == cat]
        fig_density.add_trace(go.Scatter(
            x=coords_2d[mask, 0],
            y=coords_2d[mask, 1],
            mode="markers",
            name=f"{cat} ({len(mask)})",
            marker=dict(size=8, color=category_colors[cat],
                        line=dict(width=0.5, color="white")),
            text=[descriptions[i] for i in mask],
            hovertemplate="<b>%{text}</b><br>Category: " + cat + "<extra></extra>",
        ))

    # Find sparse zones inside the data hull and mark them with gap markers
    from scipy.spatial import Delaunay
    try:
        hull = Delaunay(coords_2d)
        grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
        inside = hull.find_simplex(grid_pts) >= 0
        Zg_flat = Zg.ravel()
        Zg_inside = np.where(inside, Zg_flat, np.inf)

        # Get candidate sparse points, then cluster them to avoid duplicates
        candidate_idxs = np.argsort(Zg_inside)[:50]
        candidate_pts = grid_pts[candidate_idxs]
        # Greedily pick up to 8 sparse zones that are at least min_sep apart
        min_sep = (coords_2d.max() - coords_2d.min()) * 0.12
        chosen = []
        for ci in range(len(candidate_pts)):
            pt = candidate_pts[ci]
            too_close = False
            for prev in chosen:
                if np.linalg.norm(pt - prev) < min_sep:
                    too_close = True
                    break
            if not too_close:
                chosen.append(pt)
            if len(chosen) >= 8:
                break

        # Plot sparse zone markers with informative hover
        gap_xs, gap_ys, gap_hovers = [], [], []
        for pt in chosen:
            gx, gy = pt
            dists_to_pts = np.linalg.norm(coords_2d - np.array([gx, gy]), axis=1)
            # Find 3 nearest eval sets to describe neighborhood
            nearest_3 = np.argsort(dists_to_pts)[:3]
            neighbor_cats = set(categories[n] for n in nearest_3)
            neighbor_lines = "<br>".join(
                f"• [{categories[n]}] {descriptions[n][:60]}" for n in nearest_3
            )
            hover = (
                f"<b>SPARSE ZONE</b><br>"
                f"Nearest categories: {', '.join(sorted(neighbor_cats))}<br><br>"
                f"Nearest eval sets:<br>{neighbor_lines}"
            )
            gap_xs.append(gx)
            gap_ys.append(gy)
            gap_hovers.append(hover)

        fig_density.add_trace(go.Scatter(
            x=gap_xs, y=gap_ys,
            mode="markers",
            name="Sparse zones (gaps)",
            marker=dict(
                size=22,
                color="rgba(211,47,47,0.25)",
                symbol="x",
                line=dict(width=2.5, color="#d32f2f"),
            ),
            text=gap_hovers,
            hovertemplate="%{text}<extra></extra>",
        ))
    except Exception:
        pass

    fig_density.update_layout(
        title="Coverage Density Cluster Map — Cold Zones = Gaps",
        width=1500, height=950,
        template="plotly_white",
        legend=dict(title="Category (count)", font=dict(size=9),
                    x=1.02, y=1, xanchor="left"),
        margin=dict(l=100, r=50, t=80, b=50),
    )
    fig_density.update_xaxes(visible=False)
    fig_density.update_yaxes(visible=False, scaleanchor="x")

    fig_density.write_html("eval_gap_density.html", include_plotlyjs="inline",
                           full_html=True, config=PLOTLY_CONFIG)
    print("Saved: eval_gap_density.html")

    # ── PLOT G2: Boundary Gap Cluster Map ─────────────────────────────────
    # Midpoints between category centroids, sized by gap distance.
    centroids = {}
    for cat in unique_cats:
        mask = [i for i, c in enumerate(categories) if c == cat]
        centroids[cat] = coords_2d[mask].mean(axis=0)

    fig_boundary = go.Figure()

    # Background: all eval points (small, transparent)
    for cat in unique_cats:
        mask = [i for i, c in enumerate(categories) if c == cat]
        fig_boundary.add_trace(go.Scatter(
            x=coords_2d[mask, 0], y=coords_2d[mask, 1],
            mode="markers",
            name=f"{cat} ({len(mask)})",
            marker=dict(size=7, color=category_colors[cat], opacity=0.5,
                        line=dict(width=0.3, color="white")),
            text=[descriptions[i] for i in mask],
            hovertemplate="<b>%{text}</b><br>Category: " + cat + "<extra></extra>",
        ))

    # Compute gap midpoints
    gap_data = []
    for (cat_a, ca), (cat_b, cb) in itertools.combinations(centroids.items(), 2):
        midpoint = (ca + cb) / 2
        dists = np.linalg.norm(coords_2d - midpoint, axis=1)
        gap_size = float(dists.min())
        nearest_idx = int(dists.argmin())
        # Also find nearest from each category
        mask_a = [i for i, c in enumerate(categories) if c == cat_a]
        mask_b = [i for i, c in enumerate(categories) if c == cat_b]
        dist_a = np.linalg.norm(coords_2d[mask_a] - midpoint, axis=1)
        dist_b = np.linalg.norm(coords_2d[mask_b] - midpoint, axis=1)
        nearest_a = descriptions[mask_a[dist_a.argmin()]]
        nearest_b = descriptions[mask_b[dist_b.argmin()]]
        gap_data.append({
            "x": midpoint[0], "y": midpoint[1],
            "gap_size": gap_size,
            "cat_a": cat_a, "cat_b": cat_b,
            "nearest_a": nearest_a[:80], "nearest_b": nearest_b[:80],
        })

    gap_data.sort(key=lambda g: g["gap_size"], reverse=True)
    # Show top 25 largest gaps
    top_gaps = gap_data[:25]

    gap_sizes = [g["gap_size"] for g in top_gaps]
    max_gap = max(gap_sizes) if gap_sizes else 1
    min_gap = min(gap_sizes) if gap_sizes else 0

    fig_boundary.add_trace(go.Scatter(
        x=[g["x"] for g in top_gaps],
        y=[g["y"] for g in top_gaps],
        mode="markers",
        name="Gap markers",
        marker=dict(
            size=[8 + 25 * (g["gap_size"] - min_gap) / (max_gap - min_gap + 1e-9) for g in top_gaps],
            color=[g["gap_size"] for g in top_gaps],
            colorscale="RdYlGn_r",
            symbol="diamond",
            line=dict(width=1, color="black"),
            colorbar=dict(title=dict(text="Gap Size", side="right"), x=1.08),
        ),
        text=[f"Gap between:<br><b>{g['cat_a']}</b> and <b>{g['cat_b']}</b><br>"
              f"Gap size: {g['gap_size']:.2f}<br><br>"
              f"Nearest from {g['cat_a']}:<br>{g['nearest_a']}<br><br>"
              f"Nearest from {g['cat_b']}:<br>{g['nearest_b']}"
              for g in top_gaps],
        hovertemplate="%{text}<extra></extra>",
    ))

    fig_boundary.update_layout(
        title="Boundary Gap Cluster Map — Large Diamonds = Missing Coverage Between Categories",
        width=1400, height=950,
        template="plotly_white",
        legend=dict(title="Category (count)", font=dict(size=9)),
    )
    fig_boundary.update_xaxes(visible=False)
    fig_boundary.update_yaxes(visible=False, scaleanchor="x")

    fig_boundary.write_html("eval_gap_boundaries.html", include_plotlyjs="inline",
                            full_html=True, config=PLOTLY_CONFIG)
    print("Saved: eval_gap_boundaries.html")

    # ── PLOT G3: Size vs. Semantic Breadth Quadrant ───────────────────────
    cat_stats = []
    for cat in unique_cats:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        count = len(idxs)
        if len(idxs) >= 2:
            pairwise_dists = [1 - cos_sim[i, j] for i in idxs for j in idxs if i < j]
            breadth = float(np.mean(pairwise_dists))
        else:
            breadth = 0.0
        # Mean similarity to all other categories (centrality)
        cat_idx = cat_to_idx[cat]
        cross_sims = [cat_sim[cat_idx, j] for j in range(len(unique_cats)) if j != cat_idx]
        centrality = float(np.mean(cross_sims)) if cross_sims else 0
        cat_stats.append({
            "category": cat, "count": count,
            "breadth": breadth, "centrality": centrality,
            "color": category_colors[cat],
        })

    counts = [s["count"] for s in cat_stats]
    breadths = [s["breadth"] for s in cat_stats]
    median_count = float(np.median(counts))
    median_breadth = float(np.median(breadths))

    fig_quad = go.Figure()

    # Quadrant shading
    x_range = [0, max(counts) + 5]
    y_range = [0, max(breadths) * 1.15]

    # Top-left quadrant (broad + small = UNDER-INVESTED GAP)
    fig_quad.add_shape(type="rect",
        x0=x_range[0], x1=median_count, y0=median_breadth, y1=y_range[1],
        fillcolor="rgba(255,0,0,0.06)", line=dict(width=0))
    fig_quad.add_annotation(x=median_count * 0.3, y=y_range[1] * 0.95,
        text="UNDER-INVESTED<br>(broad topic, few evals)", showarrow=False,
        font=dict(size=11, color="#c62828"), bgcolor="rgba(255,255,255,0.7)")

    # Top-right (broad + large = OVERLOADED)
    fig_quad.add_shape(type="rect",
        x0=median_count, x1=x_range[1], y0=median_breadth, y1=y_range[1],
        fillcolor="rgba(255,165,0,0.06)", line=dict(width=0))
    fig_quad.add_annotation(x=(median_count + x_range[1]) / 2, y=y_range[1] * 0.95,
        text="OVERLOADED<br>(consider splitting)", showarrow=False,
        font=dict(size=11, color="#e65100"), bgcolor="rgba(255,255,255,0.7)")

    # Bottom-right (narrow + large = WELL-COVERED)
    fig_quad.add_shape(type="rect",
        x0=median_count, x1=x_range[1], y0=y_range[0], y1=median_breadth,
        fillcolor="rgba(0,128,0,0.06)", line=dict(width=0))
    fig_quad.add_annotation(x=(median_count + x_range[1]) / 2, y=y_range[0] + 0.02,
        text="WELL-COVERED<br>(focused & thorough)", showarrow=False,
        font=dict(size=11, color="#2e7d32"), bgcolor="rgba(255,255,255,0.7)")

    # Bottom-left (narrow + small = NICHE)
    fig_quad.add_shape(type="rect",
        x0=x_range[0], x1=median_count, y0=y_range[0], y1=median_breadth,
        fillcolor="rgba(100,100,255,0.06)", line=dict(width=0))
    fig_quad.add_annotation(x=median_count * 0.3, y=y_range[0] + 0.02,
        text="TIGHT NICHE<br>(may be fine or too narrow)", showarrow=False,
        font=dict(size=11, color="#1565c0"), bgcolor="rgba(255,255,255,0.7)")

    # Quadrant lines
    fig_quad.add_hline(y=median_breadth, line_dash="dash", line_color="gray", opacity=0.5)
    fig_quad.add_vline(x=median_count, line_dash="dash", line_color="gray", opacity=0.5)

    # Category points
    for s in cat_stats:
        fig_quad.add_trace(go.Scatter(
            x=[s["count"]], y=[s["breadth"]],
            mode="markers+text",
            name=s["category"],
            marker=dict(size=14 + s["centrality"] * 60, color=s["color"],
                        line=dict(width=1, color="white")),
            text=[s["category"]],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=(
                f"<b>{s['category']}</b><br>"
                f"Eval count: {s['count']}<br>"
                f"Semantic breadth: {s['breadth']:.3f}<br>"
                f"Cross-category centrality: {s['centrality']:.3f}"
                "<extra></extra>"
            ),
        ))

    fig_quad.update_layout(
        title="Category Size vs. Semantic Breadth — Top-Left = Biggest Coverage Gaps",
        xaxis_title="Number of Eval Sets",
        yaxis_title="Semantic Breadth (avg intra-category cosine distance)",
        width=1100, height=800,
        template="plotly_white",
        showlegend=False,
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
    )

    fig_quad.write_html("eval_gap_size_breadth.html", include_plotlyjs="inline",
                        full_html=True, config=PLOTLY_CONFIG)
    print("Saved: eval_gap_size_breadth.html")

    # ── PLOT G4: Category Coherence Dumbbell ──────────────────────────────
    coherence_data = []
    for cat in unique_cats:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        if len(idxs) < 2:
            coherence_data.append({
                "category": cat, "count": len(idxs),
                "min_sim": 0, "mean_sim": 0, "max_sim": 0,
                "min_pair": ("N/A", "N/A"), "max_pair": ("N/A", "N/A"),
            })
            continue
        sims = []
        pairs_detail = []
        for ii, a in enumerate(idxs):
            for b in idxs[ii + 1:]:
                s = cos_sim[a, b]
                sims.append(s)
                pairs_detail.append((s, a, b))
        pairs_detail.sort(key=lambda x: x[0])
        min_s, min_a, min_b = pairs_detail[0]
        max_s, max_a, max_b = pairs_detail[-1]
        coherence_data.append({
            "category": cat, "count": len(idxs),
            "min_sim": float(min_s), "mean_sim": float(np.mean(sims)), "max_sim": float(max_s),
            "min_pair": (descriptions[min_a][:70], descriptions[min_b][:70]),
            "max_pair": (descriptions[max_a][:70], descriptions[max_b][:70]),
        })

    coherence_data.sort(key=lambda x: x["mean_sim"])

    fig_dumbbell = go.Figure()

    for i, d in enumerate(coherence_data):
        color = category_colors[d["category"]]
        spread = d["max_sim"] - d["min_sim"]
        line_color = f"rgba(211,47,47,{min(1.0, 0.3 + spread)})" if spread > 0.4 else \
                     f"rgba(255,152,0,{min(1.0, 0.3 + spread)})" if spread > 0.25 else \
                     f"rgba(56,142,60,{min(1.0, 0.4 + spread)})"

        # Range line
        fig_dumbbell.add_trace(go.Scatter(
            x=[d["min_sim"], d["max_sim"]],
            y=[i, i],
            mode="lines",
            line=dict(color=line_color, width=3),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Min marker
        fig_dumbbell.add_trace(go.Scatter(
            x=[d["min_sim"]], y=[i],
            mode="markers",
            marker=dict(size=10, color=color, symbol="triangle-left",
                        line=dict(width=1, color="black")),
            showlegend=False,
            hovertemplate=(
                f"<b>{d['category']}</b> — MIN similarity: {d['min_sim']:.3f}<br><br>"
                f"Most dissimilar pair:<br>{d['min_pair'][0]}<br>vs<br>{d['min_pair'][1]}"
                "<extra></extra>"
            ),
        ))
        # Mean marker
        fig_dumbbell.add_trace(go.Scatter(
            x=[d["mean_sim"]], y=[i],
            mode="markers",
            marker=dict(size=12, color=color, symbol="circle",
                        line=dict(width=1, color="black")),
            showlegend=False,
            hovertemplate=(
                f"<b>{d['category']}</b> — MEAN similarity: {d['mean_sim']:.3f}<br>"
                f"Eval count: {d['count']}<br>Spread: {spread:.3f}"
                "<extra></extra>"
            ),
        ))
        # Max marker
        fig_dumbbell.add_trace(go.Scatter(
            x=[d["max_sim"]], y=[i],
            mode="markers",
            marker=dict(size=10, color=color, symbol="triangle-right",
                        line=dict(width=1, color="black")),
            showlegend=False,
            hovertemplate=(
                f"<b>{d['category']}</b> — MAX similarity: {d['max_sim']:.3f}<br><br>"
                f"Most similar pair:<br>{d['max_pair'][0]}<br>vs<br>{d['max_pair'][1]}"
                "<extra></extra>"
            ),
        ))

    fig_dumbbell.update_layout(
        title="Category Coherence — Wide Spread = Incoherent (Consider Splitting)",
        xaxis_title="Intra-Category Cosine Similarity",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(coherence_data))),
            ticktext=[f"{d['category']} ({d['count']})" for d in coherence_data],
            tickfont=dict(size=9),
        ),
        width=1200, height=700,
        template="plotly_white",
        margin=dict(l=280),
    )

    fig_dumbbell.write_html("eval_gap_coherence.html", include_plotlyjs="inline",
                            full_html=True, config=PLOTLY_CONFIG)
    print("Saved: eval_gap_coherence.html")

    # ── PLOT G5: Redundancy Network Graph ─────────────────────────────────
    fig_network = go.Figure()

    # Edges (draw first so they appear behind nodes)
    threshold = 0.20
    for i in range(len(unique_cats)):
        for j in range(i + 1, len(unique_cats)):
            sim = cat_sim[i, j]
            if sim >= threshold:
                ca = centroids[unique_cats[i]]
                cb = centroids[unique_cats[j]]
                # Find the most similar cross-category pair for hover
                idxs_a = [k for k, c in enumerate(categories) if c == unique_cats[i]]
                idxs_b = [k for k, c in enumerate(categories) if c == unique_cats[j]]
                best_sim, best_a, best_b = 0, 0, 0
                for a in idxs_a:
                    for b in idxs_b:
                        if cos_sim[a, b] > best_sim:
                            best_sim = cos_sim[a, b]
                            best_a, best_b = a, b

                opacity = min(1.0, 0.3 + (sim - threshold) * 3)
                width = 1 + (sim - threshold) * 20
                color = f"rgba(211,47,47,{opacity})" if sim > 0.35 else \
                        f"rgba(255,152,0,{opacity})" if sim > 0.25 else \
                        f"rgba(150,150,150,{opacity})"

                fig_network.add_trace(go.Scatter(
                    x=[ca[0], cb[0]], y=[ca[1], cb[1]],
                    mode="lines",
                    line=dict(width=width, color=color),
                    showlegend=False,
                    hoverinfo="text",
                    text=f"<b>{unique_cats[i]}</b> ↔ <b>{unique_cats[j]}</b><br>"
                         f"Avg similarity: {sim:.3f}<br><br>"
                         f"Most similar pair ({best_sim:.3f}):<br>"
                         f"• {descriptions[best_a][:70]}<br>"
                         f"• {descriptions[best_b][:70]}",
                ))

    # Nodes
    for cat in unique_cats:
        c = centroids[cat]
        count = cat_counts_map[cat]
        fig_network.add_trace(go.Scatter(
            x=[c[0]], y=[c[1]],
            mode="markers+text",
            name=cat,
            marker=dict(
                size=12 + count * 1.2,
                color=category_colors[cat],
                line=dict(width=2, color="white"),
            ),
            text=[f"{cat}\n({count})"],
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate=(
                f"<b>{cat}</b><br>"
                f"Eval count: {count}<br>"
                f"Avg internal similarity: {cat_sim[cat_to_idx[cat], cat_to_idx[cat]]:.3f}"
                "<extra></extra>"
            ),
        ))

    fig_network.update_layout(
        title=f"Category Redundancy Network — Edges Where Avg Similarity > {threshold}",
        width=1400, height=950,
        template="plotly_white",
        showlegend=False,
    )
    fig_network.update_xaxes(visible=False)
    fig_network.update_yaxes(visible=False, scaleanchor="x")

    fig_network.write_html("eval_gap_redundancy.html", include_plotlyjs="inline",
                           full_html=True, config=PLOTLY_CONFIG)
    print("Saved: eval_gap_redundancy.html")

    # ── PLOT G6: Item Isolation Lollipop (Top 30) ─────────────────────────
    isolation_data = []
    for i in range(len(descriptions)):
        same_cat = [j for j in range(len(descriptions))
                    if categories[j] == categories[i] and j != i]
        if same_cat:
            isolation = 1 - float(np.mean([cos_sim[i, j] for j in same_cat]))
        else:
            isolation = 1.0

        # Find top-3 nearest neighbors across ALL categories
        all_sims = [(cos_sim[i, j], j) for j in range(len(descriptions)) if j != i]
        all_sims.sort(reverse=True)
        top3 = all_sims[:3]

        isolation_data.append({
            "desc": descriptions[i],
            "category": categories[i],
            "isolation": isolation,
            "top3": [(float(s), descriptions[j], categories[j]) for s, j in top3],
        })

    isolation_data.sort(key=lambda x: x["isolation"], reverse=True)
    top_n = 30
    top_isolated = isolation_data[:top_n]

    fig_lollipop = go.Figure()

    for i, d in enumerate(top_isolated):
        color = category_colors[d["category"]]
        # Stem line
        fig_lollipop.add_trace(go.Scatter(
            x=[0, d["isolation"]], y=[i, i],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Dot
        neighbors_text = "<br>".join([
            f"  {s:.3f} [{c}] {desc[:60]}" for s, desc, c in d["top3"]
        ])
        fig_lollipop.add_trace(go.Scatter(
            x=[d["isolation"]], y=[i],
            mode="markers",
            marker=dict(size=10, color=color, line=dict(width=1, color="black")),
            showlegend=False,
            hovertemplate=(
                f"<b>{d['desc'][:80]}</b><br>"
                f"Category: {d['category']}<br>"
                f"Isolation score: {d['isolation']:.3f}<br><br>"
                f"Top-3 nearest neighbors (any category):<br>"
                f"{neighbors_text}"
                "<extra></extra>"
            ),
        ))

    fig_lollipop.update_layout(
        title=f"Top {top_n} Most Isolated Eval Sets — Orphaned or Misclassified?",
        xaxis_title="Isolation Score (1 − mean similarity to same-category peers)",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top_isolated))),
            ticktext=[f"[{d['category'][:20]}] {d['desc'][:55]}" for d in top_isolated],
            tickfont=dict(size=7),
            autorange="reversed",
        ),
        width=1300, height=900,
        template="plotly_white",
        margin=dict(l=420),
    )

    fig_lollipop.write_html("eval_gap_isolation.html", include_plotlyjs="inline",
                            full_html=True, config=PLOTLY_CONFIG)
    print("Saved: eval_gap_isolation.html")

    print("\nAll gap analysis plots generated.")

# ═══════════════════════════════════════════════════════════════════════════
#  COMPARISON: TF-IDF vs Semantic — what changed?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  SEMANTIC GAP ANALYSIS")
print("=" * 80)

# Category sizes
cat_counts = Counter(categories)
print("\nCategory sizes:")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[1]):
    print(f"   {cnt:3d}  {cat}")

# Most isolated (semantically)
avg_sim_per_item = []
for i in range(len(descriptions)):
    others = [cos_sim[i, j] for j in range(len(descriptions)) if j != i]
    avg_sim_per_item.append((np.mean(others), descriptions[i], categories[i]))

avg_sim_per_item.sort()
print("\nMost semantically isolated eval sets (lowest avg cosine sim to all others):")
for sim, desc, cat in avg_sim_per_item[:15]:
    print(f"   {sim:.4f}  [{cat}]  {desc[:85]}")

# Most semantically redundant pairs
print("\nMost semantically redundant pairs (highest cosine similarity):")
pairs = []
for i in range(len(descriptions)):
    for j in range(i + 1, len(descriptions)):
        pairs.append((cos_sim[i, j], i, j))
pairs.sort(reverse=True)
seen = set()
count = 0
for sim, i, j in pairs:
    if count >= 20:
        break
    key = (min(i, j), max(i, j))
    if key not in seen:
        seen.add(key)
        same_cat = "SAME" if categories[i] == categories[j] else "DIFF"
        print(f"   {sim:.4f} [{same_cat}]  [{categories[i]}] {descriptions[i][:65]}")
        print(f"                  [{categories[j]}] {descriptions[j][:65]}")
        print()
        count += 1

# Cross-category overlaps
print("Categories with highest semantic cross-similarity:")
cross_pairs = []
for i in range(len(unique_cats)):
    for j in range(i + 1, len(unique_cats)):
        cross_pairs.append((cat_sim[i, j], unique_cats[i], unique_cats[j]))
cross_pairs.sort(reverse=True)
for sim, a, b in cross_pairs[:10]:
    print(f"   {sim:.4f}  {a}  <->  {b}")

# Items that are semantically close but in DIFFERENT categories
print("\nHighest similarity pairs ACROSS categories (potential misclassifications or overlaps):")
cross_item_pairs = []
for i in range(len(descriptions)):
    for j in range(i + 1, len(descriptions)):
        if categories[i] != categories[j]:
            cross_item_pairs.append((cos_sim[i, j], i, j))
cross_item_pairs.sort(reverse=True)
for sim, i, j in cross_item_pairs[:15]:
    print(f"   {sim:.4f}  [{categories[i]}] {descriptions[i][:65]}")
    print(f"           [{categories[j]}] {descriptions[j][:65]}")
    print()

# Items that are in the SAME category but semantically distant
print("Least similar pairs WITHIN the same category (potential category sprawl):")
intra_pairs = []
for i in range(len(descriptions)):
    for j in range(i + 1, len(descriptions)):
        if categories[i] == categories[j]:
            intra_pairs.append((cos_sim[i, j], i, j))
intra_pairs.sort()
for sim, i, j in intra_pairs[:15]:
    print(f"   {sim:.4f}  [{categories[i]}] {descriptions[i][:65]}")
    print(f"           [{categories[j]}] {descriptions[j][:65]}")
    print()

# Sparse regions in semantic space
print("Sparse regions in semantic embedding space:")
dists_2d = squareform(pdist(coords_2d))
median_dist = np.median(dists_2d[np.triu_indices_from(dists_2d, k=1)])
densities = [(np.sum(dists_2d[i] < median_dist), i) for i in range(len(descriptions))]
densities.sort()
for density, idx in densities[:10]:
    print(f"   density={density:3d}  [{categories[idx]}]  {descriptions[idx][:80]}")

print(f"\n{'=' * 80}")
print("Done. Check eval_semantic_*.png files.")
