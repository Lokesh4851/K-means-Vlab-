"""K-Means Clustering Virtual Lab - Streamlit interface.

Run with:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.datasets import BLOB_KINDS, DATASET_TYPES, generate_dataset
from src.kmeans import (
    ASSIGNED, CONVERGED, INIT_METHODS, INITIALIZED, MAX_ITER_REACHED, NOT_INITIALIZED,
    PHASE_LABELS, UPDATED, KMeansLab, LabStateError, best_of_restarts, knee_point,
)
from src.visualization import data_limits, plot_clusters, plot_elbow, plot_wcss_history

st.set_page_config(page_title="K-Means Virtual Lab", layout="wide")

st.markdown(
    """
    <style>
      html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
          background-color: #ffffff !important;
          color: #111827 !important;
      }
      [data-testid="stSidebar"] {
          background-color: #ffffff !important;
          border-right: 1px solid #e5e7eb;
      }
      .block-container {
          padding-top: 1.5rem;
          padding-bottom: 2rem;
          background-color: #ffffff !important;
      }
      .phase-row {display:flex; gap:8px; flex-wrap:wrap; margin: 0.25rem 0 1rem 0;}
      .pill {padding: 4px 12px; border-radius: 4px; border: 1px solid #d1d5db;
             background: #ffffff; color: #4b5563; font-size: 0.82rem;}
      .pill.active {background:#111827; border-color:#111827; color:#ffffff; font-weight:600;}
      .pill.stopped {background:#b45309; border-color:#b45309; color:#ffffff; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)

PRETEST_QUESTIONS = [
    {
        "q": "1. What type of machine learning algorithm is K-Means clustering?",
        "options": [
            "Supervised learning (requires labeled target outputs)",
            "Unsupervised learning (groups unlabeled data by similarity)",
            "Reinforcement learning (learns via reward and penalty signals)",
            "Semi-supervised classification",
        ],
        "answer": "Unsupervised learning (groups unlabeled data by similarity)",
        "explanation": "K-Means is an unsupervised algorithm: it groups points using only their coordinates without pre-existing class labels.",
    },
    {
        "q": "2. In the K-Means algorithm, what does the parameter K represent?",
        "options": [
            "The number of features (dimensions) in the dataset",
            "The maximum number of iterations allowed",
            "The number of clusters (and centroids) to partition the data into",
            "The distance threshold for convergence",
        ],
        "answer": "The number of clusters (and centroids) to partition the data into",
        "explanation": "K specifies how many clusters and corresponding centroids the algorithm will maintain.",
    },
    {
        "q": "3. Which distance metric does standard K-Means use to assign each point to its nearest centroid?",
        "options": [
            "Euclidean distance",
            "Hamming distance",
            "Cosine similarity",
            "Jaccard distance",
        ],
        "answer": "Euclidean distance",
        "explanation": "Standard K-Means assigns each point to the centroid with the smallest Euclidean (straight-line) distance.",
    },
    {
        "q": "4. During the Update step, how is the new position of each centroid calculated?",
        "options": [
            "By picking a random data point inside the cluster",
            "By computing the arithmetic mean of all points assigned to that cluster",
            "By selecting the point farthest from the origin",
            "By taking the median of all points in the dataset",
        ],
        "answer": "By computing the arithmetic mean of all points assigned to that cluster",
        "explanation": "Moving each centroid to the arithmetic mean of its assigned points minimizes the sum of squared distances within that cluster.",
    },
    {
        "q": "5. What is the correct order of operations in K-Means?",
        "options": [
            "Update centroids -> Initialize -> Assign points",
            "Initialize centroids -> Repeat (Assign points -> Update centroids) until convergence",
            "Assign points -> Initialize centroids -> Compute Elbow",
            "Compute WCSS -> Delete outliers -> Initialize centroids",
        ],
        "answer": "Initialize centroids -> Repeat (Assign points -> Update centroids) until convergence",
        "explanation": "K-Means first places K initial centroids, then alternates between assigning points and updating centroids until movement stops.",
    },
]

POSTTEST_QUESTIONS = [
    {
        "q": "1. How does the Within-Cluster Sum of Squares (WCSS) behave after each Assign and Update step?",
        "options": [
            "It increases steadily until convergence",
            "It decreases (or stays equal) at every step and never increases",
            "It oscillates randomly between high and low values",
            "It remains constant across all iterations",
        ],
        "answer": "It decreases (or stays equal) at every step and never increases",
        "explanation": "Neither assigning points to their closest centroid nor moving centroids to the cluster mean can increase WCSS.",
    },
    {
        "q": "2. Why does K-Means++ generally produce better starting centroids than purely random initialization?",
        "options": [
            "It spreads initial centroids out by favoring points farther from already-chosen centroids",
            "It automatically chooses the optimal value of K",
            "It enables K-Means to separate concentric circles accurately",
            "It skips the Assignment step on the first iteration",
        ],
        "answer": "It spreads initial centroids out by favoring points farther from already-chosen centroids",
        "explanation": "K-Means++ selects each new centroid with probability proportional to its squared distance from the nearest existing centroid.",
    },
    {
        "q": "3. As K increases in the Elbow method, what happens to WCSS, and what is WCSS when K equals the number of distinct points?",
        "options": [
            "WCSS increases with K, and becomes infinite when K equals the number of points",
            "WCSS decreases as K increases, and becomes 0 when K equals the number of points",
            "WCSS peaks at the true number of clusters and then drops to 1",
            "WCSS does not depend on K",
        ],
        "answer": "WCSS decreases as K increases, and becomes 0 when K equals the number of points",
        "explanation": "More centroids allow points to sit closer to a center; when every point has its own centroid, all distances are 0 so WCSS = 0.",
    },
    {
        "q": "4. How does this virtual lab handle an empty cluster (a centroid that receives no points during Assignment)?",
        "options": [
            "It raises an unhandled division-by-zero error",
            "It permanently reduces K by 1",
            "It re-seeds the empty centroid at the data point farthest from its own current centroid",
            "It moves the empty centroid to the origin (0, 0)",
        ],
        "answer": "It re-seeds the empty centroid at the data point farthest from its own current centroid",
        "explanation": "An empty cluster has no mean, so the lab places its centroid at the point worst represented by the current clustering.",
    },
    {
        "q": "5. Why does K-Means fail to separate non-convex patterns like 'Two moons' or 'Concentric circles'?",
        "options": [
            "Because there are too few data points",
            "Because nearest-centroid Euclidean assignment forms linear (convex) boundaries suited for round groups",
            "Because K-Means++ only works in three dimensions",
            "Because the maximum iteration limit is too high",
        ],
        "answer": "Because nearest-centroid Euclidean assignment forms linear (convex) boundaries suited for round groups",
        "explanation": "Assigning points to the closest Euclidean centroid partitions the plane into convex cells, splitting interlocking curves or rings.",
    },
]


# ---------------------------------------------------------------- helpers
def run_action(name):
    """Button callback: perform one lab operation and remember any error message.

    Callbacks run BEFORE the page is redrawn, so the buttons and plots always
    show the state *after* the click.
    """
    lab = st.session_state.lab
    try:
        getattr(lab, name)()
        st.session_state.error = None
    except (LabStateError, ValueError) as exc:
        st.session_state.error = str(exc)


def phase_badges(phase):
    stages = [(INITIALIZED, "1 · Initialize"), (ASSIGNED, "2 · Assign"),
              (UPDATED, "3 · Update"), (CONVERGED, "4 · Converged")]
    html = '<div class="phase-row">'
    for key, label in stages:
        css = "pill"
        if phase == key:
            css += " active"
        elif key == CONVERGED and phase == MAX_ITER_REACHED:
            css += " stopped"
            label = "4 · Stopped (max iterations)"
        html += f'<span class="{css}">{label}</span>'
    return html + "</div>"


@st.cache_data(show_spinner=False)
def compute_elbow(kind, n_points, data_seed, n_true, init, algo_seed, k_max, restarts):
    """WCSS for K = 1..k_max (best of several restarts per K). Cached for speed."""
    X = generate_dataset(kind, n_points, data_seed, n_true)
    ks = list(range(1, k_max + 1))
    wcss = [best_of_restarts(X, k, init, algo_seed, restarts).wcss for k in ks]
    return ks, wcss


def render_quiz(prefix, questions):
    """Minimal multiple-choice quiz renderer used by Pre-Test and Post-Test."""
    for i, item in enumerate(questions):
        st.radio(
            item["q"],
            item["options"],
            index=None,
            key=f"{prefix}_q_{i}",
        )
    if st.button("Check answers", key=f"btn_{prefix}_submit"):
        st.session_state[f"{prefix}_submitted"] = True

    if st.session_state.get(f"{prefix}_submitted"):
        score = 0
        for i, item in enumerate(questions):
            choice = st.session_state.get(f"{prefix}_q_{i}")
            if choice == item["answer"]:
                score += 1
                st.success(f"Q{i + 1}: Correct. {item['explanation']}")
            elif choice is None:
                st.warning(f"Q{i + 1}: Not answered. Correct answer: **{item['answer']}**")
            else:
                st.error(f"Q{i + 1}: Incorrect. Correct answer: **{item['answer']}** — {item['explanation']}")
        st.markdown(f"**Score: {score} / {len(questions)}**")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("K-Means Lab")
    st.caption("An interactive virtual laboratory for K-Means clustering.")

    st.header("1 · Dataset")
    kind = st.selectbox("Dataset pattern", list(DATASET_TYPES),
                        help="The shape of the synthetic data. Try 'Two moons' to see where K-Means struggles.")
    st.caption(DATASET_TYPES[kind])
    n_points = st.slider("Number of points", 20, 1500, 300, 10, help="How many data points to generate.")
    data_seed = int(st.number_input("Data seed", 0, 999_999, 42, 1,
                                    help="Same seed + same settings = exactly the same data."))
    n_true = st.slider("Number of true groups", 2, 8, 3, disabled=kind not in BLOB_KINDS,
                       help="How many groups the generator creates (blob-type datasets only). "
                            "K-Means does not know this number - you choose K yourself.")

    st.header("2 · K-Means settings")
    k = st.slider("Number of clusters K", 1, 10, 3, help="How many centroids the algorithm will use.")
    init = st.radio("Centroid initialization", INIT_METHODS,
                    help="Random: K random data points. K-Means++: spreads the starting centroids out.")
    algo_seed = int(st.number_input("Algorithm seed", 0, 999_999, 0, 1,
                                    help="Controls the random choices made during initialization."))
    max_iter = st.slider("Maximum iterations", 1, 100, 25,
                         help="Safety limit: the run stops after this many iterations even if not converged.")

    # Rebuild data and experiment whenever a setting changes, so stale
    # centroids or labels can never be shown for new settings.
    config = dict(kind=kind, n_points=n_points, data_seed=data_seed,
                  n_true=n_true if kind in BLOB_KINDS else 0,
                  k=k, init=init, algo_seed=algo_seed, max_iter=max_iter)
    if st.session_state.get("config") != config:
        first_run = "config" not in st.session_state
        try:
            X_new = generate_dataset(kind, n_points, data_seed, max(n_true, 1))
            lab_new = KMeansLab(X_new, k, init, algo_seed, max_iter)
        except ValueError as exc:
            st.error(f"Invalid settings: {exc}")
            st.stop()
        st.session_state.update(config=config, X=X_new, lab=lab_new,
                                limits=data_limits(X_new), error=None)
        if not first_run:
            st.session_state.notice = "Settings changed: new data generated and the experiment was reset."

    lab = st.session_state.lab

    st.header("3 · Controls")
    st.button("① Initialize", key="btn_init", on_click=run_action, args=("initialize",),
              help="Place the K starting centroids (also restarts the run).")
    st.button("② Assign Points", key="btn_assign", on_click=run_action, args=("assign",),
              disabled=not lab.can_assign, help="Give every point the label of its nearest centroid.")
    st.button("③ Update Centroids", key="btn_update", on_click=run_action, args=("update",),
              disabled=not lab.can_update, help="Move each centroid to the mean of its points.")
    st.button("⏭ Next Iteration", key="btn_next", on_click=run_action, args=("next_iteration",),
              disabled=not (lab.can_assign or lab.can_update),
              help="Finish the current iteration: Assign then Update.")
    st.button("⏩ Run to Convergence", key="btn_run", on_click=run_action, args=("run_to_convergence",),
              disabled=not lab.can_run, help="Keep iterating until the centroids stop moving or the limit is hit.")
    st.button("↺ Reset", key="btn_reset", on_click=run_action, args=("reset",),
              help="Clear the experiment (the data stays).")

# ---------------------------------------------------------------- main page
X = st.session_state.X
limits = st.session_state.limits

st.title("K-Means Clustering Virtual Lab")
st.caption("Watch K-Means work one step at a time: initialize, assign, update, repeat until convergence.")

notice = st.session_state.pop("notice", None)
if notice:
    st.info(notice)
error = st.session_state.get("error")
if error:
    st.warning(error)

(
    tab_aim,
    tab_overview,
    tab_pretest,
    tab_lab,
    tab_elbow,
    tab_guide,
    tab_posttest,
    tab_conclusion,
) = st.tabs([
    "Aim",
    "Overview",
    "Pre-Test",
    "Lab",
    "Elbow method",
    "Algorithm guide",
    "Post-Test",
    "Conclusion",
])

# ---------------------------------------------------------------- aim
with tab_aim:
    st.subheader("Aim")
    st.markdown(
        "To study and visualize the step-by-step execution of the **K-Means clustering** algorithm, "
        "analyze the effect of cluster count (**K**) and centroid initialization (**Random** vs. **K-Means++**) "
        "on convergence and Within-Cluster Sum of Squares (**WCSS**), and evaluate the **Elbow method** "
        "across different 2-D dataset geometries."
    )
    st.subheader("Objectives")
    st.markdown(
        "- Understand how unsupervised clustering partitions unlabeled data into **K** distinct groups.\n"
        "- Execute K-Means one operation at a time: **Initialize**, **Assign Points** (nearest centroid via Euclidean distance), and **Update Centroids** (arithmetic mean of assigned points).\n"
        "- Observe how **WCSS** decreases monotonically at each step until the centroids converge or reach the iteration limit.\n"
        "- Use the **Elbow method** to identify a suitable number of clusters **K** and examine failure cases on non-spherical or non-convex data."
    )

# ---------------------------------------------------------------- overview
with tab_overview:
    st.subheader("What is this?")
    st.markdown(
        "**K-Means** groups data points into **K clusters** so that points in the same cluster are "
        "close to each other. This lab lets you generate data, choose K, and then advance the "
        "algorithm **one operation at a time** to see exactly what it does."
    )
    st.subheader("How to use it")
    st.markdown(
        """
        1. Choose a **dataset** and **K** in the sidebar.
        2. Open the **Lab** tab and press **① Initialize** - the starting centroids (large **X** markers) appear.
        3. Press **② Assign Points**: points take the colour of their nearest centroid.
        4. Press **③ Update Centroids**: every centroid jumps to the mean of its points. Hollow circles and arrows show where it came from.
        5. Repeat 3-4 (or use **⏭ Next Iteration**) until the lab reports **Converged**, or press **⏩ Run to Convergence**.
        6. Open **Elbow method** to compare different values of K.
        """
    )
    st.subheader("Reading the plot")
    st.markdown(
        "- **Small dots** = data points (grey before assignment, cluster colour after).\n"
        "- **Large X** = current centroid, numbered by cluster.\n"
        "- **Hollow circles / arrows** = earlier centroid positions and movement paths.\n"
        "- **WCSS** = sum of squared distances from points to their centroid; smaller means tighter clusters."
    )

# ---------------------------------------------------------------- pretest
with tab_pretest:
    st.subheader("Pre-Test")
    st.caption("Answer these foundational questions before running the experiment.")
    render_quiz("pretest", PRETEST_QUESTIONS)

# ---------------------------------------------------------------- lab
with tab_lab:
    st.markdown(phase_badges(lab.phase), unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Iteration", lab.iteration)
    m2.metric("K", lab.k)
    m3.metric("Points", len(X))
    m4.metric("WCSS", "-" if lab.wcss is None else f"{lab.wcss:,.2f}")
    m5.metric("Last centroid shift", "-" if lab.last_shift is None else f"{lab.last_shift:.4f}")
    st.caption(f"State: **{PHASE_LABELS[lab.phase]}**")

    left, right = st.columns([3, 2])
    with left:
        c1, c2 = st.columns(2)
        show_prev = c1.checkbox("Show previous centroid positions", value=True,
                                help="Hollow circles mark where each centroid used to be.")
        show_paths = c2.checkbox("Show movement paths", value=True,
                                 help="Arrows from each old centroid position to the next one.")
        title = ("Original data (not clustered yet)" if lab.phase == NOT_INITIALIZED
                 else f"Iteration {lab.iteration} - {PHASE_LABELS[lab.phase]}")
        st.pyplot(plot_clusters(X, lab.centroids, lab.labels, lab.centroid_history,
                                show_prev, show_paths, title, limits))
    with right:
        st.markdown("#### What just happened")
        st.info(lab.last_message)

        st.markdown("#### Clusters and centroids")
        if lab.centroids is None:
            st.caption("No centroids yet. Press **① Initialize**.")
        else:
            table = {"Cluster": list(range(lab.k))}
            sizes = lab.cluster_sizes
            if sizes is not None:
                table["Points"] = sizes
                table["Share"] = [f"{100 * s / len(X):.1f} %" for s in sizes]
            table["Centroid x"] = lab.centroids[:, 0].round(3)
            table["Centroid y"] = lab.centroids[:, 1].round(3)
            st.dataframe(pd.DataFrame(table), hide_index=True)
            if lab.phase == UPDATED or lab.phase == CONVERGED:
                st.caption("The WCSS above uses the **new** centroids with the labels from the last Assign step.")

    if lab.wcss_history:
        st.markdown("#### WCSS history")
        st.pyplot(plot_wcss_history(lab.wcss_history))
    with st.expander("Step-by-step log"):
        for entry in lab.log:
            st.markdown(f"**{entry['step'].capitalize()}** (iteration {entry['iteration']}): {entry['message']}")
    with st.expander("Original data before clustering"):
        st.pyplot(plot_clusters(X, title=f"{kind}: {len(X)} points", limits=limits))
        st.caption(f"Mean of the data: ({X[:, 0].mean():.2f}, {X[:, 1].mean():.2f}).")

# ---------------------------------------------------------------- elbow
with tab_elbow:
    st.subheader("Elbow method")
    st.markdown(
        "K-Means is run for K = 1, 2, 3, ... and the final **WCSS** is plotted against K. "
        "**WCSS always tends to fall as K grows**, because more centroids can sit closer to the "
        "points (with K = number of points, WCSS is 0). So we look for an **elbow**: the K after "
        "which extra clusters give only small improvements."
    )
    e1, e2 = st.columns(2)
    k_max = e1.slider("Largest K to test", 3, 12, 8)
    restarts = e2.slider("Random restarts per K", 1, 10, 5,
                         help="Each K is run this many times with different seeds; the lowest WCSS is kept.")
    if st.button("Run elbow analysis", key="btn_elbow"):
        st.session_state.elbow_request = dict(
            data_key=(kind, n_points, data_seed, n_true if kind in BLOB_KINDS else 0, init, algo_seed),
            k_max=k_max, restarts=restarts)

    request = st.session_state.get("elbow_request")
    current_key = (kind, n_points, data_seed, n_true if kind in BLOB_KINDS else 0, init, algo_seed)
    if request is None:
        st.caption("Press **Run elbow analysis** to compute the curve for the current dataset.")
    elif request["data_key"] != current_key:
        st.info("The dataset or initialization settings changed. Press **Run elbow analysis** again.")
    else:
        with st.spinner("Running K-Means for each K..."):
            ks, wcss_values = compute_elbow(kind, n_points, data_seed, max(n_true, 1), init,
                                            algo_seed, request["k_max"], request["restarts"])
        suggested = knee_point(ks, wcss_values)
        st.pyplot(plot_elbow(ks, wcss_values, suggested))
        drops = [None] + [100 * (a - b) / a if a > 0 else 0.0 for a, b in zip(wcss_values, wcss_values[1:])]
        st.dataframe(pd.DataFrame({
            "K": ks, "WCSS": [round(w, 2) for w in wcss_values],
            "Drop from previous K": ["-" if d is None else f"{d:.1f} %" for d in drops]}),
            hide_index=True)
        if suggested is None:
            st.warning("No clear elbow: the curve bends only gently. This is typical when the data "
                       "has no compact, separated groups.")
        else:
            st.success(f"A simple 'largest bend' heuristic points to **K = {suggested}**.")
        st.caption("Treat this as a hint, not an answer. The elbow can be subtle or absent, and the "
                   "best K also depends on what you want to use the clusters for.")
        if kind in BLOB_KINDS:
            st.caption(f"For reference, this dataset was generated from {n_true} groups. "
                       "Real data does not come with that label.")

# ---------------------------------------------------------------- guide
with tab_guide:
    st.subheader("The algorithm in six steps")
    st.markdown("**1. Initialization.** Choose K starting centroids - K random data points, or K-Means++, "
                "which picks the next centroid with probability proportional to its squared distance from "
                "the nearest centroid already chosen (this spreads the centroids out).")
    st.markdown("**2. Distance.** Euclidean distance between a point x and a centroid c:")
    st.latex(r"d(x,c)=\sqrt{(x_1-c_1)^2+(x_2-c_2)^2}")
    st.markdown("**3. Assignment.** Each point joins the cluster of its nearest centroid:")
    st.latex(r"\text{label}(x)=\arg\min_j\, d(x,c_j)")
    st.markdown("**4. Update.** Each centroid moves to the arithmetic mean of the points in its cluster S_j:")
    st.latex(r"c_j=\frac{1}{|S_j|}\sum_{x\in S_j}x")
    st.markdown("**5. Empty clusters.** If a centroid gets no points it has no mean. This lab re-seeds it at the "
                "data point farthest from its own centroid, and tells you when it happens.")
    st.markdown("**6. Convergence.** Repeat steps 3-4 until the largest centroid movement is at most the "
                "tolerance (1e-6) or the maximum number of iterations is reached.")
    st.subheader("WCSS")
    st.latex(r"\text{WCSS}=\sum_{j=1}^{K}\sum_{x\in S_j}\lVert x-c_j\rVert^2")
    st.markdown("Neither the Assign step nor the Update step can increase WCSS, so it falls (or stays "
                "equal) at every step. K-Means therefore always settles, but possibly in a **local** "
                "optimum that depends on the starting centroids - try different algorithm seeds.")
    st.subheader("Limitations to explore")
    st.markdown("- Prefers round, similar-sized clusters (see *Stretched blobs*, *Uneven cluster sizes*).\n"
                "- Cannot separate non-convex shapes (see *Two moons*, *Concentric circles*).\n"
                "- Always returns K clusters, even for data with no clusters (see *Uniform random*).\n"
                "- K must be chosen by you; the elbow plot is only a heuristic.")

# ---------------------------------------------------------------- posttest
with tab_posttest:
    st.subheader("Post-Test")
    st.caption("Test your understanding after completing the virtual lab experiments.")
    render_quiz("posttest", POSTTEST_QUESTIONS)

# ---------------------------------------------------------------- conclusion
with tab_conclusion:
    st.subheader("Conclusion")
    st.markdown(
        "Through this virtual lab, we observed the complete working of the **K-Means clustering** algorithm:\n\n"
        "- **Iterative refinement**: K-Means alternates between assigning each point to its nearest Euclidean centroid "
        "and updating each centroid to the arithmetic mean of its assigned points.\n"
        "- **Monotonic WCSS reduction**: Every Assign and Update step reduces or maintains the Within-Cluster Sum of Squares (**WCSS**), "
        "ensuring the algorithm converges in a finite number of steps—though the final result is a local optimum sensitive to initial centroid placement.\n"
        "- **Initialization & choosing K**: **K-Means++** spreads out initial centroids for more reliable convergence, while the **Elbow method** "
        "identifies the value of **K** beyond which adding more clusters gives diminishing reductions in WCSS.\n"
        "- **Geometric assumptions**: K-Means excels on compact, well-separated spherical clusters, but struggles with elongated shapes, "
        "unequal cluster sizes, non-convex patterns (*Two moons*, *Concentric circles*), and unclustered uniform noise."
    )
