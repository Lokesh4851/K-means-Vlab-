# K-Means Clustering Virtual Lab

An interactive Streamlit app that teaches K-Means clustering step by step. K-Means is implemented
**from scratch with NumPy** (no scikit-learn). You can generate data, choose K, and press
Initialize / Assign / Update one at a time to watch the centroids move to the **arithmetic mean**
of their points.

## Project structure

```
kmeans-vlab/
├── app.py                  Streamlit interface (sidebar, tabs, dashboard)
├── src/
│   ├── __init__.py
│   ├── kmeans.py           K-Means maths + step-by-step KMeansLab state machine
│   ├── datasets.py         Synthetic 2-D dataset generators
│   └── visualization.py    Matplotlib figures (clusters, centroid paths, WCSS, elbow)
├── tests/
│   ├── test_kmeans.py      Unit tests for the algorithm and datasets
│   └── test_app_smoke.py   Runs app.py headlessly and clicks the buttons
├── docs/report_and_viva.md Report outline and viva questions
├── conftest.py             Lets pytest import `src` (leave it empty)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup on Windows with VS Code

Install Python 3.13.x from python.org (tick **Add python.exe to PATH**) and VS Code with the
**Python** extension. Open the project folder in VS Code (File > Open Folder), then open a terminal
(Terminal > New Terminal, PowerShell) and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The browser opens at http://localhost:8501. If PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and activate again. In VS Code, press
Ctrl+Shift+P > **Python: Select Interpreter** and pick the `.venv` one. Stop the app with Ctrl+C.

Run the tests: `python -m pytest -q`

## Operating the lab

1. **Sidebar > Dataset**: choose a pattern, number of points, data seed (same seed = same data).
2. **Sidebar > K-Means settings**: choose K, initialization (Random or K-Means++), algorithm seed, max iterations.
3. **Lab tab**: press **① Initialize** (centroids appear as large X markers).
4. **② Assign Points**: points take the colour of the nearest centroid.
5. **③ Update Centroids**: each centroid moves to the mean of its points. Hollow circles and arrows show the old positions.
6. **⏭ Next Iteration** does Assign + Update; **⏩ Run to Convergence** repeats until the centroids stop moving or the max-iteration limit is hit; **↺ Reset** clears the run.
7. **Elbow method tab**: press *Run elbow analysis* to plot WCSS against K = 1..K_max.

Buttons that are not valid in the current state are disabled. Changing any sidebar setting
regenerates the data and resets the experiment, so stale centroids are never shown.

## Feature map

| Requirement | Where |
|---|---|
| Dataset generation, seed, raw-data view | `src/datasets.py`, sidebar, "Original data" expander |
| Initialize / Assign / Update / Next / Run / Reset | `KMeansLab` methods, sidebar buttons |
| Step-by-step plot, centroid paths | `plot_clusters` |
| Distance, assignment, mean update, empty clusters, convergence, max iterations, WCSS | `src/kmeans.py` |
| Elbow method | `best_of_restarts`, `knee_point`, Elbow tab |
| Dashboard (iteration, K, sizes, WCSS, centroids, explanation) | Lab tab |

## Design decisions

- **Iteration** = one Assign followed by one Update. The counter increases on Assign.
- **Convergence**: the largest centroid movement in an Update step is at most 1e-6, or the max-iteration limit is reached.
- **Empty cluster**: the centroid is re-seeded at the data point farthest from its own old centroid, and the lab says so.
- **WCSS** is always computed from the current labels and current centroids. After an Update step it uses the new centroids with the labels from the last Assign step.
- **Elbow** uses the best of several random restarts per K, because K-Means only finds a local optimum. The "largest bend" marker is a heuristic, not a verdict.

## Verification status

The 45 automated tests passed on Linux with Python 3.12.3, NumPy 2.4.4, Matplotlib 3.10.8 and
Streamlit 1.66.1. They have **not** been run on Windows or Python 3.13.5, and the interface has not been
inspected in a real browser. Run the manual checklist below on your own machine.

## Manual test checklist

- [ ] App starts with `streamlit run app.py`; the Lab tab shows grey, unclustered points.
- [ ] Assign and Update are disabled before Initialize; Update is disabled until Assign is pressed.
- [ ] Initialize shows K large X markers; the same seed gives the same markers each time.
- [ ] Assign colours every point; cluster sizes add up to the number of points.
- [ ] Update moves each centroid to the visible centre of its colour group; arrows and hollow circles appear.
- [ ] WCSS never rises in the WCSS history chart.
- [ ] Run to Convergence ends in "Converged"; pressing Next Iteration afterwards shows a message, not a crash.
- [ ] Max iterations = 1 with Run to Convergence stops with "Stopped: max iterations".
- [ ] Reset clears centroids, colours, iteration (back to 0) and the log; the data stays.
- [ ] Changing K, the dataset, a seed or the point count resets the experiment and shows a notice.
- [ ] K = 1 and K = 10 both work; "Uniform random" and "Two moons" run without errors.
- [ ] Elbow tab: the curve falls as K rises; changing a sidebar setting asks you to run it again.

## GitHub setup

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git init
git add .
git commit -m "K-Means Virtual Lab: initial version"
git branch -M main
```

Create an empty repository on github.com (no README, no .gitignore), then:

```powershell
git remote add origin https://github.com/<your-username>/kmeans-vlab.git
git push -u origin main
```

Later changes: `git add .`, `git commit -m "describe the change"`, `git push`.
Do not commit `.venv/` (already in `.gitignore`).

## Troubleshooting

- `ModuleNotFoundError: src`: run commands from the project root folder (where `app.py` is).
- `streamlit` not recognised: activate the virtual environment first.
- Port busy: `streamlit run app.py --server.port 8502`.
