#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-Queens via a Genetic Algorithm (PyGAD-only, version-robust for 2.20.0+)
-------------------------------------------------------------------------

THEORY SNAPSHOT (concise, practical):
• Representation:
  A candidate is a length-N vector q with q[i] ∈ {0..N-1} = column of the queen in row i.
  This enforces exactly one queen per row. Column/diagonal conflicts are handled by fitness.

• Conflicts (O(N)):
  Two queens (i, q[i]) and (j, q[j]) conflict if:
     - same column:   q[i] == q[j]
     - same diagonal: |q[i] - q[j]| == |i - j|
  Count conflicts in O(N) via frequency tables for columns c, and diagonal indices:
     d1 = i - q[i] (main), d2 = i + q[i] (anti). Any line with k queens adds C(k,2).

• Fitness (maximize):
  Let M = N(N-1)/2 be the number of unordered row pairs. We set:
     fitness(solution) = M - (#conflicts)
  A valid board has 0 conflicts ⇒ fitness = M. Early stop when best fitness hits M.

• GA operators:
  Tournament selection, single-point crossover, random-reset mutation over gene_space={0..N-1}.
  Not strictly a permutation GA; duplicates are allowed but penalized (column conflicts),
  which naturally steers toward permutation-like solutions.

• Complexity:
  ~ O(G * P * N) fitness work for G generations and population P. GA is heuristic; for moderate N
  it often finds valid boards quickly.

USAGE:
  - Edit the USER SETTINGS block.
  - Run: python main.py
  - You’ll get: best/mean fitness curve, best-conflicts curve, final board plot.
  - Plots are also saved to docs/ next to this file.
"""

# ========================== USER SETTINGS (no argparse) ========================
N = 8                 # board size (try 20, 50, 100…)
POP_SIZE = 200        # population size
GENERATIONS = 600     # maximum generations
TOURNAMENT_K = 4      # tournament size
MUTATION_RATE = 0.10  # per-gene mutation probability in [0,1]
ELITISM = 6           # elites to keep (or parents on older versions)
RANDOM_SEED = 42      # set None for stochastic runs
# ==============================================================================

from typing import List, Dict
import os
import inspect
import numpy as np
import matplotlib.pyplot as plt
import pygad

# ---------- paths: ensure docs/ exists beside this script (not just CWD) ------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

# ---------------------------- Fitness utilities -------------------------------

def _pairs(k: int) -> int:
    return (k * (k - 1)) // 2

def _max_pairs(n: int) -> int:
    return n * (n - 1) // 2

def count_conflicts(solution: List[int]) -> int:
    """O(N) conflict counter via column & diagonal frequency tables."""
    cols: Dict[int, int] = {}
    d1: Dict[int, int] = {}  # i - q[i]
    d2: Dict[int, int] = {}  # i + q[i]
    for r, c in enumerate(solution):
        cols[c] = cols.get(c, 0) + 1
        d1[r - c] = d1.get(r - c, 0) + 1
        d2[r + c] = d2.get(r + c, 0) + 1
    conflicts = 0
    for v in cols.values(): conflicts += _pairs(v)
    for v in d1.values():   conflicts += _pairs(v)
    for v in d2.values():   conflicts += _pairs(v)
    return conflicts

# ----------------------------- Run history ------------------------------------

best_fitness_hist: List[float] = []
mean_fitness_hist: List[float] = []
best_conflicts_hist: List[int] = []

def _update_history_from_population(population_matrix: np.ndarray, n: int):
    """Record best/mean fitness & best conflicts for plotting."""
    M = _max_pairs(n)
    fits = []
    for row in population_matrix:
        sol = [int(x) for x in row]
        fits.append(M - count_conflicts(sol))
    fits = np.array(fits, dtype=float)
    best_f = float(np.max(fits))
    mean_f = float(np.mean(fits))
    best_fitness_hist.append(best_f)
    mean_fitness_hist.append(mean_f)
    best_conflicts_hist.append(int(M - best_f))
    return best_f

# ------------------------------ GA (PyGAD) ------------------------------------

def solve_with_pygad(n: int,
                     pop: int,
                     gens: int,
                     k_tourn: int,
                     mut_prob: float,
                     elitism: int,
                     rng_seed):
    """
    Build & run a PyGAD GA for N-Queens.
    Compatible with PyGAD 2.20.0+ by introspecting GA.__init__ and
    providing required/available kwargs. Fitness function ALWAYS
    uses the 3-parameter signature required by 2.20.0.
    """
    M = _max_pairs(n)

    # 3-parameter signature (compatible with PyGAD 2.20.0+)
    def fitness_func(ga_instance, solution, solution_idx):
        sol = [int(x) for x in solution]
        return M - count_conflicts(sol)

    def on_generation(ga_instance):
        best_f = _update_history_from_population(ga_instance.population, n)
        # Early-stop in a version-tolerant way:
        try:
            ga_instance.stop_generation = True if best_f >= M else False
        except Exception:
            pass
        if best_f >= M:
            return "stop"

    # Introspect GA.__init__ to adapt to version differences.
    sig = inspect.signature(pygad.GA.__init__)
    params = set(sig.parameters.keys())

    init_kwargs = {
        "num_generations": gens,
        "sol_per_pop": pop,
        "num_genes": n,
        "fitness_func": fitness_func,
        "gene_space": list(range(n)),
        "parent_selection_type": "tournament",
        "crossover_type": "single_point",
        "mutation_type": "random",
        "on_generation": on_generation,
    }

    if "gene_type" in params:
        init_kwargs["gene_type"] = int
    if "K_tournament" in params:
        init_kwargs["K_tournament"] = k_tourn
    if "num_parents_mating" in params:
        init_kwargs["num_parents_mating"] = max(2, pop // 2)
    if "mutation_probability" in params:
        init_kwargs["mutation_probability"] = mut_prob
    elif "mutation_percent_genes" in params:
        init_kwargs["mutation_percent_genes"] = int(round(mut_prob * 100))
    if "keep_elitism" in params:
        init_kwargs["keep_elitism"] = elitism
    elif "keep_parents" in params:
        init_kwargs["keep_parents"] = elitism
    if "random_seed" in params:
        init_kwargs["random_seed"] = rng_seed

    ga = pygad.GA(**init_kwargs)
    ga.run()

    solution, solution_fitness, _ = ga.best_solution()
    best_solution = [int(x) for x in solution]
    best_gen = getattr(ga, "best_solution_generation", -1)
    if best_gen == -1:
        best_gen = len(best_fitness_hist) - 1

    return {
        "solution": best_solution,
        "fitness": float(solution_fitness),
        "generation": int(best_gen)
    }

# ------------------------------- Plot helpers ---------------------------------

def plot_curves(n: int):
    gens = np.arange(len(best_fitness_hist))

    # 1) Fitness curves
    fig1 = plt.figure()
    plt.plot(gens, best_fitness_hist, label="Best fitness")
    plt.plot(gens, mean_fitness_hist, label="Mean fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (higher is better)")
    plt.title(f"N-Queens (N={n}) — Fitness over Generations")
    plt.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(DOCS_DIR, "fitness.png"), dpi=200, bbox_inches="tight")
    plt.show()

    # 2) Best conflicts curve
    fig2 = plt.figure()
    plt.plot(gens, best_conflicts_hist)
    plt.xlabel("Generation")
    plt.ylabel("# Conflicts (lower is better)")
    plt.title(f"N-Queens (N={n}) — Best Conflicts over Generations")
    plt.tight_layout()
    fig2.savefig(os.path.join(DOCS_DIR, "conflicts.png"), dpi=200, bbox_inches="tight")
    plt.show()

def plot_board(sol: List[int],
               save_path=os.path.join(DOCS_DIR, "board.png"),
               origin: str = "top",      # "top" (like your current plot) or "bottom" (chess-style)
               labels: str = "index"     # "index" -> 0..N-1, "chess" -> a..h / 8..1
               ):
    """
    Pretty board renderer with alternating squares and a ♛ queen glyph.
    - origin="top": row 0 is drawn at the top (matches your previous plot)
      origin="bottom": row 0 is drawn at the bottom (chess-style)
    - labels="index": ticks show 0..N-1
      labels="chess": files=a.. and ranks=N..1 (or 1..N if origin='bottom')
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(sol)
    # Colors similar to common chess themes
    light = "#f0d9b5"
    dark  = "#b58863"
    border = "#333333"

    # Map row index -> y coordinate depending on origin
    def y_of_row(r):
        return (n - 1 - r) if origin == "top" else r

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal", adjustable="box")

    # Draw squares
    for r in range(n):
        for c in range(n):
            y = y_of_row(r)
            color = light if (r + c) % 2 == 0 else dark
            ax.add_patch(Rectangle((c, y), 1, 1, facecolor=color, edgecolor="none"))

    # Board border
    ax.add_patch(Rectangle((0, 0), n, n, fill=False, linewidth=2, edgecolor=border))

    # Place queens as a text glyph centered in each square
    # (fallback to 'Q' if the font lacks '♛')
    fontsize = 0.8 * (600 / n)  # scale roughly with board size
    for r, c in enumerate(sol):
        y = y_of_row(r)
        try:
            ax.text(c + 0.5, y + 0.5, "♛", ha="center", va="center",
                    fontsize=fontsize, color="#222222")
        except Exception:
            ax.text(c + 0.5, y + 0.5, "Q", ha="center", va="center",
                    fontsize=fontsize, color="#222222", fontweight="bold")

    # Ticks & labels
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_yticks([i + 0.5 for i in range(n)])
    if labels == "chess":
        files = [chr(ord('a') + i) for i in range(n)]
        if origin == "top":
            ranks = [str(n - i) for i in range(n)]    # top row is rank N
        else:
            ranks = [str(i + 1) for i in range(n)]    # bottom row is rank 1
        ax.set_xticklabels(files)
        ax.set_yticklabels(ranks)
    else:
        ax.set_xticklabels([str(i) for i in range(n)])
        ax.set_yticklabels([str(i) for i in (range(n-1, -1, -1) if origin == "top" else range(n))])

    # Remove spines; keep a clean board look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax.set_title(f"N-Queens (N={n}) — Final Placement (conflicts={count_conflicts(sol)})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()

# ---------------------------------- Main --------------------------------------

if __name__ == "__main__":
    result = solve_with_pygad(
        n=N,
        pop=POP_SIZE,
        gens=GENERATIONS,
        k_tourn=TOURNAMENT_K,
        mut_prob=MUTATION_RATE,
        elitism=ELITISM,
        rng_seed=RANDOM_SEED
    )
    plot_curves(N)
    plot_board(result["solution"])

    M = _max_pairs(N)
    print("=== GA summary ===")
    print(f"N={N}")
    print(f"Best fitness: {result['fitness']} / {M}")
    print(f"Best conflicts: {M - result['fitness']}")
    print(f"Found at generation: {result['generation']}")
    print(f"Saved plots to: {DOCS_DIR}/fitness.png, {DOCS_DIR}/conflicts.png, {DOCS_DIR}/board.png")
