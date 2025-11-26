import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. Single-step physiology
# =========================

def step_state(E, I, N, A, M,
               severity,
               dt,
               u_vagus=0.0,
               u_symp=0.0,
               u_nerve=0.0,
               u_immune_tx=0.0,
               u_perf=0.0,
               u_metab=0.0,
               u_nutrition=0.0):
    """
    One discrete-time update step (e.g., 1 hour).

    States (baseline ~1.0):
      E = Energy / metabolic state
      I = Immune / inflammation
      N = Nociceptive / threat signaling
      A = Autonomic tone
      M = Perfusion / microcirculation / mitochondrial O2

    severity in [0,1]; treatments u_* in [-1, 1].
    """
    # ---------- Baseline recovery/decay rates ----------
    kE_rec_base   = 0.15 * (1 - 0.5*severity)
    kM_rec_base   = 0.12 * (1 - 0.5*severity)
    kI_decay_base = 0.25 * (1 - 0.3*severity)
    kN_decay_base = 0.30 * (1 - 0.3*severity)
    kA_decay_base = 0.20 * (1 - 0.3*severity)

    kE_rec   = max(kE_rec_base,   0.02)
    kM_rec   = max(kM_rec_base,   0.02)
    kI_decay = max(kI_decay_base, 0.05)
    kN_decay = max(kN_decay_base, 0.05)
    kA_decay = max(kA_decay_base, 0.05)

    # ---------- Coupling coefficients ----------
    # Inflammation ↔ energy/mito, microcirculation
    k_IE = 0.10
    k_IM = 0.10
    k_ME = 0.08
    k_EI = 0.06
    k_MI = 0.06

    # Inflammation <-> autonomic/nociception
    k_IA = 0.08
    k_IN = 0.08
    k_AI = 0.05
    k_NI = 0.05

    # Energy & perfusion -> autonomic/nociception
    k_EA = 0.06
    k_EN = 0.06
    k_MN = 0.06

    # ---------- Treatment coefficients ----------
    # Vagal (cholinergic anti-inflammatory pathway)
    k_vagus_I = 0.15
    k_vagus_A = 0.12
    k_vagus_N = 0.10

    # Sympathetic drive / block
    k_symp_A  = 0.10
    k_symp_I  = 0.05

    # Nerve block / analgesia
    k_nerve   = 0.15

    # Immune modulation
    k_immuno  = 0.10

    # Perfusion support
    k_perf_M  = 0.15

    # Metabolic / mitochondrial support
    k_metab_E = 0.12
    k_metab_M = 0.10

    # Nutrition strategy
    k_nutrition_E = 0.05
    k_nutrition_I = 0.03

    pos = lambda x: max(x, 0.0)

    # ---------- Baseline tendency toward 1.0 ----------
    E_pred = E + dt * kE_rec   * (1.0 - E)
    M_pred = M + dt * kM_rec   * (1.0 - M)
    I_pred = I - dt * kI_decay * (I - 1.0)
    N_pred = N - dt * kN_decay * (N - 1.0)
    A_pred = A - dt * kA_decay * (A - 1.0)

    # ---------- Physiologic couplings ----------
    # I -> E, M (inflammatory damage)
    E_pred -= dt * k_IE * pos(I - 1.0)
    M_pred -= dt * k_IM * pos(I - 1.0)

    # M -> E (hypoperfusion → energy deficit)
    E_pred -= dt * k_ME * pos(1.0 - M)

    # E/M -> I (metabolic inflammation)
    I_pred += dt * (k_EI * pos(1.0 - E) +
                    k_MI * pos(1.0 - M))

    # I -> A, N (inflammatory reflex, sickness)
    A_pred += dt * k_IA * pos(I - 1.0)
    N_pred += dt * k_IN * pos(I - 1.0)

    # A, N -> I (autonomic & nociceptive drive on inflammation)
    I_pred += dt * (k_AI * pos(A - 1.0) +
                    k_NI * pos(N - 1.0))

    # E deficit -> A, N (energy as threat)
    A_pred += dt * k_EA * pos(1.0 - E)
    N_pred += dt * k_EN * pos(1.0 - E)

    # M deficit -> N (ischemic threat)
    N_pred += dt * k_MN * pos(1.0 - M)

    # ---------- Treatment effects ----------
    # 1) Vagal
    I_pred -= dt * k_vagus_I * u_vagus * (I - 1.0)
    A_pred -= dt * k_vagus_A * u_vagus * (A - 1.0)
    N_pred -= dt * k_vagus_N * u_vagus * (N - 1.0)

    # 2) Sympathetic
    A_pred += dt * k_symp_A * u_symp
    if u_symp > 0:
        I_pred += dt * k_symp_I * u_symp

    # 3) Nerve block
    N_pred -= dt * k_nerve * u_nerve * (N - 1.0)

    # 4) Immune modulation
    I_pred += dt * k_immuno * u_immune_tx

    # 5) Perfusion
    M_pred += dt * k_perf_M * u_perf * (1.0 - M)

    # 6) Metabolic / mito
    E_pred += dt * k_metab_E * u_metab * (1.0 - E)
    M_pred += dt * k_metab_M * u_metab * (1.0 - M)

    # 7) Nutrition
    E_pred += dt * k_nutrition_E * u_nutrition
    I_pred += dt * k_nutrition_I * u_nutrition

    return E_pred, I_pred, N_pred, A_pred, M_pred


# ==============================
# 2. Multi-step patient simulation
# ==============================

def simulate_patient_discrete(
    T=72, dt=1.0, severity=0.5,
    u_vagus=0.0,    t_vagus_on=0.0,     t_vagus_off=72.0,
    u_symp=0.0,     t_symp_on=0.0,      t_symp_off=72.0,
    u_nerve=0.0,    t_nerve_on=0.0,     t_nerve_off=72.0,
    u_immune_tx=0.0,t_immune_on=0.0,    t_immune_off=72.0,
    u_perf=0.0,     t_perf_on=0.0,      t_perf_off=72.0,
    u_metab=0.0,    t_metab_on=0.0,     t_metab_off=72.0,
    u_nutrition=0.0,t_nutrition_on=0.0, t_nutrition_off=72.0
):
    steps = int(T/dt) + 1
    t = np.linspace(0, T, steps)

    E = np.zeros(steps); I = np.zeros(steps); N = np.zeros(steps)
    A = np.zeros(steps); M = np.zeros(steps)

    # Initial injury at t=0
    E[0] = 1.0 - 0.4 * severity
    M[0] = 1.0 - 0.5 * severity
    I[0] = 1.0 + 2.0 * severity
    N[0] = 1.0 + 1.5 * severity
    A[0] = 1.0 + 1.5 * severity

    for k in range(steps-1):
        tk = t[k]

        uv     = u_vagus     if (tk >= t_vagus_on     and tk <= t_vagus_off)     else 0.0
        us     = u_symp      if (tk >= t_symp_on      and tk <= t_symp_off)      else 0.0
        un     = u_nerve     if (tk >= t_nerve_on     and tk <= t_nerve_off)     else 0.0
        ui     = u_immune_tx if (tk >= t_immune_on    and tk <= t_immune_off)    else 0.0
        up     = u_perf      if (tk >= t_perf_on      and tk <= t_perf_off)      else 0.0
        um     = u_metab     if (tk >= t_metab_on     and tk <= t_metab_off)     else 0.0
        unutr  = u_nutrition if (tk >= t_nutrition_on and tk <= t_nutrition_off) else 0.0

        E[k+1], I[k+1], N[k+1], A[k+1], M[k+1] = step_state(
            E[k], I[k], N[k], A[k], M[k],
            severity=severity,
            dt=dt,
            u_vagus=uv,
            u_symp=us,
            u_nerve=un,
            u_immune_tx=ui,
            u_perf=up,
            u_metab=um,
            u_nutrition=unutr
        )

    # NIMTI
    wE, wI, wN, wA, wM = 2.0, 1.5, 1.5, 1.0, 2.0
    devE = np.abs(E - 1.0)
    devI = np.abs(I - 1.0)
    devN = np.abs(N - 1.0)
    devA = np.abs(A - 1.0)
    devM = np.abs(M - 1.0)
    NIMTI = wE*devE + wI*devI + wN*devN + wA*devA + wM*devM

    return t, E, I, N, A, M, NIMTI


# ==========================
# 3. Plotting with phases & Death@X
# ==========================

def plot_discrete_patient(
    severity=0.5,
    u_vagus=0.0,    t_vagus_on=0.0,     t_vagus_off=72.0,
    u_symp=0.0,     t_symp_on=0.0,      t_symp_off=72.0,
    u_nerve=0.0,    t_nerve_on=0.0,     t_nerve_off=72.0,
    u_immune_tx=0.0,t_immune_on=0.0,    t_immune_off=72.0,
    u_perf=0.0,     t_perf_on=0.0,      t_perf_off=72.0,
    u_metab=0.0,    t_metab_on=0.0,     t_metab_off=72.0,
    u_nutrition=0.0,t_nutrition_on=0.0, t_nutrition_off=72.0
):
    t, E, I, N, A, M, NIMTI = simulate_patient_discrete(
        T=72, dt=1.0, severity=severity,
        u_vagus=u_vagus,        t_vagus_on=t_vagus_on,     t_vagus_off=t_vagus_off,
        u_symp=u_symp,          t_symp_on=t_symp_on,       t_symp_off=t_symp_off,
        u_nerve=u_nerve,        t_nerve_on=t_nerve_on,     t_nerve_off=t_nerve_off,
        u_immune_tx=u_immune_tx,t_immune_on=t_immune_on,   t_immune_off=t_immune_off,
        u_perf=u_perf,          t_perf_on=t_perf_on,       t_perf_off=t_perf_off,
        u_metab=u_metab,        t_metab_on=t_metab_on,     t_metab_off=t_metab_off,
        u_nutrition=u_nutrition,t_nutrition_on=t_nutrition_on, t_nutrition_off=t_nutrition_off
    )

    hyperacute_th = 12.0
    acute_th      = 7.0
    subacute_th   = 3.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ---- Top: circuits ----
    ax0 = axes[0]
    ax0.step(t, E, where='post', label="E (Energy)")
    ax0.step(t, I, where='post', label="I (Immune)")
    ax0.step(t, N, where='post', label="N (Nerve)")
    ax0.step(t, A, where='post', label="A (Autonomic)")
    ax0.step(t, M, where='post', label="M (Perf/Mito)")
    ax0.set_ylabel("State")
    ax0.legend(ncol=3, fontsize=8)

    # ---- Bottom: NIMTI ----
    ax = axes[1]
    ymin_fixed = 0.0
    ymax_fixed = 24.0
    ax.set_ylim(ymin_fixed, ymax_fixed)

    ax.step(t, NIMTI, where='post', label="NIMTI")

    for th in [hyperacute_th, acute_th, subacute_th]:
        ax.axhline(th, color="gray", linewidth=0.8)

    ax.axhspan(hyperacute_th, ymax_fixed, alpha=0.08)
    ax.axhspan(acute_th,      hyperacute_th, alpha=0.08)
    ax.axhspan(subacute_th,   acute_th,      alpha=0.08)
    ax.axhspan(ymin_fixed,    subacute_th,   alpha=0.04)

    x_right = t[-1]
    ax.text(x_right, hyperacute_th + 0.5, "Death", ha="right", va="bottom", fontsize=8)
    ax.text(x_right, acute_th + 0.5,      "Unstable/Shock",      ha="right", va="bottom", fontsize=8)
    ax.text(x_right, subacute_th + 0.5,   "Stable",   ha="right", va="bottom", fontsize=8)
    ax.text(x_right, ymin_fixed + (subacute_th - ymin_fixed)*0.3, "Resolution",
            ha="right", va="bottom", fontsize=8)

    ax.set_ylabel("NIMTI")
    ax.set_xlabel("Time (hours)")
    ax.legend(loc="upper left", fontsize=8)

    # Treatment on/off lines
    treatments = [
        ("Vagus",    t_vagus_on,     t_vagus_off),
        ("Symp",     t_symp_on,      t_symp_off),
        ("Nerve",    t_nerve_on,     t_nerve_off),
        ("Immune",   t_immune_on,    t_immune_off),
        ("Perf",     t_perf_on,      t_perf_off),
        ("Metab",    t_metab_on,     t_metab_off),
        ("Nutr",     t_nutrition_on, t_nutrition_off),
    ]
    label_y = ymax_fixed + 0.5

    for name, t_on, t_off in treatments:
        if 0.0 <= t_on <= 72.0:
            ax0.axvline(t_on, linestyle=":", alpha=0.3)
            ax.axvline(t_on, linestyle=":", alpha=0.3)
        if 0.0 <= t_off <= 72.0 and t_off != t_on:
            ax0.axvline(t_off, linestyle="--", alpha=0.3)
            ax.axvline(t_off, linestyle="--", alpha=0.3)

    # Death @ X h = first crossing of hyperacute threshold
    idx_cross = None
    for k in range(1, len(t)):
        if (NIMTI[k-1] < hyperacute_th) and (NIMTI[k] >= hyperacute_th):
            idx_cross = k
            break
    if idx_cross is None and NIMTI[0] >= hyperacute_th:
        idx_cross = 0

    if idx_cross is not None:
        t_cross = t[idx_cross]
        ax.axvline(t_cross, linestyle='-', alpha=0.7)
        ax.text(t_cross, ymax_fixed*0.85, f"Death @ {t_cross:.1f} h",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.15))
    else:
        x_mid = t[0] + 0.5*(t[-1] - t[0])
        ax.text(x_mid, ymax_fixed*0.85, "Survival over time period",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.15))

    fig.suptitle("NIMTI Circuits, Phases, Timed Treatments, Death Time", fontsize=11)
    fig.tight_layout()
    return fig
