import streamlit as st
from nimti_model import plot_discrete_patient

st.set_page_config(page_title="NIMTI Simulator", layout="wide")

st.title("Neuro-Immuno-Metabolic Threat Index (NIMTI) Simulator")

st.markdown(
    "Use the controls to set injury severity, treatment strategies, "
    "and timing. The presets (Fentanyl Overdose, Naloxone Rescue, ARDS) "
    "will move the controls to physiologically plausible patterns, but "
    "you can still adjust everything afterward."
)

# ----------------------------
# Session state for parameters
# ----------------------------

if "params" not in st.session_state:
    st.session_state.params = dict(
        severity=0.7,
        u_vagus=0.0,   t_vagus_on=0.0,   t_vagus_off=72.0,
        u_symp=0.0,    t_symp_on=0.0,    t_symp_off=72.0,
        u_nerve=0.0,   t_nerve_on=0.0,   t_nerve_off=72.0,
        u_immune_tx=0.0,t_immune_on=0.0, t_immune_off=72.0,
        u_perf=0.0,    t_perf_on=0.0,    t_perf_off=72.0,
        u_metab=0.0,   t_metab_on=0.0,   t_metab_off=72.0,
        u_nutrition=0.0,t_nutrition_on=0.0,t_nutrition_off=72.0,
    )

P = st.session_state.params  # short alias


# ----------------------------
# Preset button callbacks
# ----------------------------

def preset_fentanyl():
    P["severity"]    = 0.9
    P["u_symp"]      = -0.8
    P["u_vagus"]     =  0.6
    P["u_nerve"]     =  0.7
    P["u_perf"]      = -0.5
    P["u_metab"]     = -0.5
    P["u_immune_tx"] = 0.0
    P["u_nutrition"] = 0.0
    for key in ["t_vagus_on","t_symp_on","t_nerve_on",
                "t_immune_on","t_perf_on","t_metab_on","t_nutrition_on"]:
        P[key] = 0.0
    for key in ["t_vagus_off","t_symp_off","t_nerve_off",
                "t_immune_off","t_perf_off","t_metab_off","t_nutrition_off"]:
        P[key] = 72.0

def preset_naloxone():
    # leaves severity as-is; modifies treatment pattern
    P["u_symp"]      =  0.4
    P["u_vagus"]     = -0.2
    P["u_perf"]      =  0.6
    P["u_metab"]     =  0.5
    P["u_nerve"]     = -0.2
    P["u_immune_tx"] =  0.0
    P["u_nutrition"] =  0.1
    for key in ["t_vagus_on","t_symp_on","t_nerve_on",
                "t_immune_on","t_perf_on","t_metab_on","t_nutrition_on"]:
        P[key] = 1.0
    for key in ["t_vagus_off","t_symp_off","t_nerve_off",
                "t_immune_off","t_perf_off","t_metab_off","t_nutrition_off"]:
        P[key] = 4.0

def preset_ards():
    P["severity"]    = 0.70
    P["u_immune_tx"] =  0.6
    P["u_symp"]      =  0.0
    P["u_vagus"]     = 0.0
    P["u_perf"]      = 0.0
    P["u_metab"]     = -0.5
    P["u_nerve"]     =  0.0
    P["u_nutrition"] = 0.0
    for key in ["t_vagus_on","t_symp_on","t_nerve_on",
                "t_immune_on","t_perf_on","t_metab_on","t_nutrition_on"]:
        P[key] = 0.0
    for key in ["t_vagus_off","t_symp_off","t_nerve_off",
                "t_immune_off","t_perf_off","t_metab_off","t_nutrition_off"]:
        P[key] = 72.0

# Layout: left = controls, right = plot
left, right = st.columns([1, 2])

with left:
    st.subheader("Injury & Presets")
    P["severity"] = st.slider(
        "Injury severity", 0.0, 1.0, value=float(P["severity"]), step=0.05
    )

    col_b1, col_b2, col_b3 = st.columns(3)
    if col_b1.button("Fentanyl Overdose"):
        preset_fentanyl()
    if col_b2.button("Naloxone Rescue"):
        preset_naloxone()
    if col_b3.button("Chlorine Inhalation"):
        preset_ards()

    st.markdown("---")
    st.subheader("Vagus & Sympathetic")
    P["u_vagus"] = st.slider("u_vagus (vagal tone)", -1.0, 1.0, float(P["u_vagus"]), 0.1)
    c1, c2 = st.columns(2)
    P["t_vagus_on"]  = c1.number_input("Vagus on (h)", 0.0, 72.0, float(P["t_vagus_on"]), 1.0)
    P["t_vagus_off"] = c2.number_input("Vagus off (h)", 0.0, 72.0, float(P["t_vagus_off"]), 1.0)

    P["u_symp"] = st.slider("u_symp (sympathetic)", -1.0, 1.0, float(P["u_symp"]), 0.1)
    c1, c2 = st.columns(2)
    P["t_symp_on"]  = c1.number_input("Symp on (h)", 0.0, 72.0, float(P["t_symp_on"]), 1.0)
    P["t_symp_off"] = c2.number_input("Symp off (h)", 0.0, 72.0, float(P["t_symp_off"]), 1.0)

    st.markdown("---")
    st.subheader("Nociception & Immune")
    P["u_nerve"] = st.slider("u_nerve (nociception)", -1.0, 1.0, float(P["u_nerve"]), 0.1)
    c1, c2 = st.columns(2)
    P["t_nerve_on"]  = c1.number_input("Nerve on (h)", 0.0, 72.0, float(P["t_nerve_on"]), 1.0)
    P["t_nerve_off"] = c2.number_input("Nerve off (h)", 0.0, 72.0, float(P["t_nerve_off"]), 1.0)

    P["u_immune_tx"] = st.slider("u_immune (immune modulation)", -1.0, 1.0,
                                 float(P["u_immune_tx"]), 0.1)
    c1, c2 = st.columns(2)
    P["t_immune_on"]  = c1.number_input("Immune on (h)", 0.0, 72.0, float(P["t_immune_on"]), 1.0)
    P["t_immune_off"] = c2.number_input("Immune off (h)", 0.0, 72.0, float(P["t_immune_off"]), 1.0)

    st.markdown("---")
    st.subheader("Perfusion & Metabolism")
    P["u_perf"] = st.slider("u_perf (perfusion)", -1.0, 1.0, float(P["u_perf"]), 0.1)
    c1, c2 = st.columns(2)
    P["t_perf_on"]  = c1.number_input("Perf on (h)", 0.0, 72.0, float(P["t_perf_on"]), 1.0)
    P["t_perf_off"] = c2.number_input("Perf off (h)", 0.0, 72.0, float(P["t_perf_off"]), 1.0)

    P["u_metab"] = st.slider("u_metab (mito/metabolic)", -1.0, 1.0, float(P["u_metab"]), 0.1)
    c1, c2 = st.columns(2)
    P["t_metab_on"]  = c1.number_input("Metab on (h)", 0.0, 72.0, float(P["t_metab_on"]), 1.0)
    P["t_metab_off"] = c2.number_input("Metab off (h)", 0.0, 72.0, float(P["t_metab_off"]), 1.0)

    st.markdown("---")
    st.subheader("Nutrition / Fuel Strategy")
    P["u_nutrition"] = st.slider("u_nutrition", -1.0, 1.0, float(P["u_nutrition"]), 0.1)
    c1, c2 = st.columns(2)
    P["t_nutrition_on"]  = c1.number_input("Nutr on (h)", 0.0, 72.0, float(P["t_nutrition_on"]), 1.0)
    P["t_nutrition_off"] = c2.number_input("Nutr off (h)", 0.0, 72.0, float(P["t_nutrition_off"]), 1.0)

with right:
    st.subheader("NIMTI & Phase Trajectories")
    fig = plot_discrete_patient(
        severity=P["severity"],
        u_vagus=P["u_vagus"], t_vagus_on=P["t_vagus_on"], t_vagus_off=P["t_vagus_off"],
        u_symp=P["u_symp"],   t_symp_on=P["t_symp_on"],   t_symp_off=P["t_symp_off"],
        u_nerve=P["u_nerve"], t_nerve_on=P["t_nerve_on"], t_nerve_off=P["t_nerve_off"],
        u_immune_tx=P["u_immune_tx"], t_immune_on=P["t_immune_on"], t_immune_off=P["t_immune_off"],
        u_perf=P["u_perf"],   t_perf_on=P["t_perf_on"],   t_perf_off=P["t_perf_off"],
        u_metab=P["u_metab"], t_metab_on=P["t_metab_on"], t_metab_off=P["t_metab_off"],
        u_nutrition=P["u_nutrition"],
        t_nutrition_on=P["t_nutrition_on"], t_nutrition_off=P["t_nutrition_off"],
    )
    st.pyplot(fig, clear_figure=True)
