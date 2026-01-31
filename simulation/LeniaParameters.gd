class_name LeniaParameters
extends Resource

# === GLOBAL SIMULATION PARAMETERS ===
@export_group("Simulation Settings")
@export var res_x: float = 1024.0
@export var res_y: float = 1024.0
@export var dt: float = 0.1
@export var seed_value: float = 0.0

@export_group("Kernel Geometry")
@export var R: float = 12.0           # Kernel radius in pixels

@export_group("Initialization")
@export var init_clusters: float = 16.0
@export var init_density: float = 0.5   # Higher density for better start

@export_group("Flow Physics")
@export var temperature: float = 0.65   # Advection diffusion (s)
@export var identity_thr: float = 0.2   # Difference to be considered enemy
@export var colonize_thr: float = 0.15  # Mass needed to resist invasion
@export var theta_A: float = 1.0        # Global Density Multiplier
@export var alpha_n: float = 3.0        # Repulsion Sharpness
@export var flow_speed: float = 5.0     # Multiplier for advection force
@export var beta_selection: float = 1.0 # Selection pressure

@export_group("Signal Layer")
@export var signal_diff: float = 2.0    # Diffusion Rate
@export var signal_decay: float = 0.1   # Decay Rate
@export var signal_advect: float = 0.2  # Signal advection weight [0-1]

# === GENE RANGES (16 GENES x 2 MIN/MAX) ===
@export_group("Genetics: Physiology")
@export var g_mu_min: float = 0.12
@export var g_mu_max: float = 0.18
@export var g_sigma_min: float = 0.02
@export var g_sigma_max: float = 0.06
@export var g_radius_min: float = 0.5
@export var g_radius_max: float = 1.0
@export var g_viscosity_min: float = 0.0
@export var g_viscosity_max: float = 1.0

@export_group("Genetics: Morphology")
@export var g_shape_a_min: float = 0.0
@export var g_shape_a_max: float = 1.0
@export var g_shape_b_min: float = 0.0
@export var g_shape_b_max: float = 1.0
@export var g_shape_c_min: float = 0.0
@export var g_shape_c_max: float = 1.0
@export var g_ring_width_min: float = 0.0
@export var g_ring_width_max: float = 1.0

@export_group("Genetics: Social & Motor")
@export var g_affinity_min: float = 0.0
@export var g_affinity_max: float = 1.0
@export var g_repulsion_min: float = 0.0
@export var g_repulsion_max: float = 1.0
@export var g_density_tol_min: float = 0.0
@export var g_density_tol_max: float = 1.0
@export var g_mobility_min: float = 0.0
@export var g_mobility_max: float = 1.0

@export_group("Genetics: Communication")
@export var g_secretion_min: float = 0.0
@export var g_secretion_max: float = 1.0
@export var g_sensitivity_min: float = 0.0
@export var g_sensitivity_max: float = 1.0
@export var g_emission_hue_min: float = 0.0
@export var g_emission_hue_max: float = 1.0
@export var g_detection_hue_min: float = 0.0
@export var g_detection_hue_max: float = 1.0
