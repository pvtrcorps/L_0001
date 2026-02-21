extends Node

signal stats_updated(total_mass, population, histograms, detritus_mass)
signal species_list_updated(species_list)
signal species_hovered(info) # New signal
# SpeciesTracker is a global class
var tracker = SpeciesTracker.new()
var camera: SimulationCamera



# === GLOBAL SIMULATION PARAMETERS ===
var params = {
	"res_x": 1024.0, 
	"res_y": 1024.0,
	"dt": 0.2,
	"seed": 0.0,
	# Kernel shape (global - creates pattern types)
	"R": 16.0,           # Kernel radius in pixels

	# Initialization
	"init_clusters": 48.0,
	"init_density": 0.5,   # Higher density for better start
	"colonize_thr": 0.001, # [DUST THRESOLHD] Mass below this loses identity
	
	# Advanced Physics (Flow Lenia style)
	"temperature": 0.65,   # Advection diffusion (s). Paper default: 0.65
	"theta_A": 1.0,        # [DEPRECATED] UBO padding - not read by shaders
	"alpha_n": 4.0,        # [DEPRECATED] UBO padding - not read by shaders
	
	# Detritus Layer (replaces Signal Layer)
	"detritus_diff": 1.5,       # Detritus diffusion rate
	"mass_decay_rate": 0.005,   # Metabolic decay: living mass → detritus
	"detritus_advect": 0.8,     # Detritus advection by wind [0-2]
	"flow_speed": 1.0,     # [CANONICAL: 1.0]. Multiplier for advection force
	"fluid_momentum": 0.0, # [CANONICAL: 0.0]. Aristotelian Physics (No Inertia)
	
	"beta_selection": 1.0, # Selection pressure for negotiation rule
	
	# Wind / Atmosphere
	"wind_scale": 2.0,     # Noise scale
	"wind_strength": 0.5,  # Wind force multiplier
	"wind_speed": 0.05,     # Animation speed
	
	# Detritus Advanced (Feeding / Chemotaxis)
	"detritus_force_strength": 0.5,  # Multiplier for detritus gradient chemotaxis
	"mass_digest_rate": 0.1,         # Feeding rate: detritus → living mass
	"interaction_beta": 1.0,         # Hue-based inter-species force. 0=off
	"morph_anisotropy_gain": 1.0,    # Global gain for anisotropic morphology effects
	"morph_polarity_gain": 1.0,      # Global gain for internal polarity persistence/steering
	"morph_plasticity_gain": 1.0,    # Global gain for context-driven plasticity
	"morph_self_propulsion_gain": 1.0, # Global active propulsion along polarity axis
	
	# === GENE RANGES (16 GENES x 2 MIN/MAX) ===
	# BLOCK A: Physiology (Body)
	"g_mu_min": 0.05, "g_mu_max": 0.4,      # 1. Growth Target Density
	"g_sigma_min": 0.0, "g_sigma_max": 0.05,# 2. Growth Stability
	"g_radius_min": 0.0, "g_radius_max": 1.0,# 3. Size (Scale)
	"g_viscosity_min": 0.0, "g_viscosity_max": 1.0, # 4. Viscosity (Drag/Friction)
	
	# BLOCK B: Morphology (Shape)
	"g_shape_a_min": 0.0, "g_shape_a_max": 1.0, # 5. Ring Balance
	"g_shape_b_min": 0.0, "g_shape_b_max": 1.0, # 6. Complexity
	"g_shape_c_min": 0.0, "g_shape_c_max": 1.0, # 7. Ring Spacing
	"g_inertia_min": 0.0, "g_inertia_max": 1.0, # 8. Morphological anisotropy
	
	# BLOCK C: Social & Motor (Mind)
	"g_affinity_min": 0.0, "g_affinity_max": 1.0, # 9. Compactness (body cohesion)
	"g_repulsion_min": 0.0, "g_repulsion_max": 1.0, # 10. Hollow Core (Kernel shape)
	"g_density_tol_min": 0.0, "g_density_tol_max": 1.0, # 11. Morphological plasticity
	"g_mobility_min": 0.0, "g_mobility_max": 1.0, # 12. Speed Base
	
	# BLOCK D: Communication (Senses)
	"g_secretion_min": 0.0, "g_secretion_max": 1.0, # 13. Voice Vol
	"g_sensitivity_min": 0.0, "g_sensitivity_max": 1.0, # 14. Hearing
	"g_emission_hue_min": 0.0, "g_emission_hue_max": 1.0, # 15. Voice Pitch
	"g_detection_hue_min": 0.0, "g_detection_hue_max": 1.0 # 16. Hearing Pitch
}

# === RENDERING DEVICE RESOURCES ===
var rd: RenderingDevice
var shader_init: RID
var shader_conv: RID
var shader_conv_h: RID  # [SEPARABLE] Horizontal pass
var shader_conv_v: RID  # [SEPARABLE] Vertical pass

var shader_stats: RID
var shader_detritus: RID
var pipeline_init: RID
var pipeline_conv: RID
var pipeline_conv_h: RID  # [SEPARABLE]
var pipeline_conv_v: RID  # [SEPARABLE]

var pipeline_stats: RID
var pipeline_analysis: RID
var pipeline_flow_conservative: RID
var pipeline_normalize: RID
var pipeline_detritus: RID
var shader_analysis: RID
var shader_flow_conservative: RID
var shader_normalize: RID

# Textures: State (mass, vel, age) and Genome (8 genes packed)
var tex_state_a: RID
var tex_state_b: RID
var tex_genome_a: RID
var tex_genome_b: RID
var tex_genome_ext_a: RID # [NEW] Ext Texture (Genes 9-16)
var tex_genome_ext_b: RID
var tex_potential: RID
var tex_conv_intermediate: RID  # [SEPARABLE] H→V pass buffer
var tex_mass_accum: RID
var tex_winner_tracker: RID
var tex_detritus_a: RID  # R=mass, G=hue (replaces signal)
var tex_detritus_b: RID
var tex_detritus_mass_accum: RID  # R32_UINT atomic accumulator for detritus transport
var tex_detritus_hue_accum: RID   # R32_UINT atomic accumulator for hue transport
var tex_polarity_a: RID
var tex_polarity_b: RID

# Bridges for display
var texture_rd_state: Texture2DRD
var texture_rd_genome: Texture2DRD
var texture_rd_genome_ext: Texture2DRD # [NEW]
var texture_rd_detritus: Texture2DRD
var texture_rd_polarity: Texture2DRD

var ubo: RID
var stats_buffer: RID
var analysis_buffer: RID
var sampler_linear: RID
var sampler_nearest: RID
var ping_pong := false
var initialized := false
var paused := false # Pause state

# Uniform Set Cache
var set_cache = {}

var stats_frame_count := 0
var analysis_frame_count := 0
var last_analysis_bytes: PackedByteArray
var last_species_list = []

# Async Readback State
var stats_pending_frame := -1
var analysis_pending_frame := -1
var stats_interval := 10
var analysis_interval := 60
var params_time := 0.0
var is_analyzing := false


# Camera state delegated to SimulationCamera
# var camera_pos := Vector2(0.0, 0.0)
# var camera_zoom := 1.0
# var is_dragging := false
# var is_inspecting := false # New: Track right-click state
# var last_mouse_pos := Vector2()

@export var display_material: ShaderMaterial
@export var postprocess_material: ShaderMaterial

func _ready():
	rd = RenderingServer.get_rendering_device()
	
	if not rd:
		print("Compute shaders not supported (No RenderingDevice).")
		return

	_compile_shaders()
	_create_textures()
	_create_sampler()
	_create_uniforms()
	
	# Run Init once
	_dispatch_init()
	initialized = true
	
	# Setup Camera
	camera = SimulationCamera.new()
	add_child(camera)
	camera.inspect_requested.connect(_on_camera_inspect)
	
	print("Parametric Lenia with Detritus System initialized.")

func _process(delta):
	if not initialized or rd == null: return
	
	# Update random seed
	params["seed"] = randf() * 1000.0
	
	if not paused:
		params_time += delta
		
	# 1. Update UBO
	if not paused:
		_update_ubo()
		_dispatch_step()
	
	# 2. Async Stats Readback
	# Check if we have a pending stats request that is old enough (2 frames latency)
	if stats_pending_frame != -1 and Engine.get_process_frames() >= stats_pending_frame + 2:
		var bytes = rd.buffer_get_data(stats_buffer)
		stats_pending_frame = -1 # Reset
		
		if bytes.size() >= 652:
			var ints = bytes.to_int32_array()
			var total_mass = float(ints[0]) / 1000.0
			var population = ints[1]
			
			var histograms = []
			for g in range(16):
				var bins = []
				for b in range(10):
					bins.append(ints[2 + g * 10 + b])
				histograms.append(bins)
			
			var detritus_mass = float(ints[162]) / 1000.0
			emit_signal("stats_updated", total_mass, population, histograms, detritus_mass)

	# Schedule new stats dispatch if not pending and interval met
	stats_frame_count += 1
	if stats_pending_frame == -1 and stats_frame_count >= stats_interval:
		stats_frame_count = 0
		_dispatch_stats()
		stats_pending_frame = Engine.get_process_frames()
	
	# 3. Async Analysis Readback
	if analysis_pending_frame != -1 and Engine.get_process_frames() >= analysis_pending_frame + 2:
		var bytes = rd.buffer_get_data(analysis_buffer)
		analysis_pending_frame = -1
		
		if bytes.size() >= 294912:
			last_analysis_bytes = bytes
			# Post-process on a thread to avoid CPU spike
			WorkerThreadPool.add_task(func():
				var species_list = tracker.find_species(bytes)
				# Defer the UI update back to main thread
				call_deferred("_update_analysis_complete", species_list)
			)

	if analysis_pending_frame == -1 and not is_analyzing:
		analysis_frame_count += 1
	
	if analysis_pending_frame == -1 and not is_analyzing and analysis_frame_count >= analysis_interval:
		analysis_frame_count = 0
		is_analyzing = true
		_dispatch_analysis()
		analysis_pending_frame = Engine.get_process_frames()
	
	# Update Display Material
	if display_material:
		var current_state = tex_state_b if ping_pong else tex_state_a
		var current_genome = tex_genome_b if ping_pong else tex_genome_a
		var current_detritus = tex_detritus_b if ping_pong else tex_detritus_a
		var current_polarity = tex_polarity_b if ping_pong else tex_polarity_a
		
		if texture_rd_state.texture_rd_rid != current_state:
			texture_rd_state.texture_rd_rid = current_state
			
		if texture_rd_genome.texture_rd_rid != current_genome:
			texture_rd_genome.texture_rd_rid = current_genome
			
		if texture_rd_detritus.texture_rd_rid != current_detritus:
			texture_rd_detritus.texture_rd_rid = current_detritus
		
		if texture_rd_polarity.texture_rd_rid != current_polarity:
			texture_rd_polarity.texture_rd_rid = current_polarity
			
		var current_genome_ext = tex_genome_ext_b if ping_pong else tex_genome_ext_a
		if texture_rd_genome_ext.texture_rd_rid != current_genome_ext:
			texture_rd_genome_ext.texture_rd_rid = current_genome_ext
			
		display_material.set_shader_parameter("camera_pos", camera.camera_pos)
		display_material.set_shader_parameter("camera_zoom", camera.camera_zoom)
		display_material.set_shader_parameter("tex_genome_ext", texture_rd_genome_ext)
		display_material.set_shader_parameter("tex_polarity", texture_rd_polarity)
		
	if postprocess_material:
		postprocess_material.set_shader_parameter("camera_pos", camera.camera_pos)
		postprocess_material.set_shader_parameter("camera_zoom", camera.camera_zoom)

func _update_analysis_complete(species_list):
	is_analyzing = false
	last_species_list = species_list
	emit_signal("species_list_updated", species_list)

func _update_species_list(species_list):
	last_species_list = species_list
	emit_signal("species_list_updated", species_list)

func _update_ubo():
	# UBO layout: Must be carefully aligned to vec4 (16 bytes)
	# Matches std430 layout in shaders (Params block)
	# Total Floats: 16 (Globals) + 32 (Gene Ranges) + 4 (Wind) = 52 floats
	var buffer = PackedFloat32Array([
		# Chunk 0 (0-16 bytes): Vec2 res + float dt + float seed
		params["res_x"], params["res_y"], params["dt"], params["seed"],
		
		# Chunk 1 (16-32 bytes): R, Theta, Alpha, Temp
		params["R"], params["theta_A"], params["alpha_n"], params["temperature"],
		
		# Chunk 2 (32-48 bytes): Detritus Props + Beta
		params["detritus_advect"], params["beta_selection"], params["detritus_diff"], params["mass_decay_rate"],
		
		# Chunk 3 (48-64 bytes): Flow + Init Props
		params["flow_speed"], params["init_clusters"], params["init_density"], params["fluid_momentum"],
		
		# 2. Gene Ranges (16 Genes * 2 values = 32 floats)
		# Block A: Physiology (4 Genes)
		params["g_mu_min"], params["g_mu_max"], params["g_sigma_min"], params["g_sigma_max"],
		params["g_radius_min"], params["g_radius_max"], params["g_viscosity_min"], params["g_viscosity_max"],
		
		# Block B: Morphology (4 Genes)
		params["g_shape_a_min"], params["g_shape_a_max"], params["g_shape_b_min"], params["g_shape_b_max"],
		params["g_shape_c_min"], params["g_shape_c_max"], params["g_inertia_min"], params["g_inertia_max"],
		
		# Block C: Social / Motor (4 Genes)
		params["g_affinity_min"], params["g_affinity_max"], params["g_repulsion_min"], params["g_repulsion_max"],
		params["g_density_tol_min"], params["g_density_tol_max"], params["g_mobility_min"], params["g_mobility_max"],
		
		# Block D: Senses (4 Genes)
		params["g_secretion_min"], params["g_secretion_max"], params["g_sensitivity_min"], params["g_sensitivity_max"],
		params["g_emission_hue_min"], params["g_emission_hue_max"], params["g_detection_hue_min"], params["g_detection_hue_max"],
		
		# Chunk 4 (Wind/Atmosphere) - Appended to end
		params_time, params["wind_scale"], params["wind_strength"], params["wind_speed"],
		
		# Chunk 5 (Detritus + Morph Extras)
		params["detritus_force_strength"], params["mass_digest_rate"], params["interaction_beta"], params["morph_anisotropy_gain"],
		
		# Chunk 6 (Morph Controls + Cleanup)
		params["colonize_thr"], params["morph_polarity_gain"], params["morph_plasticity_gain"], params["morph_self_propulsion_gain"]
	])
	
	var bytes = buffer.to_byte_array()
	rd.buffer_update(ubo, 0, bytes.size(), bytes)

func _dispatch_step():
	var src_state = tex_state_a if not ping_pong else tex_state_b
	var dst_state = tex_state_b if not ping_pong else tex_state_a
	var src_genome = tex_genome_a if not ping_pong else tex_genome_b
	var dst_genome = tex_genome_b if not ping_pong else tex_genome_a
	var src_genome_ext = tex_genome_ext_a if not ping_pong else tex_genome_ext_b
	var dst_genome_ext = tex_genome_ext_b if not ping_pong else tex_genome_ext_a
	var src_detritus = tex_detritus_a if not ping_pong else tex_detritus_b
	var dst_detritus = tex_detritus_b if not ping_pong else tex_detritus_a
	var src_polarity = tex_polarity_a if not ping_pong else tex_polarity_b
	var dst_polarity = tex_polarity_b if not ping_pong else tex_polarity_a
	
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	# 1. Detritus Transport Pass (conservative)
	var key_det = "det_" + str(ping_pong)
	var set_det = set_cache.get(key_det)
	if not set_det or not set_det.is_valid():
		set_det = _create_set_detritus(src_detritus, tex_detritus_mass_accum, tex_detritus_hue_accum)
		set_cache[key_det] = set_det
		
	var compute_list_det = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_det, pipeline_detritus)
	rd.compute_list_bind_uniform_set(compute_list_det, set_det, 0)
	rd.compute_list_dispatch(compute_list_det, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# 2. Convolution Pass (Optimized 2D with 16×16 workgroups)
	var wg_conv_x = int(ceil(params["res_x"] / 16.0))
	var wg_conv_y = int(ceil(params["res_y"] / 16.0))
	
	var key_conv = "conv_" + str(ping_pong)
	var set_conv = set_cache.get(key_conv)
	if not set_conv or not set_conv.is_valid():
		set_conv = _create_set_conv(src_state, src_genome, src_detritus, src_genome_ext, tex_potential, src_polarity)
		set_cache[key_conv] = set_conv
		
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_conv)
	rd.compute_list_bind_uniform_set(compute_list, set_conv, 0)
	rd.compute_list_dispatch(compute_list, wg_conv_x, wg_conv_y, 1)
	rd.compute_list_end()
	
	# 3. Flow Pass
	var cache_key_flow = "flow_" + str(ping_pong)
	var set_flow_con = set_cache.get(cache_key_flow)
	if not set_flow_con or not set_flow_con.is_valid():
		set_flow_con = _create_set_flow_conservative(src_state, src_genome, src_genome_ext, tex_potential, src_detritus, tex_mass_accum, dst_state, dst_genome, tex_winner_tracker, src_polarity)
		set_cache[cache_key_flow] = set_flow_con
	
	var compute_list_flow = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_flow, pipeline_flow_conservative)
	rd.compute_list_bind_uniform_set(compute_list_flow, set_flow_con, 0)
	rd.compute_list_dispatch(compute_list_flow, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# 4. Normalize Pass (with detritus decay, feeding, and accumulator reconstruction)
	var key_norm = "norm_" + str(ping_pong)
	var set_norm = set_cache.get(key_norm)
	if not set_norm or not set_norm.is_valid():
		set_norm = _create_set_normalize(tex_mass_accum, tex_potential, src_state, dst_state, dst_detritus, tex_winner_tracker, dst_genome, src_genome, src_genome_ext, dst_genome_ext, src_polarity, dst_polarity, tex_detritus_mass_accum, tex_detritus_hue_accum)
		set_cache[key_norm] = set_norm
	
	var compute_list_norm = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_norm, pipeline_normalize)
	rd.compute_list_bind_uniform_set(compute_list_norm, set_norm, 0)
	rd.compute_list_dispatch(compute_list_norm, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	ping_pong = !ping_pong

func _dispatch_stats():
	var dst_state = tex_state_b if ping_pong else tex_state_a
	var dst_genome = tex_genome_b if ping_pong else tex_genome_a
	var dst_genome_ext = tex_genome_ext_b if ping_pong else tex_genome_ext_a
	var dst_detritus = tex_detritus_b if ping_pong else tex_detritus_a
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	var key_stats = "stats_" + str(ping_pong)
	var set_stats = set_cache.get(key_stats)
	if not set_stats or not set_stats.is_valid():
		set_stats = _create_set_stats(dst_state, dst_genome, dst_genome_ext, dst_detritus)
		set_cache[key_stats] = set_stats
		
	# Clear stats buffer (163 uints = 652 bytes)
	var clear_bytes = PackedByteArray()
	clear_bytes.resize(652)
	rd.buffer_update(stats_buffer, 0, 652, clear_bytes)
	
	var compute_list_stats = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_stats, pipeline_stats)
	rd.compute_list_bind_uniform_set(compute_list_stats, set_stats, 0)
	rd.compute_list_dispatch(compute_list_stats, wg_x, wg_y, 1)
	rd.compute_list_end()

func _dispatch_analysis():
	var dst_state = tex_state_b if ping_pong else tex_state_a
	var dst_genome = tex_genome_b if ping_pong else tex_genome_a
	var dst_genome_ext = tex_genome_ext_b if ping_pong else tex_genome_ext_a
	var dst_polarity = tex_polarity_b if ping_pong else tex_polarity_a
	
	var key_analysis = "analysis_" + str(ping_pong)
	var set_analysis = set_cache.get(key_analysis)
	if not set_analysis or not set_analysis.is_valid():
		set_analysis = _create_set_analysis(dst_state, dst_genome, dst_genome_ext, dst_polarity)
		set_cache[key_analysis] = set_analysis
	
	var compute_list_analysis = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_analysis, pipeline_analysis)
	rd.compute_list_bind_uniform_set(compute_list_analysis, set_analysis, 0)
	rd.compute_list_dispatch(compute_list_analysis, 8, 8, 1) 
	rd.compute_list_end()

func _dispatch_init():
	var x_groups = int(ceil(params["res_x"] / 8.0))
	var y_groups = int(ceil(params["res_y"] / 8.0))
	
	var compute_list = rd.compute_list_begin()
	
	# 3. Create set
	var uniform_set = _create_set_init(tex_state_a, tex_genome_a, tex_genome_ext_a, tex_polarity_a)
	
	# 4. Dispatch
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_init)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, x_groups, y_groups, 1)
	rd.compute_list_end()
	# rd.submit()
	# rd.sync()
	
	# After init, copy to B to be safe
	# (Alternatively just rely on first step logic)
	# Copy State A -> B
	rd.texture_copy(tex_state_a, tex_state_b, Vector3(0,0,0), Vector3(0,0,0), Vector3(params["res_x"], params["res_y"], 1), 0, 0, 0, 0)
	rd.texture_copy(tex_genome_a, tex_genome_b, Vector3(0,0,0), Vector3(0,0,0), Vector3(params["res_x"], params["res_y"], 1), 0, 0, 0, 0)
	rd.texture_copy(tex_genome_ext_a, tex_genome_ext_b, Vector3(0,0,0), Vector3(0,0,0), Vector3(params["res_x"], params["res_y"], 1), 0, 0, 0, 0)
	rd.texture_copy(tex_polarity_a, tex_polarity_b, Vector3(0,0,0), Vector3(0,0,0), Vector3(params["res_x"], params["res_y"], 1), 0, 0, 0, 0)
	rd.texture_clear(tex_detritus_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_detritus_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_mass_accum, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_winner_tracker, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_detritus_mass_accum, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_detritus_hue_accum, Color(0,0,0,0), 0, 1, 0, 1)
	
	# rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE) # barrier automatically inserted
	ping_pong = false

func _create_uniforms():
	# UBO: 52 floats * 4 bytes = 208 bytes
	var buffer = PackedFloat32Array()
	buffer.resize(60)
	var bytes = buffer.to_byte_array()
	ubo = rd.storage_buffer_create(bytes.size(), bytes)
	
	# Stats Buffer: 163 uints = 652 bytes (added detritus_mass)
	var stats_bytes = PackedByteArray()
	stats_bytes.resize(652)
	stats_buffer = rd.storage_buffer_create(652, stats_bytes)
	
	# Analysis Buffer: 4096 cells * 18 floats * 4 bytes = 294912 bytes
	var analysis_bytes = PackedByteArray()
	analysis_bytes.resize(294912)
	analysis_buffer = rd.storage_buffer_create(294912, analysis_bytes)

func _create_sampler():
	var sampler_state_linear = RDSamplerState.new()
	sampler_state_linear.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state_linear.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state_linear.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler_state_linear.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler_linear = rd.sampler_create(sampler_state_linear)
	
	var sampler_state_nearest = RDSamplerState.new()
	sampler_state_nearest.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state_nearest.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state_nearest.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler_state_nearest.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler_nearest = rd.sampler_create(sampler_state_nearest)

func _create_textures():
	var fmt = RDTextureFormat.new()
	fmt.width = int(params["res_x"])
	fmt.height = int(params["res_y"])
	fmt.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | 
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	)
	
	tex_state_a = rd.texture_create(fmt, RDTextureView.new())
	tex_state_b = rd.texture_create(fmt, RDTextureView.new())
	tex_genome_a = rd.texture_create(fmt, RDTextureView.new())
	tex_genome_b = rd.texture_create(fmt, RDTextureView.new())
	tex_genome_ext_a = rd.texture_create(fmt, RDTextureView.new())
	tex_genome_ext_b = rd.texture_create(fmt, RDTextureView.new())
	tex_potential = rd.texture_create(fmt, RDTextureView.new())
	
	# Detritus Textures (RGBA32F: R=mass, G=hue, BA=reserved)
	var fmt_det = RDTextureFormat.new()
	fmt_det.width = int(params["res_x"])
	fmt_det.height = int(params["res_y"])
	fmt_det.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt_det.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | 
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	)
	tex_detritus_a = rd.texture_create(fmt_det, RDTextureView.new())
	tex_detritus_b = rd.texture_create(fmt_det, RDTextureView.new())
	tex_polarity_a = rd.texture_create(fmt_det, RDTextureView.new())
	tex_polarity_b = rd.texture_create(fmt_det, RDTextureView.new())
	
	# Intermediate buffer for separable convolution (H→V pass)
	tex_conv_intermediate = rd.texture_create(fmt_det, RDTextureView.new())
	
	# Atomic Mass Accumulation (R32_UINT)
	var fmt_atomic = RDTextureFormat.new()
	fmt_atomic.width = int(params["res_x"])
	fmt_atomic.height = int(params["res_y"])
	fmt_atomic.format = RenderingDevice.DATA_FORMAT_R32_UINT
	fmt_atomic.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	)
	tex_mass_accum = rd.texture_create(fmt_atomic, RDTextureView.new())
	tex_winner_tracker = rd.texture_create(fmt_atomic, RDTextureView.new())
	tex_detritus_mass_accum = rd.texture_create(fmt_atomic, RDTextureView.new())
	tex_detritus_hue_accum = rd.texture_create(fmt_atomic, RDTextureView.new())
	
	# Create Texture2DRD bridges for display
	texture_rd_state = Texture2DRD.new()
	texture_rd_genome = Texture2DRD.new()
	texture_rd_genome_ext = Texture2DRD.new()
	texture_rd_detritus = Texture2DRD.new()
	texture_rd_polarity = Texture2DRD.new()
	texture_rd_state.texture_rd_rid = tex_state_a
	texture_rd_genome.texture_rd_rid = tex_genome_a
	texture_rd_genome_ext.texture_rd_rid = tex_genome_ext_a
	texture_rd_detritus.texture_rd_rid = tex_detritus_a
	texture_rd_polarity.texture_rd_rid = tex_polarity_a
	
	if display_material:
		display_material.set_shader_parameter("tex_state", texture_rd_state)
		display_material.set_shader_parameter("tex_genome", texture_rd_genome)
		display_material.set_shader_parameter("tex_detritus", texture_rd_detritus)
		display_material.set_shader_parameter("tex_polarity", texture_rd_polarity)

func _compile_shaders():
	var paths = {
		"init": "res://simulation/shaders/compute_init.glsl",
		"conv": "res://simulation/shaders/compute_convolution.glsl",
		"conv_h": "res://simulation/shaders/compute_conv_h.glsl",
		"conv_v": "res://simulation/shaders/compute_conv_v.glsl",
		"stats": "res://simulation/shaders/compute_stats.glsl",
		"analysis": "res://simulation/shaders/compute_analysis.glsl",
		"flow_con": "res://simulation/shaders/compute_flow_conservative.glsl",
		"norm": "res://simulation/shaders/compute_normalize.glsl",
		"detritus": "res://simulation/shaders/compute_detritus.glsl"
	}
	
	shader_init = _load_shader(paths["init"])
	shader_conv = _load_shader(paths["conv"])
	shader_conv_h = _load_shader(paths["conv_h"])
	shader_conv_v = _load_shader(paths["conv_v"])
	shader_stats = _load_shader(paths["stats"])
	shader_analysis = _load_shader(paths["analysis"])
	shader_flow_conservative = _load_shader(paths["flow_con"])
	shader_normalize = _load_shader(paths["norm"])
	shader_detritus = _load_shader(paths["detritus"])
	
	# Validate Shaders before creating pipelines
	if not shader_init.is_valid(): push_error("Shader Init invalid")
	else: pipeline_init = rd.compute_pipeline_create(shader_init)
	
	if not shader_conv.is_valid(): push_error("Shader Conv invalid")
	else: pipeline_conv = rd.compute_pipeline_create(shader_conv)
	
	# Separable Convolution Pipelines
	if not shader_conv_h.is_valid(): push_error("Shader Conv H invalid")
	else: pipeline_conv_h = rd.compute_pipeline_create(shader_conv_h)
	
	if not shader_conv_v.is_valid(): push_error("Shader Conv V invalid")
	else: pipeline_conv_v = rd.compute_pipeline_create(shader_conv_v)
	
	if not shader_stats.is_valid(): push_error("Shader Stats invalid")
	else: pipeline_stats = rd.compute_pipeline_create(shader_stats)
	
	if not shader_analysis.is_valid(): push_error("Shader Analysis invalid")
	else: pipeline_analysis = rd.compute_pipeline_create(shader_analysis)
	
	if not shader_flow_conservative.is_valid(): push_error("Shader FlowCon invalid")
	else: pipeline_flow_conservative = rd.compute_pipeline_create(shader_flow_conservative)
	
	if not shader_normalize.is_valid(): push_error("Shader Norm invalid")
	else: pipeline_normalize = rd.compute_pipeline_create(shader_normalize)
	
	if not shader_detritus.is_valid(): push_error("Shader Detritus invalid")
	else: pipeline_detritus = rd.compute_pipeline_create(shader_detritus)

func _load_shader(path: String) -> RID:
	# Prioritize Direct Source Loading to avoid .import lag
	if FileAccess.file_exists(path):
		var file = FileAccess.open(path, FileAccess.READ)
		var code = file.get_as_text()
		
		# Strip #[compute] directive
		if code.begins_with("#[compute]"):
			code = code.replace("#[compute]", "")
			
		var src = RDShaderSource.new()
		src.source_compute = code
		var spirv = rd.shader_compile_spirv_from_source(src)
		if spirv.compile_error_compute != "":
			push_error("Shader Compile Error: " + path + "\n" + spirv.compile_error_compute)
			return RID()
		return rd.shader_create_from_spirv(spirv)
	return RID()

# === UNIFORM SET CREATION HELPERS ===

func _create_set_init(dst_state: RID, dst_genome: RID, dst_genome_ext: RID, dst_polarity: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_state = RDUniform.new()
	u_state.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_state.binding = 1
	u_state.add_id(dst_state)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_genome.binding = 2
	u_genome.add_id(dst_genome)
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_genome_ext.binding = 3
	u_genome_ext.add_id(dst_genome_ext)

	var u_polarity = RDUniform.new()
	u_polarity.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_polarity.binding = 4
	u_polarity.add_id(dst_polarity)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_genome_ext, u_polarity], shader_init, 0)

func _create_set_detritus(src_det: RID, dst_mass_accum: RID, dst_hue_accum: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_src = RDUniform.new()
	u_src.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src.binding = 1
	u_src.add_id(sampler_linear)
	u_src.add_id(src_det)
	
	var u_mass_accum = RDUniform.new()
	u_mass_accum.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mass_accum.binding = 2
	u_mass_accum.add_id(dst_mass_accum)
	
	var u_hue_accum = RDUniform.new()
	u_hue_accum.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_hue_accum.binding = 3
	u_hue_accum.add_id(dst_hue_accum)
	
	return rd.uniform_set_create([u_ubo, u_src, u_mass_accum, u_hue_accum], shader_detritus, 0)

func _create_set_conv(src_state: RID, src_genome: RID, src_det: RID, src_genome_ext: RID, dst_potential: RID, src_polarity: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_state = RDUniform.new()
	u_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_state.binding = 1
	u_state.add_id(sampler_linear)
	u_state.add_id(src_state)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome.binding = 2
	u_genome.add_id(sampler_nearest)
	u_genome.add_id(src_genome)
	
	var u_det = RDUniform.new()
	u_det.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_det.binding = 3
	u_det.add_id(sampler_linear)
	u_det.add_id(src_det)
	
	var u_potential = RDUniform.new()
	u_potential.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_potential.binding = 4
	u_potential.add_id(dst_potential)
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext.binding = 5
	u_genome_ext.add_id(sampler_nearest)
	u_genome_ext.add_id(src_genome_ext)

	var u_polarity = RDUniform.new()
	u_polarity.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_polarity.binding = 6
	u_polarity.add_id(sampler_linear)
	u_polarity.add_id(src_polarity)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_det, u_potential, u_genome_ext, u_polarity], shader_conv, 0)

# === SEPARABLE CONVOLUTION UNIFORM SETS ===

func _create_set_conv_h(src_state: RID, src_detritus: RID, dst_intermediate: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_state = RDUniform.new()
	u_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_state.binding = 1
	u_state.add_id(sampler_nearest)
	u_state.add_id(src_state)
	
	var u_detritus = RDUniform.new()
	u_detritus.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_detritus.binding = 2
	u_detritus.add_id(sampler_nearest)
	u_detritus.add_id(src_detritus)
	
	var u_intermediate = RDUniform.new()
	u_intermediate.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_intermediate.binding = 3
	u_intermediate.add_id(dst_intermediate)
	
	return rd.uniform_set_create([u_ubo, u_state, u_detritus, u_intermediate], shader_conv_h, 0)

func _create_set_conv_v(src_intermediate: RID, src_genome: RID, src_genome_ext: RID, dst_potential: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_intermediate = RDUniform.new()
	u_intermediate.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_intermediate.binding = 1
	u_intermediate.add_id(sampler_nearest)
	u_intermediate.add_id(src_intermediate)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome.binding = 2
	u_genome.add_id(sampler_nearest)
	u_genome.add_id(src_genome)
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext.binding = 3
	u_genome_ext.add_id(sampler_nearest)
	u_genome_ext.add_id(src_genome_ext)
	
	var u_potential = RDUniform.new()
	u_potential.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_potential.binding = 4
	u_potential.add_id(dst_potential)
	
	return rd.uniform_set_create([u_ubo, u_intermediate, u_genome, u_genome_ext, u_potential], shader_conv_v, 0)


func _create_set_stats(tex_state: RID, tex_genome: RID, tex_genome_ext: RID, tex_det: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_state = RDUniform.new()
	u_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_state.binding = 1
	u_state.add_id(sampler_linear)
	u_state.add_id(tex_state)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome.binding = 2
	u_genome.add_id(sampler_nearest)
	u_genome.add_id(tex_genome)
	
	var u_stats = RDUniform.new()
	u_stats.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_stats.binding = 3
	u_stats.add_id(stats_buffer)
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext.binding = 4
	u_genome_ext.add_id(sampler_nearest)
	u_genome_ext.add_id(tex_genome_ext)
	
	var u_det = RDUniform.new()
	u_det.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_det.binding = 5
	u_det.add_id(sampler_linear)
	u_det.add_id(tex_det)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_stats, u_genome_ext, u_det], shader_stats, 0)

func _create_set_analysis(tex_state: RID, tex_genome: RID, tex_genome_ext: RID, tex_polarity: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_state = RDUniform.new()
	u_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_state.binding = 1
	u_state.add_id(sampler_nearest) # Changed to NEAREST to align with Grid
	u_state.add_id(tex_state)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome.binding = 2
	u_genome.add_id(sampler_nearest)
	u_genome.add_id(tex_genome)
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext.binding = 3
	u_genome_ext.add_id(sampler_nearest)
	u_genome_ext.add_id(tex_genome_ext)

	var u_polarity = RDUniform.new()
	u_polarity.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_polarity.binding = 4
	u_polarity.add_id(sampler_linear)
	u_polarity.add_id(tex_polarity)
	
	var u_analysis = RDUniform.new()
	u_analysis.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_analysis.binding = 5
	u_analysis.add_id(analysis_buffer)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_genome_ext, u_polarity, u_analysis], shader_analysis, 0)

func _create_set_flow_conservative(src_state: RID, src_genome: RID, src_genome_ext: RID, src_potential: RID, src_det: RID, dst_mass: RID, dst_state: RID, _dst_genome: RID, dst_winner: RID, src_polarity: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_state = RDUniform.new()
	u_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_state.binding = 1
	u_state.add_id(sampler_linear)
	u_state.add_id(src_state)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome.binding = 2
	u_genome.add_id(sampler_nearest)
	u_genome.add_id(src_genome)
	
	var u_pot = RDUniform.new()
	u_pot.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_pot.binding = 3
	u_pot.add_id(sampler_linear)
	u_pot.add_id(src_potential)
	
	var u_mass = RDUniform.new()
	u_mass.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mass.binding = 4
	u_mass.add_id(dst_mass)
	
	var u_new_state = RDUniform.new()
	u_new_state.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_new_state.binding = 5
	u_new_state.add_id(dst_state)
	
	# u_new_genome binding 6 removed (unused in shader)
	
	var u_det = RDUniform.new()
	u_det.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_det.binding = 7
	u_det.add_id(sampler_linear)
	u_det.add_id(src_det)

	var u_winner = RDUniform.new()
	u_winner.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_winner.binding = 8
	u_winner.add_id(dst_winner)
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext.binding = 9
	u_genome_ext.add_id(sampler_nearest)
	u_genome_ext.add_id(src_genome_ext)

	var u_polarity = RDUniform.new()
	u_polarity.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_polarity.binding = 10
	u_polarity.add_id(sampler_linear)
	u_polarity.add_id(src_polarity)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_pot, u_mass, u_new_state, u_det, u_winner, u_genome_ext, u_polarity], shader_flow_conservative, 0)


func _create_set_normalize(src_mass: RID, src_pot: RID, old_state: RID, dst_state: RID, dst_det: RID, src_winner: RID, dst_genome: RID, old_genome: RID, src_genome_ext: RID, dst_genome_ext: RID, src_polarity: RID, dst_polarity: RID, det_mass_accum: RID, det_hue_accum: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_mass = RDUniform.new()
	u_mass.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_mass.binding = 1
	u_mass.add_id(src_mass)
	
	var u_pot = RDUniform.new()
	u_pot.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_pot.binding = 2
	u_pot.add_id(sampler_linear)
	u_pot.add_id(src_pot)
	
	var u_old_state = RDUniform.new()
	u_old_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_old_state.binding = 3
	u_old_state.add_id(sampler_linear)
	u_old_state.add_id(old_state)
	
	var u_new_state = RDUniform.new()
	u_new_state.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_new_state.binding = 4
	u_new_state.add_id(dst_state)
	
	var u_det = RDUniform.new()
	u_det.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_det.binding = 5
	u_det.add_id(dst_det)
	
	var u_winner = RDUniform.new()
	u_winner.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_winner.binding = 6
	u_winner.add_id(src_winner)
	
	var u_new_genome = RDUniform.new()
	u_new_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_new_genome.binding = 7
	u_new_genome.add_id(dst_genome)
	
	var u_old_genome = RDUniform.new()
	u_old_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_old_genome.binding = 8
	u_old_genome.add_id(sampler_nearest)
	u_old_genome.add_id(old_genome)
	
	var u_genome_ext_src = RDUniform.new()
	u_genome_ext_src.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext_src.binding = 9
	u_genome_ext_src.add_id(sampler_nearest)
	u_genome_ext_src.add_id(src_genome_ext)
	
	var u_genome_ext_dst = RDUniform.new()
	u_genome_ext_dst.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_genome_ext_dst.binding = 10
	u_genome_ext_dst.add_id(dst_genome_ext)

	var u_polarity_src = RDUniform.new()
	u_polarity_src.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_polarity_src.binding = 11
	u_polarity_src.add_id(sampler_linear)
	u_polarity_src.add_id(src_polarity)

	var u_polarity_dst = RDUniform.new()
	u_polarity_dst.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_polarity_dst.binding = 12
	u_polarity_dst.add_id(dst_polarity)

	var u_det_mass_accum = RDUniform.new()
	u_det_mass_accum.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_det_mass_accum.binding = 13
	u_det_mass_accum.add_id(det_mass_accum)

	var u_det_hue_accum = RDUniform.new()
	u_det_hue_accum.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_det_hue_accum.binding = 14
	u_det_hue_accum.add_id(det_hue_accum)
	
	return rd.uniform_set_create([u_ubo, u_mass, u_pot, u_old_state, u_new_state, u_det, u_winner, u_new_genome, u_old_genome, u_genome_ext_src, u_genome_ext_dst, u_polarity_src, u_polarity_dst, u_det_mass_accum, u_det_hue_accum], shader_normalize, 0)

# === PUBLIC API ===

func reset_simulation():
	set_cache.clear()
	_dispatch_init()
	params["seed"] = randf() * 1000.0

func clear_simulation():
	rd.texture_clear(tex_state_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_state_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_genome_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_genome_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_genome_ext_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_genome_ext_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_detritus_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_detritus_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_polarity_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_polarity_b, Color(0,0,0,0), 0, 1, 0, 1)
	# rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE) # barrier automatically inserted

func change_resolution(w: float, h: float):
	if w == params["res_x"] and h == params["res_y"]:
		return
		
	print("Changing resolution to %dx%d..." % [w, h])
	initialized = false
	paused = true
	
	# Wait for device to be idle before freeing
	rd.free_rid(pipeline_init)
	rd.free_rid(pipeline_conv)
	rd.free_rid(pipeline_conv_h)
	rd.free_rid(pipeline_conv_v)
	rd.free_rid(pipeline_stats)
	rd.free_rid(pipeline_analysis)
	rd.free_rid(pipeline_flow_conservative)
	rd.free_rid(pipeline_normalize)
	rd.free_rid(pipeline_detritus)
	
	_free_resources()
	
	params["res_x"] = w
	params["res_y"] = h
	
	# Re-compile pipelines (shaders persist)
	pipeline_init = rd.compute_pipeline_create(shader_init)
	pipeline_conv = rd.compute_pipeline_create(shader_conv)
	pipeline_conv_h = rd.compute_pipeline_create(shader_conv_h)
	pipeline_conv_v = rd.compute_pipeline_create(shader_conv_v)
	pipeline_stats = rd.compute_pipeline_create(shader_stats)
	pipeline_analysis = rd.compute_pipeline_create(shader_analysis)
	pipeline_flow_conservative = rd.compute_pipeline_create(shader_flow_conservative)
	pipeline_normalize = rd.compute_pipeline_create(shader_normalize)
	pipeline_detritus = rd.compute_pipeline_create(shader_detritus)
	
	_create_textures()
	# Uniforms depend on texture RIDs, so recreate them?
	# Wait, _create_uniforms creates UBO/SSBO which are fixed size mostly,
	# BUT stats and analysis buffers depend on resolution!
	_create_uniforms() 
	
	# Reset Cache
	set_cache.clear()
	
	# Update camera limits? 
	# Camera doesn't have hard limits currently, just zoom.
	
	# Initialize
	_dispatch_init()
	initialized = true
	paused = false
	print("Resolution changed.")

func _free_resources():
	# Free Textures
	if tex_state_a.is_valid(): rd.free_rid(tex_state_a)
	if tex_state_b.is_valid(): rd.free_rid(tex_state_b)
	if tex_genome_a.is_valid(): rd.free_rid(tex_genome_a)
	if tex_genome_b.is_valid(): rd.free_rid(tex_genome_b)
	if tex_genome_ext_a.is_valid(): rd.free_rid(tex_genome_ext_a)
	if tex_genome_ext_b.is_valid(): rd.free_rid(tex_genome_ext_b)
	if tex_detritus_a.is_valid(): rd.free_rid(tex_detritus_a)
	if tex_detritus_b.is_valid(): rd.free_rid(tex_detritus_b)
	if tex_detritus_mass_accum.is_valid(): rd.free_rid(tex_detritus_mass_accum)
	if tex_detritus_hue_accum.is_valid(): rd.free_rid(tex_detritus_hue_accum)
	if tex_polarity_a.is_valid(): rd.free_rid(tex_polarity_a)
	if tex_polarity_b.is_valid(): rd.free_rid(tex_polarity_b)
	if tex_conv_intermediate.is_valid(): rd.free_rid(tex_conv_intermediate)
	if tex_potential.is_valid(): rd.free_rid(tex_potential)
	if tex_mass_accum.is_valid(): rd.free_rid(tex_mass_accum)
	if tex_winner_tracker.is_valid(): rd.free_rid(tex_winner_tracker)
	
	# Free Buffers
	if ubo.is_valid(): rd.free_rid(ubo)
	if stats_buffer.is_valid(): rd.free_rid(stats_buffer)
	if analysis_buffer.is_valid(): rd.free_rid(analysis_buffer)
	
	# Invalidate RIDs
	tex_state_a = RID()
	tex_state_b = RID()
	tex_genome_a = RID()
	tex_genome_b = RID()
	tex_genome_ext_a = RID()
	tex_genome_ext_b = RID()
	tex_detritus_a = RID()
	tex_detritus_b = RID()
	tex_detritus_mass_accum = RID()
	tex_detritus_hue_accum = RID()
	tex_polarity_a = RID()
	tex_polarity_b = RID()
	tex_conv_intermediate = RID()
	tex_potential = RID()
	tex_mass_accum = RID()
	tex_winner_tracker = RID()
	ubo = RID()
	stats_buffer = RID()
	analysis_buffer = RID()

func set_parameter(param_name: String, value: float):
	if params.has(param_name):
		params[param_name] = value

func get_parameter(param_name: String) -> float:
	return params.get(param_name, 0.0)

func set_highlight_genes(genes: Dictionary, active: bool):
	if display_material:
		display_material.set_shader_parameter("u_show_select", active)
		if active and genes.has("mu"):
			var v = Vector4(genes["mu"], genes["sigma"], genes["radius"], genes["repulsion"])
			display_material.set_shader_parameter("u_select_vector", v)

func get_species_info_at(uv: Vector2) -> Dictionary:
	if last_analysis_bytes.is_empty(): return {}
	
	# Map UV (0-1) to Grid (64x64)
	var gx = int(uv.x * 64.0)
	var gy = int(uv.y * 64.0)
	
	var floats = last_analysis_bytes.to_float32_array()
	var best_m = -1.0
	var best_idx = -1
	
	# 1. First check the EXACT cell under the cursor
	if gx >= 0 and gx < 64 and gy >= 0 and gy < 64:
		var idx = (gy * 64 + gx) * 18
		if idx * 4 < last_analysis_bytes.size():
			var m = floats[idx]
			if m > 0.001:
				best_m = m
				best_idx = idx
	
	# 2. If nothing EXACTLY under cursor, look at 3x3 neighbors to find peak
	if best_idx == -1:
		for ny in range(gy - 1, gy + 2):
			for nx in range(gx - 1, gx + 2):
				if nx < 0 or nx >= 64 or ny < 0 or ny >= 64: continue
				if nx == gx and ny == gy: continue # Already checked
				
				var idx = (ny * 64 + nx) * 18
				if idx * 4 >= last_analysis_bytes.size(): continue
				
				var m = floats[idx]
				if m > best_m:
					best_m = m
					best_idx = idx
	
	if best_idx == -1 or best_m < 0.001: return {}
	
	var base = best_idx
	var info = {
		"mu": floats[base+1],
		"sigma": floats[base+2],
		"radius": floats[base+3],
		"viscosity": floats[base+4],
		"shape_a": floats[base+5],
		"shape_b": floats[base+6],
		"shape_c": floats[base+7],
		"anisotropy": floats[base+8],
		"compactness": floats[base+9],
		"repulsion": floats[base+10],
		"plasticity": floats[base+11],
		"mobility": floats[base+12],
		"secretion": floats[base+13],
		"sensitivity": floats[base+14],
		"emission_hue": floats[base+15],
		"detection_hue": floats[base+16],
		"polarity": floats[base+17],
		"mass": best_m
	}
	
	# Find matching species in last_species_list
	for s in last_species_list:
		if SpeciesTracker.get_gene_distance(s.genes, info) < SpeciesTracker.GENE_SIMILARITY_THRESHOLD:
			info["id"] = s.id
			info["name"] = s.name
			info["color"] = s.color
			return info
			
	# If no match, generate a name on the fly for this "Wild" mutant
	var dummy_species = SpeciesTracker.Species.new()
	dummy_species.genes = info
	dummy_species._generate_name()
	
	info["id"] = "Wild"
	info["name"] = dummy_species.name
	return info

# === CAMERA CALLBACKS ===

func _on_camera_inspect(uv: Vector2, active: bool):
	if active:
		var info = get_species_info_at(uv)
		if info.has("id"):
			set_highlight_genes(info, true)
			emit_signal("species_hovered", info)
		else:
			set_highlight_genes({}, false)
			emit_signal("species_hovered", {})
	else:
		set_highlight_genes({}, false)
		emit_signal("species_hovered", {})

# Input handling moved to SimulationCamera.gd
