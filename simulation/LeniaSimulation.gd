extends Node

signal stats_updated(total_mass, population, histograms)
signal species_list_updated(species_list)
signal species_hovered(info) # New signal
# SpeciesTracker is a global class
var tracker = SpeciesTracker.new()
var camera: SimulationCamera



# === GLOBAL SIMULATION PARAMETERS ===
var params = {
	"res_x": 1024.0, 
	"res_y": 1024.0,
	"dt": 0.1,
	"seed": 0.0,
	# Kernel shape (global - creates pattern types)
	"R": 8.0,           # Kernel radius in pixels

	# Initialization
	"init_clusters": 16.0,
	"init_density": 1.0,   # Higher density for better start
	
	# Advanced Physics (Flow Lenia style)
	"temperature": 0.65,   # Advection diffusion (s). Paper default: 0.65
	"identity_thr": 0.2,   # Difference to be considered enemy (used in localized kernel if implemented)
	"colonize_thr": 0.15,  # Mass needed to resist invasion
	"theta_A": 5.0,        # Global Density Multiplier
	"alpha_n": 1.0,        # Repulsion Sharpness
	
	# Signal Layer
	"signal_diff": 1.0,    # Diffusion Rate
	"signal_decay": 0.1,   # Decay Rate
	"signal_advect": 1.0,  # Signal advection weight [0-1] (how much signals follow mass flow)
	"flow_speed": 1.0,     # Multiplier for advection force (decopuled from dt)
	
	"beta_selection": 1.0, # Selection pressure for negotiation rule
	
	# === GENE RANGES (16 GENES x 2 MIN/MAX) ===
	# BLOCK A: Physiology (Body)
	"g_mu_min": 0.0, "g_mu_max": 1.0,      # 1. Growth Target Density
	"g_sigma_min": 0.0, "g_sigma_max": 1.0,# 2. Growth Stability
	"g_radius_min": 0.0, "g_radius_max": 1.0,# 3. Size (Scale)
	"g_viscosity_min": 0.0, "g_viscosity_max": 1.0, # 4. Viscosity (Mass/Inertia)
	
	# BLOCK B: Morphology (Shape)
	"g_shape_a_min": 0.0, "g_shape_a_max": 1.0, # 5. Ring Balance
	"g_shape_b_min": 0.0, "g_shape_b_max": 1.0, # 6. Complexity
	"g_shape_c_min": 0.0, "g_shape_c_max": 1.0, # 7. Ring Spacing
	"g_ring_width_min": 0.0, "g_ring_width_max": 1.0, # 8. Ring Width (Sharpness)
	
	# BLOCK C: Social & Motor (Mind)
	"g_affinity_min": 0.0, "g_affinity_max": 1.0, # 9. Cohesion
	"g_repulsion_min": 0.0, "g_repulsion_max": 1.0, # 10. Spacing
	"g_density_tol_min": 0.0, "g_density_tol_max": 1.0, # 11. Overcrowding Tol
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

var shader_stats: RID
var shader_signal: RID # [NEW]
var pipeline_init: RID
var pipeline_conv: RID

var pipeline_stats: RID
var pipeline_analysis: RID
var pipeline_flow_conservative: RID
var pipeline_normalize: RID
var pipeline_signal: RID # [NEW]
var shader_analysis: RID
var shader_flow_conservative: RID
var shader_normalize: RID

# Textures: State (mass, vel, age) and Genome (8 genes packed)
var tex_state_a: RID
var tex_state_b: RID
var tex_genome_a: RID
var tex_genome_b: RID
var tex_genome_ext_a: RID # [NEW] Ext Texture (Genes 9-16)
var tex_genome_ext_b: RID # [NEW]
var tex_potential: RID
var tex_mass_accum: RID
var tex_winner_tracker: RID # [NEW]
var tex_signal_a: RID # [NEW]
var tex_signal_b: RID # [NEW]

# Bridges for display
var texture_rd_state: Texture2DRD
var texture_rd_genome: Texture2DRD
var texture_rd_genome_ext: Texture2DRD # [NEW]
var texture_rd_signal: Texture2DRD # [NEW]

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

# Analysis Thread
var analysis_thread: Thread
var is_analysis_running := false

# Async Readback State
var stats_readback_frame: int = -1
var analysis_readback_frame: int = -1
const READBACK_DELAY = 2 # Frames to wait for GPU

# Staggered offsets
var stats_frame_count := 0
var analysis_frame_count := 10 # Offset by 10 frames to avoid double-spike
var last_analysis_bytes: PackedByteArray
var last_species_list = []

# Camera state delegated to SimulationCamera
# var camera_pos := Vector2(0.0, 0.0)
# var camera_zoom := 1.0
# var is_dragging := false
# var is_inspecting := false # New: Track right-click state
# var last_mouse_pos := Vector2()

@export var display_material: ShaderMaterial

func _ready():
	rd = RenderingServer.get_rendering_device()
	
	if not rd:
		print("Compute shaders not supported (No RenderingDevice).")
		return

	_compile_shaders()
	_create_textures()
	_create_sampler()
	_create_uniforms()
	
	# Update UBO with initial params before shader runs
	_update_ubo()
	
	# Run Init once
	_dispatch_init()
	initialized = true
	
	# Setup Camera
	camera = SimulationCamera.new()
	add_child(camera)
	camera.inspect_requested.connect(_on_camera_inspect)
	
	print("Parametric Lenia with Signaling initialized.")

func _exit_tree():
	if analysis_thread and analysis_thread.is_started():
		analysis_thread.wait_to_finish()



func resize_simulation(new_size: int):
	# 1. Stop processing
	initialized = false
	
	# 2. Wait for background thread
	if analysis_thread and analysis_thread.is_started():
		analysis_thread.wait_to_finish()
	is_analysis_running = false
	
	# 3. Cleanup existing GPU resources
	_cleanup_gpu_resources()
	
	# 4. Update Params
	params["res_x"] = float(new_size)
	params["res_y"] = float(new_size)
	
	# 5. Re-create resources with new size
	_create_textures()
	_create_sampler()
	_create_uniforms()
	
	# 6. Clear cache (RIDs are invalid)
	set_cache.clear()
	
	_update_ubo()
	
	# 7. Initialize
	_dispatch_init()
	initialized = true
	
	print("Simulation Resized to: ", new_size)

func _cleanup_gpu_resources():
	# Free Buffers
	if ubo.is_valid(): rd.free_rid(ubo)
	if stats_buffer.is_valid(): rd.free_rid(stats_buffer)
	if analysis_buffer.is_valid(): rd.free_rid(analysis_buffer)
	
	# Free Textures
	var textures_to_free = [
		tex_state_a, tex_state_b,
		tex_genome_a, tex_genome_b, tex_genome_ext_a, tex_genome_ext_b,
		tex_potential, tex_mass_accum, tex_winner_tracker,
		tex_signal_a, tex_signal_b
	]
	
	for rid in textures_to_free:
		if rid.is_valid(): rd.free_rid(rid)

func _process(_delta):
	if not initialized: return
	
	# Update random seed
	params["seed"] = randf() * 1000.0
	
	# 1. Update UBO
	if not paused:
		_update_ubo()
		_dispatch_step()
	
	# 2. Performance Counters & Async Readbacks
	stats_frame_count += 1
	analysis_frame_count += 1

	var current_frame = Engine.get_process_frames()
	
	# === STATS SCHEDULE (Every 15 frames) ===
	if stats_frame_count >= 15:
		stats_frame_count = 0 
		_dispatch_stats()
		stats_readback_frame = current_frame + READBACK_DELAY

	# === STATS READBACK ===
	if current_frame == stats_readback_frame:
		# Retrieve Data (Should be ready now)
		var bytes = rd.buffer_get_data(stats_buffer)
		if bytes.size() >= 648:
			var ints = bytes.to_int32_array()
			var total_mass = float(ints[0]) / 1000.0
			var population = ints[1]
			
			# Parse histograms (16 genes * 10 bins)
			var histograms = []
			for g in range(16):
				var bins = []
				for b in range(10):
					bins.append(ints[2 + g * 10 + b])
				histograms.append(bins)
			emit_signal("stats_updated", total_mass, population, histograms)
	
	# === ANALYSIS SCHEDULE (Every 60 frames) ===
	if analysis_frame_count >= 60:
		analysis_frame_count = 0
		if not is_analysis_running: # Don't queue if previous is still running
			_dispatch_analysis()
			analysis_readback_frame = current_frame + READBACK_DELAY
	
	# === ANALYSIS READBACK & THREAD START ===
	if current_frame == analysis_readback_frame:
		var bytes = rd.buffer_get_data(analysis_buffer)
		if bytes.size() >= 294912:
			last_analysis_bytes = bytes
			
			# Start Background Thread
			if analysis_thread:
				if analysis_thread.is_started():
					analysis_thread.wait_to_finish() # Clean up previous (should be done)
			
			analysis_thread = Thread.new()
			is_analysis_running = true
			analysis_thread.start(_run_species_analysis.bind(bytes))
	
	# Update Display Material
	if display_material:
		var current_state = tex_state_b if ping_pong else tex_state_a
		var current_genome = tex_genome_b if ping_pong else tex_genome_a
		var current_signal = tex_signal_b if ping_pong else tex_signal_a
		
		if texture_rd_state.texture_rd_rid != current_state:
			texture_rd_state.texture_rd_rid = current_state
			
		if texture_rd_genome.texture_rd_rid != current_genome:
			texture_rd_genome.texture_rd_rid = current_genome
			
		if texture_rd_signal.texture_rd_rid != current_signal:
			texture_rd_signal.texture_rd_rid = current_signal
			
		var current_genome_ext = tex_genome_ext_b if ping_pong else tex_genome_ext_a
		if texture_rd_genome_ext.texture_rd_rid != current_genome_ext:
			texture_rd_genome_ext.texture_rd_rid = current_genome_ext
			
		display_material.set_shader_parameter("camera_pos", camera.camera_pos)
		display_material.set_shader_parameter("camera_zoom", camera.camera_zoom)
		display_material.set_shader_parameter("tex_genome_ext", texture_rd_genome_ext)

func _run_species_analysis(bytes: PackedByteArray):
	# Heavy lifting (200k+ iterations) done here
	var species_list = tracker.find_species(bytes)
	
	# Return to main thread
	call_deferred("_on_analysis_complete", species_list)

func _on_analysis_complete(species_list):
	is_analysis_running = false
	last_species_list = species_list
	emit_signal("species_list_updated", species_list)
	
	# Clean up thread reference if needed, though wait_to_finish covers it


func _update_ubo():
	# UBO layout: Must be carefully aligned to vec4 (16 bytes)
	# Matches std430 layout in shaders (Params block)
	# Total Floats: 16 (Globals) + 32 (Gene Ranges) = 48 floats
	var buffer = PackedFloat32Array([
		# Chunk 0 (0-16 bytes): Vec2 res + float dt + float seed
		params["res_x"], params["res_y"], params["dt"], params["seed"],
		
		# Chunk 1 (16-32 bytes): R, Theta, Alpha, Temp
		params["R"], params["theta_A"], params["alpha_n"], params["temperature"],
		
		# Chunk 2 (32-48 bytes): Signal Props + Beta
		params["signal_advect"], params["beta_selection"], params["signal_diff"], params["signal_decay"],
		
		# Chunk 3 (48-64 bytes): Flow + Init Props
		params["flow_speed"], params["init_clusters"], params["init_density"], params["colonize_thr"],
		
		# 2. Gene Ranges (16 Genes * 2 values = 32 floats)
		# Block A: Physiology (4 Genes)
		params["g_mu_min"], params["g_mu_max"], params["g_sigma_min"], params["g_sigma_max"],
		params["g_radius_min"], params["g_radius_max"], params["g_viscosity_min"], params["g_viscosity_max"],
		
		# Block B: Morphology (4 Genes)
		params["g_shape_a_min"], params["g_shape_a_max"], params["g_shape_b_min"], params["g_shape_b_max"],
		params["g_shape_c_min"], params["g_shape_c_max"], params["g_ring_width_min"], params["g_ring_width_max"],
		
		# Block C: Social / Motor (4 Genes)
		params["g_affinity_min"], params["g_affinity_max"], params["g_repulsion_min"], params["g_repulsion_max"],
		params["g_density_tol_min"], params["g_density_tol_max"], params["g_mobility_min"], params["g_mobility_max"],
		
		# Block D: Senses (4 Genes)
		params["g_secretion_min"], params["g_secretion_max"], params["g_sensitivity_min"], params["g_sensitivity_max"],
		params["g_emission_hue_min"], params["g_emission_hue_max"], params["g_detection_hue_min"], params["g_detection_hue_max"]
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
	var src_signal = tex_signal_a if not ping_pong else tex_signal_b
	var dst_signal = tex_signal_b if not ping_pong else tex_signal_a
	
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	# 1. Signal Evolution Pass
	var key_signal = "sig_" + str(ping_pong)
	var set_signal = set_cache.get(key_signal)
	if not set_signal or not set_signal.is_valid():
		set_signal = _create_set_signal(src_signal, dst_signal, src_state)
		set_cache[key_signal] = set_signal
		
	var compute_list_signal = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_signal, pipeline_signal)
	rd.compute_list_bind_uniform_set(compute_list_signal, set_signal, 0)
	rd.compute_list_dispatch(compute_list_signal, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE) # barrier automatically inserted
	
	# 2. Convolution Pass
	var key_conv = "conv_" + str(ping_pong)
	var set_conv = set_cache.get(key_conv)
	if not set_conv or not set_conv.is_valid():
		set_conv = _create_set_conv(src_state, src_genome, dst_signal, src_genome_ext, tex_potential)
		set_cache[key_conv] = set_conv
		
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_conv)
	rd.compute_list_bind_uniform_set(compute_list, set_conv, 0)
	rd.compute_list_dispatch(compute_list, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# 3. Flow Pass
	var cache_key_flow = "flow_" + str(ping_pong)
	var set_flow_con = set_cache.get(cache_key_flow)
	if not set_flow_con or not set_flow_con.is_valid():
		# Signature: src_state, src_genome, src_genome_ext, src_potential, src_sig, dst_mass, dst_state, dst_genome, dst_winner
		set_flow_con = _create_set_flow_conservative(src_state, src_genome, src_genome_ext, tex_potential, dst_signal, tex_mass_accum, dst_state, dst_genome, tex_winner_tracker)
		set_cache[cache_key_flow] = set_flow_con
	
	var compute_list_flow = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_flow, pipeline_flow_conservative)
	rd.compute_list_bind_uniform_set(compute_list_flow, set_flow_con, 0)
	rd.compute_list_dispatch(compute_list_flow, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE) # barrier automatically inserted
	
	# 4. Normalize Pass
	var key_norm = "norm_" + str(ping_pong)
	var set_norm = set_cache.get(key_norm)
	if not set_norm or not set_norm.is_valid():
		# Signature: src_mass, src_pot, old_state, dst_state, dst_sig, src_winner, dst_genome, old_genome, src_genome_ext, dst_genome_ext
		set_norm = _create_set_normalize(tex_mass_accum, tex_potential, src_state, dst_state, dst_signal, tex_winner_tracker, dst_genome, src_genome, src_genome_ext, dst_genome_ext)
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
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	var key_stats = "stats_" + str(ping_pong)
	var set_stats = set_cache.get(key_stats)
	if not set_stats or not set_stats.is_valid():
		set_stats = _create_set_stats(dst_state, dst_genome, dst_genome_ext)
		set_cache[key_stats] = set_stats
		
	# Clear stats buffer (162 uints = 648 bytes)
	var clear_bytes = PackedByteArray()
	clear_bytes.resize(648)
	rd.buffer_update(stats_buffer, 0, 648, clear_bytes)
	
	var compute_list_stats = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_stats, pipeline_stats)
	rd.compute_list_bind_uniform_set(compute_list_stats, set_stats, 0)
	rd.compute_list_dispatch(compute_list_stats, wg_x, wg_y, 1)
	rd.compute_list_end()

func _dispatch_analysis():
	var dst_state = tex_state_b if ping_pong else tex_state_a
	var dst_genome = tex_genome_b if ping_pong else tex_genome_a
	var dst_genome_ext = tex_genome_ext_b if ping_pong else tex_genome_ext_a
	
	var key_analysis = "analysis_" + str(ping_pong)
	var set_analysis = set_cache.get(key_analysis)
	if not set_analysis or not set_analysis.is_valid():
		set_analysis = _create_set_analysis(dst_state, dst_genome, dst_genome_ext)
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
	var uniform_set = _create_set_init(tex_state_a, tex_genome_a, tex_genome_ext_a)
	
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
	rd.texture_clear(tex_signal_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_winner_tracker, Color(0,0,0,0), 0, 1, 0, 1)
	
	# rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE) # barrier automatically inserted
	ping_pong = false

func _create_uniforms():
	# UBO: 48 floats * 4 bytes = 192 bytes
	var buffer = PackedFloat32Array()
	buffer.resize(48)
	var bytes = buffer.to_byte_array()
	ubo = rd.storage_buffer_create(bytes.size(), bytes)
	
	# Stats Buffer: 162 uints = 648 bytes
	var stats_bytes = PackedByteArray()
	stats_bytes.resize(648)
	stats_buffer = rd.storage_buffer_create(648, stats_bytes)
	
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
	
	# Signal Textures (RGBA32F for compatibility)
	var fmt_sig = RDTextureFormat.new()
	fmt_sig.width = int(params["res_x"])
	fmt_sig.height = int(params["res_y"])
	fmt_sig.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt_sig.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | 
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT |
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	)
	tex_signal_a = rd.texture_create(fmt_sig, RDTextureView.new())
	tex_signal_b = rd.texture_create(fmt_sig, RDTextureView.new())
	
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
	
	# Create Texture2DRD bridges for display
	texture_rd_state = Texture2DRD.new()
	texture_rd_genome = Texture2DRD.new()
	texture_rd_genome_ext = Texture2DRD.new()
	texture_rd_signal = Texture2DRD.new()
	texture_rd_state.texture_rd_rid = tex_state_a
	texture_rd_genome.texture_rd_rid = tex_genome_a
	texture_rd_genome_ext.texture_rd_rid = tex_genome_ext_a
	texture_rd_signal.texture_rd_rid = tex_signal_a
	
	if display_material:
		display_material.set_shader_parameter("tex_state", texture_rd_state)
		display_material.set_shader_parameter("tex_genome", texture_rd_genome)
		display_material.set_shader_parameter("tex_signal", texture_rd_signal)

func _compile_shaders():
	var paths = {
		"init": "res://simulation/shaders/compute_init.glsl",
		"conv": "res://simulation/shaders/compute_convolution.glsl",
		"stats": "res://simulation/shaders/compute_stats.glsl",
		"analysis": "res://simulation/shaders/compute_analysis.glsl",
		"flow_con": "res://simulation/shaders/compute_flow_conservative.glsl",
		"norm": "res://simulation/shaders/compute_normalize.glsl",
		"signal": "res://simulation/shaders/compute_signal.glsl"
	}
	
	shader_init = _load_shader(paths["init"])
	shader_conv = _load_shader(paths["conv"])
	shader_stats = _load_shader(paths["stats"])
	shader_analysis = _load_shader(paths["analysis"])
	shader_flow_conservative = _load_shader(paths["flow_con"])
	shader_normalize = _load_shader(paths["norm"])
	shader_signal = _load_shader(paths["signal"])
	
	# Validate Shaders before creating pipelines
	if not shader_init.is_valid(): push_error("Shader Init invalid")
	else: pipeline_init = rd.compute_pipeline_create(shader_init)
	
	if not shader_conv.is_valid(): push_error("Shader Conv invalid")
	else: pipeline_conv = rd.compute_pipeline_create(shader_conv)
	
	if not shader_stats.is_valid(): push_error("Shader Stats invalid")
	else: pipeline_stats = rd.compute_pipeline_create(shader_stats)
	
	if not shader_analysis.is_valid(): push_error("Shader Analysis invalid")
	else: pipeline_analysis = rd.compute_pipeline_create(shader_analysis)
	
	if not shader_flow_conservative.is_valid(): push_error("Shader FlowCon invalid")
	else: pipeline_flow_conservative = rd.compute_pipeline_create(shader_flow_conservative)
	
	if not shader_normalize.is_valid(): push_error("Shader Norm invalid")
	else: pipeline_normalize = rd.compute_pipeline_create(shader_normalize)
	
	if not shader_signal.is_valid(): push_error("Shader Signal invalid")
	else: pipeline_signal = rd.compute_pipeline_create(shader_signal)

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

func _create_set_init(dst_state: RID, dst_genome: RID, dst_genome_ext: RID) -> RID:
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
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_genome_ext], shader_init, 0)

func _create_set_signal(src_sig: RID, dst_sig: RID, src_state: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_src = RDUniform.new()
	u_src.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src.binding = 1
	u_src.add_id(sampler_linear)
	u_src.add_id(src_sig)
	
	var u_dst = RDUniform.new()
	u_dst.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst.binding = 2
	u_dst.add_id(dst_sig)
	
	var u_state = RDUniform.new()
	u_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_state.binding = 3
	u_state.add_id(sampler_linear)
	u_state.add_id(src_state)
	
	return rd.uniform_set_create([u_ubo, u_src, u_dst, u_state], shader_signal, 0)

func _create_set_conv(src_state: RID, src_genome: RID, src_sig: RID, src_genome_ext: RID, dst_potential: RID) -> RID:
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
	
	var u_sig = RDUniform.new()
	u_sig.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_sig.binding = 3
	u_sig.add_id(sampler_linear)
	u_sig.add_id(src_sig)
	
	var u_potential = RDUniform.new()
	u_potential.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_potential.binding = 4
	u_potential.add_id(dst_potential)
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext.binding = 5
	u_genome_ext.add_id(sampler_nearest)
	u_genome_ext.add_id(src_genome_ext)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_sig, u_potential, u_genome_ext], shader_conv, 0)



func _create_set_stats(tex_state: RID, tex_genome: RID, tex_genome_ext: RID) -> RID:
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
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_stats, u_genome_ext], shader_stats, 0)

func _create_set_analysis(tex_state: RID, tex_genome: RID, tex_genome_ext: RID) -> RID:
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
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext.binding = 3
	u_genome_ext.add_id(sampler_nearest)
	u_genome_ext.add_id(tex_genome_ext)
	
	var u_analysis = RDUniform.new()
	u_analysis.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_analysis.binding = 4
	u_analysis.add_id(analysis_buffer)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_genome_ext, u_analysis], shader_analysis, 0)

func _create_set_flow_conservative(src_state: RID, src_genome: RID, src_genome_ext: RID, src_potential: RID, src_sig: RID, dst_mass: RID, dst_state: RID, dst_genome: RID, dst_winner: RID) -> RID:
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
	
	var u_sig = RDUniform.new()
	u_sig.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_sig.binding = 7
	u_sig.add_id(sampler_linear)
	u_sig.add_id(src_sig)

	var u_winner = RDUniform.new()
	u_winner.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_winner.binding = 8
	u_winner.add_id(dst_winner)
	
	var u_genome_ext = RDUniform.new()
	u_genome_ext.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome_ext.binding = 9
	u_genome_ext.add_id(sampler_nearest)
	u_genome_ext.add_id(src_genome_ext)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_pot, u_mass, u_new_state, u_sig, u_winner, u_genome_ext], shader_flow_conservative, 0)


func _create_set_normalize(src_mass: RID, src_pot: RID, old_state: RID, dst_state: RID, dst_sig: RID, src_winner: RID, dst_genome: RID, old_genome: RID, src_genome_ext: RID, dst_genome_ext: RID) -> RID:
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
	
	var u_new_sig = RDUniform.new()
	u_new_sig.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_new_sig.binding = 5
	u_new_sig.add_id(dst_sig)
	
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
	
	return rd.uniform_set_create([u_ubo, u_mass, u_pot, u_old_state, u_new_state, u_new_sig, u_winner, u_new_genome, u_old_genome, u_genome_ext_src, u_genome_ext_dst], shader_normalize, 0)

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
	rd.texture_clear(tex_signal_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_signal_b, Color(0,0,0,0), 0, 1, 0, 1)
	# rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE) # barrier automatically inserted

func set_parameter(param_name: String, value: float):
	if params.has(param_name):
		params[param_name] = value

func get_parameter(param_name: String) -> float:
	return params.get(param_name, 0.0)

func set_highlight_genes(genes: Dictionary, active: bool):
	if display_material:
		display_material.set_shader_parameter("u_show_select", active)
		if active and genes.has("mu"):
			var v = Vector4(genes["mu"], genes["sigma"], genes["radius"], genes["affinity"])
			display_material.set_shader_parameter("u_select_vector", v)

func get_species_info_at(uv: Vector2) -> Dictionary:
	if last_analysis_bytes.is_empty(): return {}
	
	# Map UV (0-1) to Grid (64x64)
	var gx = int(uv.x * 64.0)
	var gy = int(uv.y * 64.0)
	if gx < 0 or gx >= 64 or gy < 0 or gy >= 64: return {}
	
	var idx = (gy * 64 + gx) * 18 # 18 floats per cell
	if (idx + 17) * 4 >= last_analysis_bytes.size(): return {} 
	
	var floats = last_analysis_bytes.to_float32_array()
	var base = idx
	
	var m = floats[base]
	if m < 0.05: return {} # Empty
	
	var info = {
		"mu": floats[base+1],
		"sigma": floats[base+2],
		"radius": floats[base+3],
		"viscosity": floats[base+4],
		"shape_a": floats[base+5],
		"shape_b": floats[base+6],
		"shape_c": floats[base+7],
		"ring_width": floats[base+8],
		"affinity": floats[base+9],
		"repulsion": floats[base+10],
		"density_tol": floats[base+11],
		"mobility": floats[base+12],
		"secretion": floats[base+13],
		"sensitivity": floats[base+14],
		"emission_hue": floats[base+15],
		"detection_hue": floats[base+16],
		"mass": m
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
