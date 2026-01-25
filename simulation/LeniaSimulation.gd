extends Node

signal stats_updated(total_mass, population, histograms)
signal species_list_updated(species_list)
signal species_hovered(info) # New signal
# SpeciesTracker is a global class
var tracker = SpeciesTracker.new()


# === GLOBAL SIMULATION PARAMETERS ===
var params = {
	"res_x": 1024.0, 
	"res_y": 1024.0,
	"dt": 0.1,
	"seed": 0.0,
	# Kernel shape (global - creates pattern types)
	"R": 12.0,           # Kernel radius in pixels

	# Initialization
	"init_clusters": 16.0,
	"init_density": 0.5,   # Higher density for better start
	
	# Advanced Physics (Flow Lenia style)
	"temperature": 0.65,   # Advection diffusion (s). Paper default: 0.65
	"identity_thr": 0.2,   # Difference to be considered enemy (used in localized kernel if implemented)
	"colonize_thr": 0.15,  # Mass needed to resist invasion
	
	# Signal Layer
	"signal_diff": 2.0,    # Diffusion Rate
	"signal_decay": 0.1,   # Decay Rate
	
	# Gene Ranges (Min, Max) [0.0, 1.0]
	"g_mu_min": 0.0, "g_mu_max": 1.0,
	"g_sigma_min": 0.0, "g_sigma_max": 1.0, 
	"g_radius_min": 0.0, "g_radius_max": 1.0,
	"g_flow_min": 0.0, "g_flow_max": 1.0,
	"g_affinity_min": 0.0, "g_affinity_max": 1.0,
	"g_lambda_min": 0.0, "g_lambda_max": 1.0,
	"g_secretion_min": 0.0, "g_secretion_max": 1.0,
	"g_perception_min": 0.0, "g_perception_max": 1.0,
	
	# Pressure / Density Control (Eq 5)
	"theta_A": 2.0,       # Critical Mass (Repulsion kicks in above this)
	"alpha_n": 2.0        # Smoothness of the transition to repulsion
}

# === RENDERING DEVICE RESOURCES ===
var rd: RenderingDevice
var shader_init: RID
var shader_conv: RID
var shader_flow: RID
var shader_stats: RID
var shader_signal: RID # [NEW]
var pipeline_init: RID
var pipeline_conv: RID
var pipeline_flow: RID
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
var tex_potential: RID
var tex_mass_accum: RID
var tex_winner_tracker: RID # [NEW]
var tex_signal_a: RID # [NEW]
var tex_signal_b: RID # [NEW]

# Bridges for display
var texture_rd_state: Texture2DRD
var texture_rd_genome: Texture2DRD
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

var stats_frame_count := 0
var analysis_frame_count := 0
var last_analysis_bytes: PackedByteArray
var last_species_list = []

# Camera state
var camera_pos := Vector2(0.0, 0.0)
var camera_zoom := 1.0
var is_dragging := false
var is_inspecting := false # New: Track right-click state
var last_mouse_pos := Vector2()

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
	
	# Run Init once
	_dispatch_init()
	initialized = true
	print("Parametric Lenia with Signaling initialized.")

func _process(_delta):
	if not initialized: return
	
	# Update random seed
	params["seed"] = randf() * 1000.0
	
	# 1. Update UBO
	if not paused:
		_update_ubo()
		_dispatch_step()
	
	# 2. Performance Counters & Throttled Readbacks
	stats_frame_count += 1
	if stats_frame_count >= 10:
		stats_frame_count = 0
		# Run Stats Pass ONLY when needed
		_dispatch_stats()
		var bytes = rd.buffer_get_data(stats_buffer)
		if bytes.size() >= 328:
			var data = bytes.to_int32_array()
			var total_mass = float(data[0]) / 1000.0
			var population = data[1]
			var histograms = []
			for i in range(2, 82): histograms.append(data[i])
			emit_signal("stats_updated", total_mass, population, histograms)
	
	analysis_frame_count += 1
	if analysis_frame_count >= 30:
		analysis_frame_count = 0
		# Run Analysis Pass ONLY when needed
		_dispatch_analysis()
		var bytes = rd.buffer_get_data(analysis_buffer)
		if bytes.size() >= 163840:
			last_analysis_bytes = bytes
			var species_list = tracker.find_species(bytes)
			last_species_list = species_list
			emit_signal("species_list_updated", species_list)
	
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
			
		display_material.set_shader_parameter("camera_pos", camera_pos)
		display_material.set_shader_parameter("camera_zoom", camera_zoom)

func _update_ubo():
	# UBO layout: 32 floats (128 bytes)
	var buffer = PackedFloat32Array([
		params["res_x"], params["res_y"],      # vec2 u_res
		params["dt"], params["seed"],          # float u_dt, u_seed
		# Kernel parameters
		# Kernel parameters
		params["R"],                           # float u_R
		params["theta_A"],                     # Repurposed _pad1 (Critical Mass)
		params["alpha_n"],                     # Repurposed _pad2 (Alpha Exponent)
		params["temperature"],                 # float u_temperature (was _pad3)
		# Evolution
		0.0,                                   # Padding (Removed mutation_rate)
		0.0,                                   # Padding (Removed base_decay)
		# Initialization  
		params["init_clusters"],               # float u_init_clusters
		params["init_density"],                # float u_init_density
		params["colonize_thr"],                # Threshold for colony resistance
		0.0,                                   # Padding
		# Gene Ranges (12 floats)
		params["g_mu_min"], params["g_mu_max"],
		params["g_sigma_min"], params["g_sigma_max"],
		params["g_radius_min"], params["g_radius_max"],
		params["g_flow_min"], params["g_flow_max"],
		params["g_affinity_min"], params["g_affinity_max"],
		params["g_lambda_min"], params["g_lambda_max"],
		# Signal Params & More Ranges (4 floats)
		params["signal_diff"], params["signal_decay"],
		params["g_secretion_min"], params["g_secretion_max"],
		params["g_perception_min"], params["g_perception_max"],
		0.0, 0.0                               # Padding to 32 floats
	])
	var bytes = buffer.to_byte_array()
	rd.buffer_update(ubo, 0, bytes.size(), bytes)

func _dispatch_step():
	var src_state = tex_state_b if ping_pong else tex_state_a
	var dst_state = tex_state_a if ping_pong else tex_state_b
	var src_genome = tex_genome_b if ping_pong else tex_genome_a
	var dst_genome = tex_genome_a if ping_pong else tex_genome_b
	
	var src_signal = tex_signal_b if ping_pong else tex_signal_a
	var dst_signal = tex_signal_a if ping_pong else tex_signal_b
	
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	# 1. Signal Evolution Pass
	var key_signal = "sig_" + str(ping_pong)
	var set_signal = set_cache.get(key_signal)
	if not set_signal or not set_signal.is_valid():
		set_signal = _create_set_signal(src_signal, dst_signal)
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
		set_conv = _create_set_conv(src_state, src_genome, dst_signal, tex_potential)
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
		set_flow_con = _create_set_flow_conservative(src_state, src_genome, dst_signal, tex_potential, dst_state, dst_genome)
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
		set_norm = _create_set_normalize(tex_potential, src_genome, dst_state, dst_signal, dst_genome)
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
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	var key_stats = "stats_" + str(ping_pong)
	var set_stats = set_cache.get(key_stats)
	if not set_stats or not set_stats.is_valid():
		set_stats = _create_set_stats(dst_state, dst_genome)
		set_cache[key_stats] = set_stats
		
	# Clear stats buffer (82 uints = 328 bytes)
	var clear_bytes = PackedByteArray()
	clear_bytes.resize(328)
	rd.buffer_update(stats_buffer, 0, 328, clear_bytes)
	
	var compute_list_stats = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_stats, pipeline_stats)
	rd.compute_list_bind_uniform_set(compute_list_stats, set_stats, 0)
	rd.compute_list_dispatch(compute_list_stats, wg_x, wg_y, 1)
	rd.compute_list_end()

func _dispatch_analysis():
	var dst_state = tex_state_b if ping_pong else tex_state_a
	var dst_genome = tex_genome_b if ping_pong else tex_genome_a
	
	var key_analysis = "analysis_" + str(ping_pong)
	var set_analysis = set_cache.get(key_analysis)
	if not set_analysis or not set_analysis.is_valid():
		set_analysis = _create_set_analysis(dst_state, dst_genome)
		set_cache[key_analysis] = set_analysis
	
	var compute_list_analysis = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_analysis, pipeline_analysis)
	rd.compute_list_bind_uniform_set(compute_list_analysis, set_analysis, 0)
	rd.compute_list_dispatch(compute_list_analysis, 8, 8, 1) 
	rd.compute_list_end()

func _dispatch_init():
	var set_init = _create_set_init(tex_state_a, tex_genome_a)
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_init)
	rd.compute_list_bind_uniform_set(compute_list, set_init, 0)
	rd.compute_list_dispatch(compute_list, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# Also clear buffers
	rd.texture_clear(tex_state_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_genome_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_signal_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_signal_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_winner_tracker, Color(0,0,0,0), 0, 1, 0, 1)
	
	# rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE) # barrier automatically inserted
	ping_pong = false

func _create_uniforms():
	# UBO: 34 floats = 136 bytes (aligned to 8 bytes for vec2s)
	var buffer = PackedFloat32Array()
	buffer.resize(34)
	var bytes = buffer.to_byte_array()
	ubo = rd.storage_buffer_create(bytes.size(), bytes)
	
	# Stats Buffer: 82 uints = 328 bytes
	var stats_bytes = PackedByteArray()
	stats_bytes.resize(328)
	stats_buffer = rd.storage_buffer_create(328, stats_bytes)
	
	# Analysis Buffer: 4096 cells * 10 floats (40 bytes) = 163840 bytes
	var analysis_bytes = PackedByteArray()
	analysis_bytes.resize(163840)
	analysis_buffer = rd.storage_buffer_create(163840, analysis_bytes)

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
	texture_rd_signal = Texture2DRD.new()
	texture_rd_state.texture_rd_rid = tex_state_a
	texture_rd_genome.texture_rd_rid = tex_genome_a
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

func _create_set_init(dst_state: RID, dst_genome: RID) -> RID:
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
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome], shader_init, 0)

func _create_set_signal(src_sig: RID, dst_sig: RID) -> RID:
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
	
	return rd.uniform_set_create([u_ubo, u_src, u_dst], shader_signal, 0)

func _create_set_conv(src_state: RID, src_genome: RID, src_sig: RID, dst_potential: RID) -> RID:
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
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_sig, u_potential], shader_conv, 0)

func _create_set_flow(src_state: RID, src_genome: RID, src_potential: RID, 
					  dst_state: RID, dst_genome: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_src_state = RDUniform.new()
	u_src_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_state.binding = 1
	u_src_state.add_id(sampler_linear)
	u_src_state.add_id(src_state)
	
	var u_src_genome = RDUniform.new()
	u_src_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_genome.binding = 2
	u_src_genome.add_id(sampler_nearest)
	u_src_genome.add_id(src_genome)
	
	var u_src_potential = RDUniform.new()
	u_src_potential.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_potential.binding = 3
	u_src_potential.add_id(sampler_linear)
	u_src_potential.add_id(src_potential)
	
	var u_dst_state = RDUniform.new()
	u_dst_state.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst_state.binding = 4
	u_dst_state.add_id(dst_state)
	
	var u_dst_genome = RDUniform.new()
	u_dst_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst_genome.binding = 5
	u_dst_genome.add_id(dst_genome)
	
	return rd.uniform_set_create([u_ubo, u_src_state, u_src_genome, u_src_potential, 
								  u_dst_state, u_dst_genome], shader_flow, 0)

func _create_set_stats(tex_state: RID, tex_genome: RID) -> RID:
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
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_stats], shader_stats, 0)

func _create_set_analysis(tex_state: RID, tex_genome: RID) -> RID:
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
	
	var u_analysis = RDUniform.new()
	u_analysis.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_analysis.binding = 3
	u_analysis.add_id(analysis_buffer)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_analysis], shader_analysis, 0)

func _create_set_flow_conservative(src_state: RID, src_genome: RID, src_sig: RID, tex_pot: RID, dst_state: RID, dst_genome: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_src_state = RDUniform.new()
	u_src_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_state.binding = 1
	u_src_state.add_id(sampler_linear)
	u_src_state.add_id(src_state)
	
	var u_src_genome = RDUniform.new()
	u_src_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_genome.binding = 2
	u_src_genome.add_id(sampler_nearest)
	u_src_genome.add_id(src_genome)
	
	var u_pot = RDUniform.new()
	u_pot.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_pot.binding = 3
	u_pot.add_id(sampler_linear)
	u_pot.add_id(tex_pot)
	
	var u_atomic = RDUniform.new()
	u_atomic.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_atomic.binding = 4
	u_atomic.add_id(tex_mass_accum)
	
	var u_dst_state = RDUniform.new()
	u_dst_state.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst_state.binding = 5
	u_dst_state.add_id(dst_state)
	
	var u_dst_genome = RDUniform.new()
	u_dst_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst_genome.binding = 6
	u_dst_genome.add_id(dst_genome)
	
	var u_sig = RDUniform.new()
	u_sig.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_sig.binding = 7
	u_sig.add_id(sampler_linear)
	u_sig.add_id(src_sig)
	
	var u_winner = RDUniform.new()
	u_winner.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_winner.binding = 8
	u_winner.add_id(tex_winner_tracker)
	
	return rd.uniform_set_create([u_ubo, u_src_state, u_src_genome, u_pot, u_atomic, u_dst_state, u_dst_genome, u_sig, u_winner], shader_flow_conservative, 0)

func _create_set_normalize(tex_pot: RID, old_genome: RID, dst_state: RID, dst_sig: RID, dst_genome: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_atomic = RDUniform.new()
	u_atomic.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_atomic.binding = 1
	u_atomic.add_id(tex_mass_accum)
	
	var u_pot = RDUniform.new()
	u_pot.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_pot.binding = 2
	u_pot.add_id(sampler_linear)
	u_pot.add_id(tex_pot)
	
	var u_dst_state = RDUniform.new()
	u_dst_state.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst_state.binding = 4
	u_dst_state.add_id(dst_state)
	
	var u_dst_sig = RDUniform.new()
	u_dst_sig.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst_sig.binding = 5
	u_dst_sig.add_id(dst_sig)
	
	var u_winner = RDUniform.new()
	u_winner.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_winner.binding = 6
	u_winner.add_id(tex_winner_tracker)
	
	var u_new_genome = RDUniform.new()
	u_new_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_new_genome.binding = 7
	u_new_genome.add_id(dst_genome)
	
	var u_old_genome = RDUniform.new()
	u_old_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_old_genome.binding = 8
	u_old_genome.add_id(sampler_nearest)
	u_old_genome.add_id(old_genome)
	
	return rd.uniform_set_create([u_ubo, u_atomic, u_pot, u_dst_state, u_dst_sig, u_winner, u_new_genome, u_old_genome], shader_normalize, 0)

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
	
	var idx = (gy * 64 + gx) * 10 # 10 floats per cell
	if (idx + 8) * 4 >= last_analysis_bytes.size(): return {} 
	
	var floats = last_analysis_bytes.to_float32_array()
	var base = idx
	
	var m = floats[base]
	if m < 0.05: return {} # Empty
	
	var mu = floats[base+1]
	var sig = floats[base+2]
	var rad = floats[base+3]
	var flow = floats[base+4]
	var aff = floats[base+5]
	var den = floats[base+6] # Was lambda
	var sec = floats[base+7]
	var per = floats[base+8]
	
	var info = {
		"mu": mu, "sigma": sig,
		"radius": rad, "flow": flow,
		"affinity": aff, "density_tol": den,
		"secretion": sec, "perception": per,
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

# === INPUT HANDLING ===

func _input(event):
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_MIDDLE:
			is_dragging = event.pressed
			last_mouse_pos = event.position
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera_zoom = min(camera_zoom * 1.1, 20.0)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera_zoom = max(camera_zoom * 0.9, 0.1)
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			is_inspecting = event.pressed
			if not is_inspecting:
				# Clear highlight/tooltip when released
				set_highlight_genes({}, false)
				emit_signal("species_hovered", {})
			
	elif event is InputEventMouseMotion:
		if is_dragging:
			var viewport_size = get_viewport().get_visible_rect().size
			var delta = (event.position - last_mouse_pos) / viewport_size.y
			camera_pos -= delta / camera_zoom
			last_mouse_pos = event.position
		elif is_inspecting:
			# Hover Logic (Only if right-clicking)
			var viewport_size = get_viewport().get_visible_rect().size
			var aspect = viewport_size.x / viewport_size.y
			var uv = event.position / viewport_size
			
			# Match Shader Transform: Screen -> Texture
			uv -= Vector2(0.5, 0.5)
			
			if aspect > 1.0:
				uv.x *= aspect
			else:
				uv.y /= aspect
				
			uv /= camera_zoom
			uv += camera_pos
			uv += Vector2(0.5, 0.5)
			
			var info = get_species_info_at(uv)
			if info.has("id"):
				set_highlight_genes(info, true)
				emit_signal("species_hovered", info)
			else:
				set_highlight_genes({}, false)
				emit_signal("species_hovered", {})
