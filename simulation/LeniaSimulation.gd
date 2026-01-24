extends Node

signal stats_updated(total_mass, population, histograms)
signal species_list_updated(species_list)
signal species_hovered(info) # New signal
const SpeciesTracker = preload("res://simulation/SpeciesTracker.gd")
var tracker = SpeciesTracker.new()


# === GLOBAL SIMULATION PARAMETERS ===
var params = {
	"res_x": 1024.0, 
	"res_y": 1024.0,
	"dt": 0.25,
	"seed": 0.0,
	# Kernel shape (global - creates pattern types)
	"R": 12.0,           # Kernel radius in pixels

	# Evolution
	"mutation_rate": 0.02,
	"base_decay": 0.015,
	# Initialization
	"init_clusters": 16.0,
	"init_density": 0.5,   # Higher density for better start
	
	# Advanced Physics / Combat
	"repulsion": 8.0,      # Repulsion Force
	"damage": 2.0,         # Combat Damage
	"identity_thr": 0.2,   # Difference to be considered enemy
	"colonize_thr": 0.15,  # Mass needed to resist invasion
	
	# Gene Ranges (Min, Max) [0.0, 1.0]
	"g_mu_min": 0.0, "g_mu_max": 1.0,
	"g_sigma_min": 0.0, "g_sigma_max": 1.0, 
	"g_radius_min": 0.0, "g_radius_max": 1.0,
	"g_flow_min": 0.0, "g_flow_max": 1.0,
	"g_affinity_min": 0.0, "g_affinity_max": 1.0,
	"g_lambda_min": 0.0, "g_lambda_max": 1.0
}

# === RENDERING DEVICE RESOURCES ===
var rd: RenderingDevice
var shader_init: RID
var shader_conv: RID
var shader_flow: RID
var shader_stats: RID
var pipeline_init: RID
var pipeline_conv: RID
var pipeline_flow: RID
var pipeline_stats: RID
var pipeline_analysis: RID
var pipeline_flow_conservative: RID
var pipeline_normalize: RID
var shader_analysis: RID
var shader_flow_conservative: RID
var shader_normalize: RID

# Textures: State (mass, vel, age) and Genome (6 genes)
var tex_state_a: RID
var tex_state_b: RID
var tex_genome_a: RID
var tex_genome_b: RID
var tex_potential: RID
var tex_mass_accum: RID

# Bridges for display
var texture_rd_state: Texture2DRD
var texture_rd_genome: Texture2DRD

var ubo: RID
var stats_buffer: RID
var analysis_buffer: RID
var sampler: RID
var ping_pong := false
var initialized := false
var paused := false # Pause state

# Uniform Set Cache
var set_cache = {}

var stats_frame_count := 0
var last_analysis_bytes: PackedByteArray
var last_species_list = []

# Camera state
var camera_pos := Vector2(0.0, 0.0)
var camera_zoom := 1.0
var is_dragging := false
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
	print("Parametric Lenia initialized: ", int(params["res_x"]), "x", int(params["res_y"]))

func _process(_delta):
	if not initialized: return
	
	# Update random seed
	params["seed"] = randf() * 1000.0
	
	if not paused:
		# Update UBO
		_update_ubo()
		
		# Dispatch Simulation Step
		_dispatch_step()
	
	# Statistics Readback (every 10 frames)
	stats_frame_count += 1
	if stats_frame_count >= 10:
		stats_frame_count = 0
		var bytes = rd.buffer_get_data(stats_buffer)
		# Parse: total_mass (uint), population (uint), histograms (60 uints)
		# 62 uints * 4 = 248 bytes
		if bytes.size() >= 248:
			var data = bytes.to_int32_array() # Note: Godot uses signed int array for convenience
			# Interpret as unsigned where needed
			var total_mass = float(data[0]) / 1000.0 # Unscale
			var population = data[1]
			var histograms = []
			# Copy histograms (indices 2 to 61)
			for i in range(2, 62):
				histograms.append(data[i])
			emit_signal("stats_updated", total_mass, population, histograms)
	
	# Species Analysis (every 30 frames ~ 0.5s)
	if stats_frame_count % 30 == 0:
		var bytes = rd.buffer_get_data(analysis_buffer)
		if bytes.size() >= 131072:
			last_analysis_bytes = bytes
			# Run heavy task in a thread? For now main thread is fine for 64x64 grid
			var species_list = tracker.find_species(bytes)
			last_species_list = species_list
			emit_signal("species_list_updated", species_list)
	
	# Update Display Material
	if display_material:
		var current_state = tex_state_b if ping_pong else tex_state_a
		var current_genome = tex_genome_b if ping_pong else tex_genome_a
		
		if texture_rd_state.texture_rd_rid != current_state:
			texture_rd_state.texture_rd_rid = current_state
			
		if texture_rd_genome.texture_rd_rid != current_genome:
			texture_rd_genome.texture_rd_rid = current_genome
			
		display_material.set_shader_parameter("camera_pos", camera_pos)
		display_material.set_shader_parameter("camera_zoom", camera_zoom)

func _update_ubo():
	# UBO layout: 28 floats
	var buffer = PackedFloat32Array([
		params["res_x"], params["res_y"],      # vec2 u_res
		params["dt"], params["seed"],          # float u_dt, u_seed
		# Kernel parameters
		params["R"],                           # float u_R
		params["repulsion"], params["damage"], params["identity_thr"], # Replaces unused w1,w2,w3
		# Evolution
		params["mutation_rate"],               # float u_mutation_rate
		params["base_decay"],                  # float u_base_decay
		# Initialization  
		params["init_clusters"],               # float u_init_clusters
		params["init_density"],                # float u_init_density
		params["colonize_thr"],                # _pad0 usage: Colonization Threshold
		0.0,                                   # Padding for std430 vec2 alignment (Offset 52->56)
		# Gene Ranges (12 floats)
		params["g_mu_min"], params["g_mu_max"],
		params["g_sigma_min"], params["g_sigma_max"],
		params["g_radius_min"], params["g_radius_max"],
		params["g_flow_min"], params["g_flow_max"],
		params["g_affinity_min"], params["g_affinity_max"],
		params["g_lambda_min"], params["g_lambda_max"],
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0           # padding to 32 scans
	])
	var bytes = buffer.to_byte_array()
	rd.buffer_update(ubo, 0, bytes.size(), bytes)

func _dispatch_step():
	var src_state = tex_state_b if ping_pong else tex_state_a
	var dst_state = tex_state_a if ping_pong else tex_state_b
	var src_genome = tex_genome_b if ping_pong else tex_genome_a
	var dst_genome = tex_genome_a if ping_pong else tex_genome_b
	
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	# 1. Convolution Pass: Calculate potential from state using genome
	var set_conv = _create_set_conv(src_state, src_genome, tex_potential)
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_conv)
	rd.compute_list_bind_uniform_set(compute_list, set_conv, 0)
	rd.compute_list_dispatch(compute_list, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	rd.compute_list_end()
	
	# 2. Flow Pass (Conservative: Atomic Push)
	var cache_key_flow = "flow_" + str(ping_pong)
	var set_flow_con: RID
	if set_cache.has(cache_key_flow) and set_cache[cache_key_flow].is_valid():
		set_flow_con = set_cache[cache_key_flow]
	else:
		set_flow_con = _create_set_flow_conservative(src_state, src_genome, tex_potential, dst_state, dst_genome)
		set_cache[cache_key_flow] = set_flow_con
	
	var compute_list_flow = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_flow, pipeline_flow_conservative)
	rd.compute_list_bind_uniform_set(compute_list_flow, set_flow_con, 0)
	rd.compute_list_dispatch(compute_list_flow, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# Memory Barrier
	rd.barrier(RenderingDevice.BARRIER_MASK_COMPUTE)
	
	# 3. Normalize Pass (Atomic -> Float + Growth)
	var cache_key_norm = "norm_" + str(ping_pong)
	var set_norm: RID
	if set_cache.has(cache_key_norm) and set_cache[cache_key_norm].is_valid():
		set_norm = set_cache[cache_key_norm]
	else:
		set_norm = _create_set_normalize(tex_potential, dst_genome, dst_state)
		set_cache[cache_key_norm] = set_norm
	
	var compute_list_norm = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_norm, pipeline_normalize)
	rd.compute_list_bind_uniform_set(compute_list_norm, set_norm, 0)
	rd.compute_list_dispatch(compute_list_norm, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# === STATS DISPATCH ===
	# Clear stats buffer first (using new byte array of zeros)
	var clear_bytes = PackedByteArray()
	clear_bytes.resize(248) # Zeros
	rd.buffer_update(stats_buffer, 0, 248, clear_bytes)
	
	# We want stats of the NEW state (dst_state)
	var set_stats = _create_set_stats(dst_state, dst_genome)
	
	var compute_list_stats = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_stats, pipeline_stats)
	rd.compute_list_bind_uniform_set(compute_list_stats, set_stats, 0)
	rd.compute_list_dispatch(compute_list_stats, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# === ANALYSIS DISPATCH ===
	# 64x64 threads = 1 global group if local_size=8x8 and we dispatch 8x8 groups.
	# Shader uses grid index. Total threads needed: 64x64.
	# Local size 8x8. Groups needed: 64/8 = 8.
	var set_analysis = _create_set_analysis(dst_state, dst_genome)
	
	var compute_list_analysis = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list_analysis, pipeline_analysis)
	rd.compute_list_bind_uniform_set(compute_list_analysis, set_analysis, 0)
	rd.compute_list_dispatch(compute_list_analysis, 8, 8, 1) # 8*8 * 8*8 = 64*64 threads
	rd.compute_list_end()
	
	ping_pong = !ping_pong

func _dispatch_init():
	var set_init = _create_set_init(tex_state_a, tex_genome_a)
	var wg_x = int(ceil(params["res_x"] / 8.0))
	var wg_y = int(ceil(params["res_y"] / 8.0))
	
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_init)
	rd.compute_list_bind_uniform_set(compute_list, set_init, 0)
	rd.compute_list_dispatch(compute_list, wg_x, wg_y, 1)
	rd.compute_list_end()
	
	# Also clear buffer B
	rd.texture_clear(tex_state_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_genome_b, Color(0,0,0,0), 0, 1, 0, 1)
	
	ping_pong = false

func _create_uniforms():
	# UBO: 32 floats = 128 bytes (aligned to 16 bytes)
	var buffer = PackedFloat32Array()
	buffer.resize(32)
	var bytes = buffer.to_byte_array()
	ubo = rd.storage_buffer_create(bytes.size(), bytes)
	
	# Stats Buffer: 62 uints = 248 bytes
	var stats_bytes = PackedByteArray()
	stats_bytes.resize(248)
	stats_buffer = rd.storage_buffer_create(248, stats_bytes)
	
	# Analysis Buffer: 4096 cells * 32 bytes = 131072 bytes
	var analysis_bytes = PackedByteArray()
	analysis_bytes.resize(131072)
	analysis_buffer = rd.storage_buffer_create(131072, analysis_bytes)

func _create_sampler():
	var sampler_state = RDSamplerState.new()
	sampler_state.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler = rd.sampler_create(sampler_state)

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
	
	# Atomic Mass Accumulation (R32_UINT)
	var fmt_atomic = RDTextureFormat.new()
	fmt_atomic.width = int(params["res_x"])
	fmt_atomic.height = int(params["res_y"])
	fmt_atomic.format = RenderingDevice.DATA_FORMAT_R32_UINT
	fmt_atomic.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT | 
		RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	tex_mass_accum = rd.texture_create(fmt_atomic, RDTextureView.new())
	
	# Create Texture2DRD bridges for display
	texture_rd_state = Texture2DRD.new()
	texture_rd_genome = Texture2DRD.new()
	texture_rd_state.texture_rd_rid = tex_state_a
	texture_rd_genome.texture_rd_rid = tex_genome_a
	
	if display_material:
		display_material.set_shader_parameter("tex_state", texture_rd_state)
		display_material.set_shader_parameter("tex_genome", texture_rd_genome)

func _compile_shaders():
	shader_init = _load_shader("res://simulation/shaders/compute_init.glsl")
	pipeline_init = rd.compute_pipeline_create(shader_init)
	
	shader_conv = _load_shader("res://simulation/shaders/compute_convolution.glsl")
	pipeline_conv = rd.compute_pipeline_create(shader_conv)
	
	shader_flow = _load_shader("res://simulation/shaders/compute_flow.glsl")
	pipeline_flow = rd.compute_pipeline_create(shader_flow)

	shader_stats = _load_shader("res://simulation/shaders/compute_stats.glsl")
	pipeline_stats = rd.compute_pipeline_create(shader_stats)

	shader_analysis = _load_shader("res://simulation/shaders/compute_analysis.glsl")
	pipeline_analysis = rd.compute_pipeline_create(shader_analysis)
	
	shader_flow_conservative = _load_shader("res://simulation/shaders/compute_flow_conservative.glsl")
	pipeline_flow_conservative = rd.compute_pipeline_create(shader_flow_conservative)
	
	shader_normalize = _load_shader("res://simulation/shaders/compute_normalize.glsl")
	pipeline_normalize = rd.compute_pipeline_create(shader_normalize)

func _load_shader(path: String) -> RID:
	# Try loading as a Resource first (Best practice in Godot 4)
	if FileAccess.file_exists(path + ".import"):
		var shader_res = load(path)
		if shader_res and shader_res is RDShaderFile:
			var spirv = shader_res.get_spirv()
			return rd.shader_create_from_spirv(spirv)
	
	# Fallback: Manual compilation
	var file = FileAccess.open(path, FileAccess.READ)
	var code = file.get_as_text()
	
	# Strip #[compute] directive if present
	if code.begins_with("#[compute]"):
		code = code.replace("#[compute]", "")
		
	var src = RDShaderSource.new()
	src.source_compute = code
	var spirv = rd.shader_compile_spirv_from_source(src)
	if spirv.compile_error_compute != "":
		push_error("Shader Compile Error: " + path + "\n" + spirv.compile_error_compute)
		return RID()
	return rd.shader_create_from_spirv(spirv)

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

func _create_set_conv(src_state: RID, src_genome: RID, dst_potential: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_state = RDUniform.new()
	u_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_state.binding = 1
	u_state.add_id(sampler)
	u_state.add_id(src_state)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome.binding = 2
	u_genome.add_id(sampler)
	u_genome.add_id(src_genome)
	
	var u_potential = RDUniform.new()
	u_potential.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_potential.binding = 3
	u_potential.add_id(dst_potential)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_potential], shader_conv, 0)

func _create_set_flow(src_state: RID, src_genome: RID, src_potential: RID, 
					  dst_state: RID, dst_genome: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_src_state = RDUniform.new()
	u_src_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_state.binding = 1
	u_src_state.add_id(sampler)
	u_src_state.add_id(src_state)
	
	var u_src_genome = RDUniform.new()
	u_src_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_genome.binding = 2
	u_src_genome.add_id(sampler)
	u_src_genome.add_id(src_genome)
	
	var u_src_potential = RDUniform.new()
	u_src_potential.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_potential.binding = 3
	u_src_potential.add_id(sampler)
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
	u_state.add_id(sampler)
	u_state.add_id(tex_state)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome.binding = 2
	u_genome.add_id(sampler)
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
	u_state.add_id(sampler)
	u_state.add_id(tex_state)
	
	var u_genome = RDUniform.new()
	u_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_genome.binding = 2
	u_genome.add_id(sampler)
	u_genome.add_id(tex_genome)
	
	var u_analysis = RDUniform.new()
	u_analysis.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_analysis.binding = 3
	u_analysis.add_id(analysis_buffer)
	
	return rd.uniform_set_create([u_ubo, u_state, u_genome, u_analysis], shader_analysis, 0)

func _create_set_flow_conservative(src_state: RID, src_genome: RID, tex_pot: RID, dst_state: RID, dst_genome: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_src_state = RDUniform.new()
	u_src_state.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_state.binding = 1
	u_src_state.add_id(sampler)
	u_src_state.add_id(src_state)
	
	var u_src_genome = RDUniform.new()
	u_src_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_src_genome.binding = 2
	u_src_genome.add_id(sampler)
	u_src_genome.add_id(src_genome)
	
	var u_pot = RDUniform.new()
	u_pot.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_pot.binding = 3
	u_pot.add_id(sampler)
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
	
	return rd.uniform_set_create([u_ubo, u_src_state, u_src_genome, u_pot, u_atomic, u_dst_state, u_dst_genome], shader_flow_conservative, 0)

func _create_set_normalize(tex_pot: RID, dst_genome: RID, dst_state: RID) -> RID:
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
	u_pot.add_id(sampler)
	u_pot.add_id(tex_pot)
	
	var u_new_genome = RDUniform.new()
	u_new_genome.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
	u_new_genome.binding = 3
	u_new_genome.add_id(sampler)
	u_new_genome.add_id(dst_genome)
	
	var u_dst_state = RDUniform.new()
	u_dst_state.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst_state.binding = 4
	u_dst_state.add_id(dst_state)
	
	return rd.uniform_set_create([u_ubo, u_atomic, u_pot, u_new_genome, u_dst_state], shader_normalize, 0)

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

func set_parameter(name: String, value: float):
	if params.has(name):
		params[name] = value

func get_parameter(name: String) -> float:
	return params.get(name, 0.0)

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
	
	var idx = (gy * 64 + gx) * 8 # 8 floats per cell
	if idx + 7 >= last_analysis_bytes.size() / 4: return {} # Float32 array index check
	
	# Read from bytes manually (slower than Array access but fine for 1 lookup)
	# Actually bytes.to_float32_array() creates copy. Slow if done every frame?
	# Better to cache FloatArray?
	# Let's decode just the chunk we need using stream buffer wrapper or just to_float32_array ONCE in _process.
	# But that takes memory? 131kb is small.
	# I'll optimize later. For now, create array from cached bytes.
	var floats = last_analysis_bytes.to_float32_array()
	var base = idx
	
	var m = floats[base]
	if m < 0.05: return {} # Empty
	
	var mu = floats[base+1]
	var sig = floats[base+2]
	var rad = floats[base+3]
	var flow = floats[base+4]
	var aff = floats[base+5]
	var lam = floats[base+6]
	
	var info = {
		"mu": mu, "sigma": sig,
		"radius": rad, "flow": flow,
		"affinity": aff, "lambda": lam,
		"mass": m
	}
	
	# Find matching species in last_species_list
	for s in last_species_list:
		# Use genes logic
		if abs(s.genes["mu"] - mu) + abs(s.genes["sigma"] - sig) < 0.15:
			info["id"] = s.id
			info["name"] = s.name
			info["color"] = s.color
			return info
			
	info["id"] = "?"
	info["name"] = "Unknown"
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
			
	elif event is InputEventMouseMotion:
		if is_dragging:
			var viewport_size = get_viewport().get_visible_rect().size
			var delta = (event.position - last_mouse_pos) / viewport_size.y
			camera_pos -= delta / camera_zoom
			last_mouse_pos = event.position
		else:
			# Hover Logic
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
