extends Node

# Simulation Parameters
var params = {
	"res_x": 800.0, "res_y": 800.0,
	"mouse_x": 0.0, "mouse_y": 0.0,
	"dt": 0.25,
	"seed": 0.0,
	"density": 0.35,
	"init_grid": 16.0,
	"R": 8.0,
	"k_w1": 0.6, "k_w2": 1.0, "k_w3": 0.8,
	"g_mu_base": 0.12, "g_mu_range": 0.22, "g_sigma": 0.035,
	"force_flow": 1.0,
	"force_rep": 2.0,
	"decay": 0.012,
	"eat_rate": 0.6,
	"chemotaxis": 0.8,
	"mutation": 0.01,
	"inertia": 0.90,
	"brush_size": 40.0, "brush_hue": 0.0,
	"brush_mode": 1.0,
	"show_waste": 1.0,
	"mouse_click": 0.0    
}

# Resources
var rd: RenderingDevice
var shader_init: RID
var shader_conv: RID
var shader_flow: RID
var pipeline_init: RID
var pipeline_conv: RID
var pipeline_flow: RID

var tex_living_a: RID
var tex_living_b: RID
var tex_waste_a: RID
var tex_waste_b: RID
var tex_potential: RID

# Bridges for display
var texture_rd_living: Texture2DRD
var texture_rd_waste: Texture2DRD

var ubo: RID
var ping_pong := false
var initialized := false

# Camera state
var camera_pos := Vector2(0.0, 0.0)
var camera_zoom := 1.0
var is_dragging := false
var last_mouse_pos := Vector2()

@export var display_material: ShaderMaterial

func _ready():
	# Use global RD if possible, or create local for compute if needed. 
	# For display sync, global (get_rendering_device) is preferred in Godot 4.x
	rd = RenderingServer.get_rendering_device()
	
	if not rd:
		print("Compute shaders not supported (No RenderingDevice).")
		return

	_compile_shaders()
	_create_textures()
	_create_uniforms()
	
	# Run Init once
	_dispatch_init()
	initialized = true

func _process(delta):
	if not initialized: return
	
	# Update inputs (mouse, seed drift if needed)
	params["seed"] = randf() * 100.0
	
	# Update UBO
	_update_ubo()
	
	# Dispatch Simulation Steps
	_dispatch_step()
	
	# Update Display Material
	if display_material:
		# We need to update the Texture2DRD RIDs if they change, but we swap meaning of A/B
		# Actually Texture2DRD holds one RID. We should update the RID it points to? 
		# API says: texture_rd.texture_rd_rid = new_rid
		
		var current_living = tex_living_b if ping_pong else tex_living_a
		var current_waste = tex_waste_b if ping_pong else tex_waste_a
		
		if texture_rd_living.texture_rd_rid != current_living:
			texture_rd_living.texture_rd_rid = current_living
			
		if texture_rd_waste.texture_rd_rid != current_waste:
			texture_rd_waste.texture_rd_rid = current_waste
			
		display_material.set_shader_parameter("show_waste", params["show_waste"])
		display_material.set_shader_parameter("camera_pos", camera_pos)
		display_material.set_shader_parameter("camera_zoom", camera_zoom)

func _update_ubo():
	var buffer = PackedFloat32Array([
		params.res_x, params.res_y,
		params.mouse_x, params.mouse_y,
		params.dt, params.seed, params.density, params.init_grid,
		params.R, params.k_w1, params.k_w2, params.k_w3,
		params.g_mu_base, params.g_mu_range, params.g_sigma, params.force_flow,
		params.force_rep, params.decay, params.eat_rate, params.chemotaxis,
		params.mutation, params.inertia, params.brush_size, params.brush_hue,
		params.brush_mode, params.show_waste, params.mouse_click, 0.0
	])
	var bytes = buffer.to_byte_array()
	rd.buffer_update(ubo, 0, bytes.size(), bytes)

func _dispatch_step():
	var uniform_sets = []
	
	var src_living = tex_living_b if ping_pong else tex_living_a
	var dst_living = tex_living_a if ping_pong else tex_living_b
	var src_waste = tex_waste_b if ping_pong else tex_waste_a
	var dst_waste = tex_waste_a if ping_pong else tex_waste_b
	
	# 1. Convolution
	var set_conv = _create_set_conv(src_living, tex_potential)
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_conv)
	rd.compute_list_bind_uniform_set(compute_list, set_conv, 0)
	rd.compute_list_dispatch(compute_list, int(params.res_x)/8, int(params.res_y)/8, 1)
	rd.compute_list_end()
	
	# 2. Flow
	var set_flow = _create_set_flow(src_living, src_waste, tex_potential, dst_living, dst_waste)
	
	compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_flow)
	rd.compute_list_bind_uniform_set(compute_list, set_flow, 0)
	rd.compute_list_dispatch(compute_list, int(params.res_x)/8, int(params.res_y)/8, 1)
	rd.compute_list_end()
	
	ping_pong = !ping_pong

func _dispatch_init():
	var set_init = _create_set_init(tex_living_a, tex_waste_a)
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline_init)
	rd.compute_list_bind_uniform_set(compute_list, set_init, 0)
	rd.compute_list_dispatch(compute_list, int(params.res_x)/8, int(params.res_y)/8, 1)
	rd.compute_list_end()


func _create_uniforms():
	# UBO
	var buffer = PackedFloat32Array([0.0]) # Placeholder size
	buffer.resize(28) 
	var bytes = buffer.to_byte_array()
	ubo = rd.storage_buffer_create(bytes.size(), bytes)

func _create_textures():
	var fmt = RDTextureFormat.new()
	fmt.width = int(params.res_x)
	fmt.height = int(params.res_y)
	fmt.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT | RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	
	tex_living_a = rd.texture_create(fmt, RDTextureView.new())
	tex_living_b = rd.texture_create(fmt, RDTextureView.new())
	tex_waste_a = rd.texture_create(fmt, RDTextureView.new())
	tex_waste_b = rd.texture_create(fmt, RDTextureView.new())
	tex_potential = rd.texture_create(fmt, RDTextureView.new())
	
	# Create Bridges
	texture_rd_living = Texture2DRD.new()
	texture_rd_waste = Texture2DRD.new()
	texture_rd_living.texture_rd_rid = tex_living_a
	texture_rd_waste.texture_rd_rid = tex_waste_a
	
	if display_material:
		display_material.set_shader_parameter("tex_living", texture_rd_living)
		display_material.set_shader_parameter("tex_waste", texture_rd_waste)

func _compile_shaders():
	shader_init = _load_shader("res://simulation/shaders/compute_init.glsl")
	pipeline_init = rd.compute_pipeline_create(shader_init)
	
	shader_conv = _load_shader("res://simulation/shaders/compute_convolution.glsl")
	pipeline_conv = rd.compute_pipeline_create(shader_conv)
	
	shader_flow = _load_shader("res://simulation/shaders/compute_flow.glsl")
	pipeline_flow = rd.compute_pipeline_create(shader_flow)

func _load_shader(path: String) -> RID:
	# Try loading as a Resource first (Best practice in Godot 4)
	if FileAccess.file_exists(path + ".import"): # Check if imported
		var shader_res = load(path)
		if shader_res and shader_res is RDShaderFile:
			var spirv = shader_res.get_spirv()
			return rd.shader_create_from_spirv(spirv)
	
	# Fallback: Manual compilation (Useful if files are new or not re-imported yet)
	var file = FileAccess.open(path, FileAccess.READ)
	var code = file.get_as_text()
	
	# Strip #[compute] directive if present, as it confuses the raw GLSL compiler
	if code.begins_with("#[compute]"):
		code = code.replace("#[compute]", "")
		
	var src = RDShaderSource.new()
	src.source_compute = code
	var spirv = rd.shader_compile_spirv_from_source(src)
	if spirv.compile_error_compute != "":
		push_error("Shader Compile Error: " + path + "\n" + spirv.compile_error_compute)
		return RID() # Return empty RID on error
	return rd.shader_create_from_spirv(spirv)

# Helpers for Uniform Sets (Binding boilerplate)
func _create_set_init(dst_liv: RID, dst_waste: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_liv = RDUniform.new()
	u_liv.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_liv.binding = 1
	u_liv.add_id(dst_liv)
	
	var u_waste = RDUniform.new()
	u_waste.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_waste.binding = 2
	u_waste.add_id(dst_waste)
	
	return rd.uniform_set_create([u_ubo, u_liv, u_waste], shader_init, 0)

func _create_set_conv(src_liv: RID, dst_pot: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	var u_src = RDUniform.new()
	u_src.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE # Sampler + Texture
	u_src.binding = 1
	# We need a sampler state!
	# Important: In Godot RD, standard samplers are usually created via sampler_create
	var sampler_state = RDSamplerState.new()
	sampler_state.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	var sampler = rd.sampler_create(sampler_state)
	
	u_src.add_id(sampler)
	u_src.add_id(src_liv)
	
	var u_dst = RDUniform.new()
	u_dst.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u_dst.binding = 2
	u_dst.add_id(dst_pot)
	
	return rd.uniform_set_create([u_ubo, u_src, u_dst], shader_conv, 0)

func _create_set_flow(src_l: RID, src_w: RID, src_p: RID, dst_l: RID, dst_w: RID) -> RID:
	var u_ubo = RDUniform.new()
	u_ubo.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u_ubo.binding = 0
	u_ubo.add_id(ubo)
	
	# Samplers
	var sampler_state = RDSamplerState.new()
	sampler_state.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	var sampler = rd.sampler_create(sampler_state)

	var u_sl = RDUniform.new(); u_sl.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE; u_sl.binding = 1; u_sl.add_id(sampler); u_sl.add_id(src_l)
	var u_sw = RDUniform.new(); u_sw.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE; u_sw.binding = 2; u_sw.add_id(sampler); u_sw.add_id(src_w)
	var u_sp = RDUniform.new(); u_sp.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE; u_sp.binding = 3; u_sp.add_id(sampler); u_sp.add_id(src_p)
	
	var u_dl = RDUniform.new(); u_dl.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE; u_dl.binding = 4; u_dl.add_id(dst_l)
	var u_dw = RDUniform.new(); u_dw.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE; u_dw.binding = 5; u_dw.add_id(dst_w)

	return rd.uniform_set_create([u_ubo, u_sl, u_sw, u_sp, u_dl, u_dw], shader_flow, 0)

func reset_simulation():
	_dispatch_init()
	params["seed"] = randf() * 100.0

func clear_simulation():
	rd.texture_clear(tex_living_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_living_b, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_waste_a, Color(0,0,0,0), 0, 1, 0, 1)
	rd.texture_clear(tex_waste_b, Color(0,0,0,0), 0, 1, 0, 1)

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
