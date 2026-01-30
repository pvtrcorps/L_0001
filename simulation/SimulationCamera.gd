class_name SimulationCamera
extends Node

# Signals
signal camera_updated(pos: Vector2, zoom: float)
signal inspect_requested(uv: Vector2, active: bool)

# State
var camera_pos := Vector2(0.0, 0.0)
var camera_zoom := 1.0
var is_dragging := false
var is_inspecting := false
var last_mouse_pos := Vector2()

# Configuration
var zoom_speed := 1.1
var min_zoom := 0.1
var max_zoom := 20.0

func _input(event):
	var viewport = get_viewport()
	if not viewport: return
	var viewport_rect = viewport.get_visible_rect()
	var viewport_size = viewport_rect.size
	
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_MIDDLE:
			is_dragging = event.pressed
			last_mouse_pos = event.position
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			camera_zoom = min(camera_zoom * zoom_speed, max_zoom)
			emit_signal("camera_updated", camera_pos, camera_zoom)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			camera_zoom = max(camera_zoom / zoom_speed, min_zoom)
			emit_signal("camera_updated", camera_pos, camera_zoom)
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			is_inspecting = event.pressed
			if is_inspecting:
				# Trigger initial inspect
				_handle_inspect_input(event.position, viewport_size)
			else:
				emit_signal("inspect_requested", Vector2(), false)
			
	elif event is InputEventMouseMotion:
		if is_dragging:
			var delta = (event.position - last_mouse_pos) / viewport_size.y
			camera_pos -= delta / camera_zoom
			last_mouse_pos = event.position
			emit_signal("camera_updated", camera_pos, camera_zoom)
		elif is_inspecting:
			_handle_inspect_input(event.position, viewport_size)

func _handle_inspect_input(screen_pos: Vector2, viewport_size: Vector2):
	var aspect = viewport_size.x / viewport_size.y
	var uv = screen_pos / viewport_size
	
	# Match Shader Transform: Screen -> Texture
	uv -= Vector2(0.5, 0.5)
	
	if aspect > 1.0:
		uv.x *= aspect
	else:
		uv.y /= aspect
		
	uv /= camera_zoom
	uv += camera_pos
	uv += Vector2(0.5, 0.5)
	
	emit_signal("inspect_requested", uv, true)
