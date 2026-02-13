extends Control

signal range_changed(min_value: float, max_value: float)

@export var min_limit: float = 0.0:
	set(value):
		min_limit = value
		queue_redraw()

@export var max_limit: float = 1.0:
	set(value):
		max_limit = max(value, min_limit + 0.0001)
		queue_redraw()

@export var step: float = 0.01
@export var handle_radius: float = 7.0

var min_value: float = 0.25:
	set(value):
		min_value = _snap(clamp(value, min_limit, max_value))
		queue_redraw()

var max_value: float = 0.75:
	set(value):
		max_value = _snap(clamp(value, min_value, max_limit))
		queue_redraw()

var _dragging_min := false
var _dragging_max := false

func _ready():
	custom_minimum_size = Vector2(0, 24)
	mouse_filter = MOUSE_FILTER_STOP

func setup(initial_min: float, initial_max: float, limit_min: float, limit_max: float, step_value: float = 0.01) -> void:
	min_limit = limit_min
	max_limit = limit_max
	step = step_value
	min_value = clamp(initial_min, min_limit, max_limit)
	max_value = clamp(initial_max, min_value, max_limit)
	queue_redraw()

func _draw() -> void:
	var center_y := size.y * 0.5
	var left := handle_radius
	var right: float = max(left + 1.0, size.x - handle_radius)

	# Base track
	draw_line(Vector2(left, center_y), Vector2(right, center_y), Color(0.25, 0.25, 0.25), 4.0)

	# Active range
	var min_x := _value_to_x(min_value)
	var max_x := _value_to_x(max_value)
	draw_line(Vector2(min_x, center_y), Vector2(max_x, center_y), Color(0.2, 0.8, 1.0), 5.0)

	# Handles
	draw_circle(Vector2(min_x, center_y), handle_radius, Color(0.85, 0.85, 0.9))
	draw_circle(Vector2(max_x, center_y), handle_radius, Color(0.85, 0.85, 0.9))

func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		var mouse_event := event as InputEventMouseButton
		if mouse_event.pressed:
			var mouse_x := mouse_event.position.x
			var min_x := _value_to_x(min_value)
			var max_x := _value_to_x(max_value)
			if abs(mouse_x - min_x) <= abs(mouse_x - max_x):
				_dragging_min = true
			else:
				_dragging_max = true
			_update_drag(mouse_event.position.x)
		else:
			_dragging_min = false
			_dragging_max = false
	elif event is InputEventMouseMotion:
		if _dragging_min or _dragging_max:
			var mouse_event := event as InputEventMouseMotion
			_update_drag(mouse_event.position.x)

func _update_drag(mouse_x: float) -> void:
	var value := _x_to_value(mouse_x)
	if _dragging_min:
		var next_min := _snap(clamp(value, min_limit, max_value))
		if not is_equal_approx(next_min, min_value):
			min_value = next_min
			emit_signal("range_changed", min_value, max_value)
	elif _dragging_max:
		var next_max := _snap(clamp(value, min_value, max_limit))
		if not is_equal_approx(next_max, max_value):
			max_value = next_max
			emit_signal("range_changed", min_value, max_value)

func _value_to_x(value: float) -> float:
	if is_equal_approx(max_limit, min_limit):
		return handle_radius
	var t := (value - min_limit) / (max_limit - min_limit)
	return lerp(handle_radius, max(size.x - handle_radius, handle_radius + 1.0), t) as float

func _x_to_value(x: float) -> float:
	var left := handle_radius
	var right: float = max(left + 1.0, size.x - handle_radius)
	var t := inverse_lerp(left, right, clamp(x, left, right))
	return lerp(min_limit, max_limit, t)

func _snap(value: float) -> float:
	if step <= 0.0:
		return value
	return snappedf(value, step)
