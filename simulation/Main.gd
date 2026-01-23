extends Node

@onready var sim = $LeniaSimulation
@onready var ui_container = $CanvasLayer/UI/Panel/Scroll/VBox

# UI Schema: Group Name -> List of [ParamKey, Label, Min, Max, Step]
var ui_schema = {
	"Simulation": [
		["dt", "Time Step (dt)", 0.05, 0.6, 0.01],
		["density", "Initial Density", 0.0, 1.0, 0.01],
		["show_waste", "Show Waste", 0.0, 1.0, 1.0]
	],
	"Life Rules": [
		["g_mu_base", "Mu Base", 0.05, 0.3, 0.001],
		["g_mu_range", "Mu Range", 0.0, 0.6, 0.01],
		["g_sigma", "Sigma Base", 0.01, 0.1, 0.001]
	],
	"Physics Kernel": [
		["k_w1", "Kernel W1 (Inner)", 0.1, 1.5, 0.01],
		["k_w2", "Kernel W2 (Mid)", 0.1, 1.5, 0.01],
		["k_w3", "Kernel W3 (Outer)", 0.1, 1.5, 0.01],
		["R", "Kernel Radius", 4.0, 20.0, 1.0]
	],
	"Fluid Dynamics": [
		["force_flow", "Force Flow", 0.0, 2.0, 0.05],
		["force_rep", "Force Repulsion", 0.0, 4.0, 0.1],
		["chemotaxis", "Chemotaxis", 0.0, 2.0, 0.05],
		["inertia", "Inertia", 0.0, 1.0, 0.01]
	],
	"Metabolism": [
		["mutation", "Mutation Rate", 0.0, 0.1, 0.001],
		["eat_rate", "Eat Rate", 0.0, 1.0, 0.01],
		["decay", "Decay Rate", 0.001, 0.05, 0.001]
	]
}

func _ready():
	_build_ui()
	# Initial Update
	sim.reset_simulation()

func _build_ui():
	for group in ui_schema:
		var header = Label.new()
		header.text = group.to_upper()
		header.add_theme_color_override("font_color", Color(0, 0.9, 1.0))
		header.add_theme_font_size_override("font_size", 14)
		ui_container.add_child(header)
		
		for item in ui_schema[group]:
			var key = item[0]
			var lbl_text = item[1]
			var min_v = item[2]
			var max_v = item[3]
			var step_v = item[4]
			
			var container = VBoxContainer.new()
			container.add_theme_constant_override("separation", 0)
			
			var lbl_hbox = HBoxContainer.new()
			var label = Label.new()
			label.text = lbl_text
			label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			label.add_theme_font_size_override("font_size", 10)
			label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
			
			var val_label = Label.new()
			val_label.text = str(sim.params[key])
			val_label.name = "Val_" + key
			val_label.add_theme_font_size_override("font_size", 10)
			val_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
			
			lbl_hbox.add_child(label)
			lbl_hbox.add_child(val_label)
			
			var slider = HSlider.new()
			slider.min_value = min_v
			slider.max_value = max_v
			slider.step = step_v
			slider.value = sim.params[key]
			slider.size_flags_vertical = Control.SIZE_EXPAND_FILL
			
			# Connect
			slider.value_changed.connect(func(v): 
				sim.params[key] = v
				val_label.text = str(v).pad_decimals(3)
			)
			
			container.add_child(lbl_hbox)
			container.add_child(slider)
			ui_container.add_child(container)
		
		ui_container.add_child(HSeparator.new())

	# Buttons
	var btn_reset = Button.new()
	btn_reset.text = "RESET SIMULATION"
	btn_reset.pressed.connect(func(): sim.reset_simulation())
	ui_container.add_child(btn_reset)
	
	var btn_clear = Button.new()
	btn_clear.text = "CLEAR"
	btn_clear.pressed.connect(func(): sim.clear_simulation())
	ui_container.add_child(btn_clear)

func _process(delta):
	pass
