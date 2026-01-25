extends Node

@onready var sim = $LeniaSimulation
@onready var ui_container = $CanvasLayer/UI/Panel/Scroll/VBox
const GeneHistogram = preload("res://simulation/GeneHistogram.gd")
var histogram_display
var species_container
var tooltip_panel
var tooltip_label



# UI Schema: Group Name -> List of [ParamKey, Label, Min, Max, Step]
# New Parametric Lenia parameters
var ui_schema = {
	"Simulation": [
		["dt", "Time Step (dt)", 0.05, 1.0, 0.01],
		["init_density", "Initial Density", 0.1, 0.8, 0.01],
		["init_clusters", "Initial Clusters", 4.0, 64.0, 1.0]
	],
	"Kernel Geometry": [
		["R", "Kernel Base Radius (R)", 8.0, 30.0, 1.0]
	],
	"God Mode (Physics)": [
		["repulsion", "Repulsion Force", 0.0, 20.0, 0.5],
		["damage", "Combat Damage", 0.0, 10.0, 0.1],
		["identity_thr", "Identity Threshold", 0.0, 1.0, 0.01],
		["colonize_thr", "Colonization Thr (Defense)", 0.0, 0.5, 0.01]
	],
	"Chemical Signal": [
		["signal_diff", "Diffusion Rate", 0.0, 10.0, 0.1],
		["signal_decay", "Decay Rate", 0.0, 1.0, 0.01]
	],
	"Evolution": [
		["mutation_rate", "Mutation Rate", 0.0, 0.2, 0.005],
		["base_decay", "Base Decay", 0.0, 0.1, 0.005]
	],
	"Gene Pools (Init)": [
		["g_mu_min", "Mu Min (Archetype)", 0.0, 1.0, 0.05],
		["g_mu_max", "Mu Max", 0.0, 1.0, 0.05],
		["g_sigma_min", "Sigma Min (Stability)", 0.0, 1.0, 0.05],
		["g_sigma_max", "Sigma Max", 0.0, 1.0, 0.05],
		["g_radius_min", "Radius Min", 0.0, 1.0, 0.05],
		["g_radius_max", "Radius Max", 0.0, 1.0, 0.05],
		["g_flow_min", "Flow Min", 0.0, 1.0, 0.05],
		["g_flow_max", "Flow Max", 0.0, 1.0, 0.05],
		["g_secretion_min", "Secretion Min", 0.0, 1.0, 0.05],
		["g_secretion_max", "Secretion Max", 0.0, 1.0, 0.05],
		["g_perception_min", "Sensus Min", 0.0, 1.0, 0.05],
		["g_perception_max", "Sensus Max", 0.0, 1.0, 0.05]
	]
}

# Statistics labels
var stats_labels = {}

func _ready():
	_build_ui()
	sim.reset_simulation()
	sim.reset_simulation()
	sim.reset_simulation()
	sim.stats_updated.connect(_on_stats_updated)
	sim.species_hovered.connect(_on_species_hovered)

func _build_ui():
	# Stats Header
	var stats_header = Label.new()
	stats_header.text = "STATISTICS"
	stats_header.add_theme_color_override("font_color", Color(1.0, 0.8, 0.2))
	stats_header.add_theme_font_size_override("font_size", 14)
	stats_header.add_theme_font_size_override("font_size", 14)
	ui_container.add_child(stats_header)

	# Floating Tooltip (Child of CanvasLayer)
	tooltip_panel = PanelContainer.new()
	tooltip_panel.visible = false
	tooltip_panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var style = StyleBoxFlat.new()
	style.bg_color = Color(0, 0, 0, 0.7)
	style.set_corner_radius_all(4)
	style.content_margin_left = 8
	style.content_margin_right = 8
	style.content_margin_top = 4
	style.content_margin_bottom = 4
	tooltip_panel.add_theme_stylebox_override("panel", style)
	
	tooltip_label = Label.new()
	tooltip_label.add_theme_font_size_override("font_size", 11)
	tooltip_panel.add_child(tooltip_label)
	
	get_node("CanvasLayer").add_child(tooltip_panel)

	# Stats display
	var stats_box = VBoxContainer.new()
	stats_box.add_theme_constant_override("separation", 2)
	
	for stat_name in ["Total Mass", "Coverage", "Diversity"]:
		var lbl = Label.new()
		lbl.text = stat_name + ": ---"
		lbl.add_theme_font_size_override("font_size", 10)
		lbl.add_theme_color_override("font_color", Color(0.6, 0.8, 0.6))
		stats_labels[stat_name] = lbl
		stats_box.add_child(lbl)
	
	ui_container.add_child(stats_box)
	ui_container.add_child(HSeparator.new())
	
	# Parameter sliders
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
			var initial_val = sim.get_parameter(key)
			val_label.text = str(initial_val).pad_decimals(3)
			val_label.name = "Val_" + key
			val_label.add_theme_font_size_override("font_size", 10)
			val_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
			
			lbl_hbox.add_child(label)
			lbl_hbox.add_child(val_label)
			
			var slider = HSlider.new()
			slider.min_value = min_v
			slider.max_value = max_v
			slider.step = step_v
			slider.value = initial_val
			slider.size_flags_vertical = Control.SIZE_EXPAND_FILL
			
			# Connect using the new API
			slider.value_changed.connect(func(v): 
				sim.set_parameter(key, v)
				val_label.text = str(v).pad_decimals(3)
			)
			
			container.add_child(lbl_hbox)
			container.add_child(slider)
			ui_container.add_child(container)
		
		ui_container.add_child(HSeparator.new())

	# Buttons
	var btn_row = HBoxContainer.new()
	ui_container.add_child(btn_row)
	
	var btn_pause = Button.new()
	btn_pause.text = "PAUSE"
	btn_pause.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_pause.pressed.connect(func(): 
		sim.paused = !sim.paused
		btn_pause.text = "RESUME" if sim.paused else "PAUSE"
	)
	btn_row.add_child(btn_pause)

	var btn_reset = Button.new()
	btn_reset.text = "RESET"
	btn_reset.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_reset.pressed.connect(func(): sim.reset_simulation())
	btn_row.add_child(btn_reset)
	
	var btn_clear = Button.new()
	btn_clear.text = "CLEAR"
	btn_clear.pressed.connect(func(): sim.clear_simulation())
	ui_container.add_child(btn_clear)

	# Gene Histogram
	ui_container.add_child(HSeparator.new())
	var hist_header = Label.new()
	hist_header.text = "GENE HISTOGRAMS"
	hist_header.add_theme_color_override("font_color", Color(1.0, 0.8, 0.2))
	hist_header.add_theme_font_size_override("font_size", 14)
	ui_container.add_child(hist_header)
	
	histogram_display = GeneHistogram.new()
	ui_container.add_child(histogram_display)

func _on_stats_updated(total_mass, population, histograms):
	# Update Labels
	if stats_labels.has("Total Mass"):
		stats_labels["Total Mass"].text = "Total Mass: " + str(int(total_mass))
		
	if stats_labels.has("Coverage"):
		# Population count / Total Pixels (1024*1024 = 1048576)
		var coverage = (float(population) / 1048576.0) * 100.0
		stats_labels["Coverage"].text = "Coverage: " + "%.2f" % coverage + "%"
		
	if stats_labels.has("Diversity"):
		stats_labels["Diversity"].text = "Diversity: (Calc...)"
		
	# Update Histogram
	if histogram_display:
		histogram_display.update_histograms(histograms)


		
func _on_species_hovered(info):
	if tooltip_panel:
		if info.is_empty():
			tooltip_panel.visible = false
		else:
			tooltip_panel.visible = true
			var txt = "Species #%s: %s\n" % [str(info.get("id", "?")), info.get("name", "Unknown")]
			txt += "Mass: %d\n" % int(info.get("mass", 0) * 1000)
			txt += "Archetype: %.2f\n" % info.get("mu", 0.0)
			txt += "Stability: %.2f\n" % info.get("sigma", 0.0)
			txt += "Cohesion: %.2f\n" % info.get("affinity", 0.0)
			txt += "Mobility: %.2f\n" % info.get("flow", 0.0)
			txt += "Secretion: %.2f\n" % info.get("secretion", 0.0)
			txt += "Sensus: %.2f" % info.get("perception", 0.0)
			
			if tooltip_label:
				tooltip_label.text = txt

	# Move tooltip to mouse
	if tooltip_panel.visible:
		var mpos = get_viewport().get_mouse_position()
		tooltip_panel.position = mpos + Vector2(16, 16)

func _process(_delta):
	# Update tooltip pos if visible
	if tooltip_panel and tooltip_panel.visible:
		var mpos = get_viewport().get_mouse_position()
		tooltip_panel.position = mpos + Vector2(16, 16)
