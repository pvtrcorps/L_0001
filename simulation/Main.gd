extends Node

@onready var sim = $LeniaSimulation
@onready var ui_container = $CanvasLayer/UI/Panel/Scroll/VBox
const GeneHistogram = preload("res://simulation/GeneHistogram.gd")
const GeneRangeSlider = preload("res://simulation/GeneRangeSlider.gd")
var histogram_display
var species_container
var tooltip_panel
var tooltip_label
var deck_bar
var biomass_label
var deploy_hint_label
var deck_buttons = []
var selected_slot := 0


# UI Schema: Group Name -> List of [ParamKey, Label, Min, Max, Step]
# New Parametric Lenia parameters
var ui_schema = {
	"Simulation": [
		["dt", "Time Step (dt)", 0.01, 1.0, 0.01],
		["init_density", "Initial Density", 0.0, 1.0, 0.01],
		["init_clusters", "Initial Clusters", 1.0, 64.0, 1.0]
	],
	"Kernel Geometry": [
		["R", "Kernel Base Radius (R)", 8.0, 16.0, 1.0]
	],
	"Flow Physics": [
		["temperature", "Temperature (s)", 0.0, 3.0, 0.05], # Advection diffusion (s). Paper default: 0.65
		["theta_A", "Global Density Mult", 0.1, 10.0, 0.1], # Global Density Multiplier. Canonical 1.0
		["alpha_n", "Repulsion Sharpness (n)", 0.0, 4.0, 0.1], # Repulsion Sharpness. Canonical 2.0
		["beta_selection", "Selection Pressure (β)", 0.0, 3.0, 0.1],
		["interaction_beta", "Kernel Interaction (β)", 0.0, 10.0, 0.1],
		["genetic_barrier", "Genetic Barrier (Inmiscibility)", 0.0, 1.0, 0.05],
		["flow_speed", "Flow Speed", 0.0, 10.0, 0.5],
		["fluid_momentum", "Fluid Inertia", 0.0, 1.0, 0.05]
	],
	"Chemical Signal": [
		["signal_diff", "Diffusion Rate", 0.0, 10.0, 0.1],
		["signal_decay", "Decay Rate", 0.0, 1.0, 0.001],
		["signal_advect", "Advection Weight", 0.0, 1.0, 0.01],
		["signal_force_strength", "Signal Pull Force", 0.0, 100.0, 0.5],
		["signal_emission_strength", "Signal Emission (Volume)", 0.0, 20.0, 0.1]
	],
	"Wind / Atmosphere": [
		["wind_scale", "Wind Scale", 0.0, 10.0, 0.1],
		["wind_strength", "Wind Strength", 0.0, 5.0, 0.05],
		["wind_speed", "Wind Speed", 0.0, 2.0, 0.05]
	],
	"Gene Pools (Init)": [
		["g_mu", "Archetype (Mu)", 0.0, 1.0, 0.05],
		["g_sigma", "Stability (Sigma)", 0.0, 1.0, 0.05],
		["g_radius", "Effect Radius", 0.0, 1.0, 0.05],
		["g_viscosity", "Viscosity", 0.0, 1.0, 0.05],
		["g_shape_a", "Shape A (Ring Balance)", 0.0, 1.0, 0.05],
		["g_shape_b", "Shape B (Complexity)", 0.0, 1.0, 0.05],
		["g_shape_c", "Shape C (Ring Spacing)", 0.0, 1.0, 0.05],
		["g_inertia", "Inertia", 0.0, 1.0, 0.05],
		["g_affinity", "Cohesion (Affinity)", 0.0, 1.0, 0.05],
		["g_repulsion", "Repulsion", 0.0, 1.0, 0.05],
		["g_density_tol", "Density Tolerance", 0.0, 1.0, 0.05],
		["g_mobility", "Mobility", 0.0, 1.0, 0.05],
		["g_secretion", "Secretion", 0.0, 1.0, 0.05],
		["g_sensitivity", "Sensitivity", 0.0, 1.0, 0.05],
		["g_emission_hue", "Emission Hue", 0.0, 1.0, 0.05],
		["g_detection_hue", "Detection Hue", 0.0, 1.0, 0.05]
	]
}

# Statistics labels
var stats_labels = {}

func _ready():
	_build_ui()
	sim.reset_simulation() # Reset to apply new genes
	sim.stats_updated.connect(_on_stats_updated)
	sim.species_hovered.connect(_on_species_hovered)
	sim.gameplay_updated.connect(_on_gameplay_updated)

func _build_ui():
	# Stats Header
	var stats_header = Label.new()
	stats_header.text = "STATISTICS"
	stats_header.add_theme_color_override("font_color", Color(1.0, 0.8, 0.2))
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
	
	# Tooltips for parameters
	var tooltips = {
		"dt": "Time step integration delta. Lower values provide more accurate physics but slower simulation speed.",
		"init_density": "Density of the initial random noise. Higher values mean more starting matter.",
		"init_clusters": "Number of random initialization clusters placed on the grid.",
		"R": "Kernel Radius. The size of the sensing neighborhood for each cell.",
		"temperature": "Physics temperature. Controls the rate of diffusion/advection in the flow simulation.",
		"theta_A": "Critical Mass (Alpha). The density threshold where repulsion forces begin to dominate.",
		"alpha_n": "Repulsion Sharpness. Controls how abruptly the repulsion force kicks in.",
		"flow_speed": "Advection Strength Multiplier. Increases flow force without changing time step (dt).",
		"fluid_momentum": "Fluid Momentum/Inertia. 1.0 = Fluid Motion, 0.0 = Direct Movement (Easier to see forces).",
		"signal_diff": "Diffusion Rate. How fast the chemical signal spreads to neighboring cells.",
		"signal_decay": "Decay Rate. How fast the chemical signal dissipates over time.",
		"signal_advect": "Advection Weight. How much the chemical signal is dragged by the mass flow.",
		"signal_force_strength": "Signal Pull Force. Global Multiplier for the attraction strength of signals.",
		"signal_emission_strength": "Signal Emission (Volume). Global Multiplier for how much signal creatures produce.",
		"beta_selection": "Selection Pressure (β). Controls genome competition strength. 0.0=mass only, 1.0=balanced, 2.0=highly competitive.",
		
		# Genes (single range slider controls both min/max)
		"g_mu": "Archetype (Mu). Optimal density for growth.",
		"g_sigma": "Stability (Sigma). Tolerance range around the optimal density.",
		"g_radius": "Effect Radius. Relative size of the creature's influence.",
		"g_viscosity": "Viscosity. Internal drag/friction in movement.",
		"g_shape_a": "Shape A. Controls ring balance in kernel shape.",
		"g_shape_b": "Shape B. Controls pattern complexity.",
		"g_shape_c": "Shape C. Controls ring spacing.",
		"g_inertia": "Inertia. Resistance to directional changes.",
		"g_affinity": "Cohesion (Affinity). Attraction strength to own species.",
		"g_repulsion": "Repulsion. Separation force against overcrowding.",
		"g_density_tol": "Density Tolerance (Lambda). Width of the growth function.",
		"g_mobility": "Mobility (Flow). How fast the creature can move/flow.",
		"g_secretion": "Secretion. Amount of chemical signal produced.",
		"g_sensitivity": "Sensitivity. Response to the chemical signal.",
		"g_emission_hue": "Emission Hue. Identity/pitch of emitted signal.",
		"g_detection_hue": "Detection Hue. Preferred hue/pitch to react to."
	}
	
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
			var is_gene_range = group == "Gene Pools (Init)"
			
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
			if initial_val == null: initial_val = 0.0 # Fallback for new keys
			val_label.name = "Val_" + key
			val_label.add_theme_font_size_override("font_size", 10)
			val_label.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
			if is_gene_range:
				var initial_min = sim.get_parameter(key + "_min")
				var initial_max = sim.get_parameter(key + "_max")
				if initial_min == null: initial_min = min_v
				if initial_max == null: initial_max = max_v
				val_label.text = "%s - %s" % [str(initial_min).pad_decimals(3), str(initial_max).pad_decimals(3)]
			else:
				val_label.text = str(initial_val).pad_decimals(3)
			
			lbl_hbox.add_child(label)
			lbl_hbox.add_child(val_label)
			
			var slider_control: Control
			if is_gene_range:
				var range_slider = GeneRangeSlider.new()
				var range_min = sim.get_parameter(key + "_min")
				var range_max = sim.get_parameter(key + "_max")
				if range_min == null: range_min = min_v
				if range_max == null: range_max = max_v
				range_slider.setup(range_min, range_max, min_v, max_v, step_v)
				range_slider.range_changed.connect(func(v_min, v_max):
					sim.set_parameter(key + "_min", v_min)
					sim.set_parameter(key + "_max", v_max)
					val_label.text = "%s - %s" % [str(v_min).pad_decimals(3), str(v_max).pad_decimals(3)]
				)
				slider_control = range_slider
			else:
				var slider = HSlider.new()
				slider.min_value = min_v
				slider.max_value = max_v
				slider.step = step_v
				slider.value = initial_val
				slider.value_changed.connect(func(v):
					sim.set_parameter(key, v)
					val_label.text = str(v).pad_decimals(3)
				)
				slider_control = slider

			slider_control.size_flags_vertical = Control.SIZE_EXPAND_FILL
			
			# Apply Tooltips
			if tooltips.has(key):
				var tt = tooltips[key]
				label.tooltip_text = tt
				val_label.tooltip_text = tt
				slider_control.tooltip_text = tt
			
			container.add_child(lbl_hbox)
			container.add_child(slider_control)
			ui_container.add_child(container)
		
		ui_container.add_child(HSeparator.new())

	# Buttons
	var btn_row = HBoxContainer.new()
	ui_container.add_child(btn_row)
	
	var btn_pause = Button.new()
	btn_pause.text = "PAUSE"
	btn_pause.tooltip_text = "Pause or Resume the simulation."
	btn_pause.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_pause.pressed.connect(func(): 
		sim.paused = !sim.paused
		btn_pause.text = "RESUME" if sim.paused else "PAUSE"
	)
	btn_row.add_child(btn_pause)

	var btn_reset = Button.new()
	btn_reset.text = "RESET"
	btn_reset.tooltip_text = "Reset the simulation with new random seed."
	btn_reset.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	btn_reset.pressed.connect(func(): sim.reset_simulation())
	btn_row.add_child(btn_reset)
	
	var btn_clear = Button.new()
	btn_clear.text = "CLEAR"
	btn_clear.tooltip_text = "Clear the simulation grid (remove all life)."
	btn_clear.pressed.connect(func(): sim.clear_simulation())
	ui_container.add_child(btn_clear)
	
	# Resolution Selector
	ui_container.add_child(HSeparator.new())
	var res_hbox = HBoxContainer.new()
	var res_lbl = Label.new()
	res_lbl.text = "Resolution:"
	res_lbl.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
	res_hbox.add_child(res_lbl)
	
	var res_opt = OptionButton.new()
	res_opt.add_item("1024 x 1024", 0)
	res_opt.add_item("2048 x 2048", 1)
	res_opt.add_item("4096 x 4096", 2)
	
	# Set default selection based on current param
	var curr_res = int(sim.params["res_x"])
	if curr_res == 2048: res_opt.selected = 1
	elif curr_res == 4096: res_opt.selected = 2
	else: res_opt.selected = 0
	
	res_opt.item_selected.connect(func(idx):
		var size = 1024.0
		if idx == 1: size = 2048.0
		elif idx == 2: size = 4096.0
		sim.change_resolution(size, size)
	)
	res_hbox.add_child(res_opt)
	ui_container.add_child(res_hbox)

	# Strategic deck HUD
	ui_container.add_child(HSeparator.new())
	var deck_title = Label.new()
	deck_title.text = "DECK / BIOMASS"
	deck_title.add_theme_color_override("font_color", Color(1.0, 0.8, 0.2))
	deck_title.add_theme_font_size_override("font_size", 14)
	ui_container.add_child(deck_title)
	
	biomass_label = Label.new()
	biomass_label.text = "Biomass: ---"
	biomass_label.add_theme_color_override("font_color", Color(0.7, 0.9, 1.0))
	ui_container.add_child(biomass_label)
	
	deploy_hint_label = Label.new()
	deploy_hint_label.text = "Left click map to deploy selected species."
	deploy_hint_label.add_theme_font_size_override("font_size", 10)
	deploy_hint_label.add_theme_color_override("font_color", Color(0.75, 0.75, 0.75))
	ui_container.add_child(deploy_hint_label)
	
	deck_bar = HBoxContainer.new()
	deck_bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	deck_bar.add_theme_constant_override("separation", 4)
	ui_container.add_child(deck_bar)
	
	for i in range(4):
		var idx = i
		var btn = Button.new()
		btn.text = "Slot %d" % (idx + 1)
		btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		btn.pressed.connect(func():
			selected_slot = idx
			sim.select_deck_slot(idx)
		)
		deck_buttons.append(btn)
		deck_bar.add_child(btn)

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
		var total_pixels = max(1.0, sim.params["res_x"] * sim.params["res_y"])
		var coverage = (float(population) / total_pixels) * 100.0
		stats_labels["Coverage"].text = "Coverage: " + "%.2f" % coverage + "%"
		
	if stats_labels.has("Diversity"):
		stats_labels["Diversity"].text = "Diversity: " + "%.3f" % _compute_diversity(histograms)
		
	# Update Histogram
	if histogram_display:
		histogram_display.update_histograms(histograms)

func _compute_diversity(histograms) -> float:
	# Shannon entropy normalized by number of bins.
	# 0.0 => all genes concentrated in one bin, 1.0 => maximally spread.
	var total = 0.0
	for bins in histograms:
		for value in bins:
			total += float(value)

	if total <= 0.0:
		return 0.0

	var entropy = 0.0
	var bin_count = 0
	for bins in histograms:
		for value in bins:
			var p = float(value) / total
			if p > 0.0:
				entropy -= p * log(p)
			bin_count += 1

	if bin_count <= 1:
		return 0.0

	var max_entropy = log(float(bin_count))
	if max_entropy <= 0.0:
		return 0.0

	return clamp(entropy / max_entropy, 0.0, 1.0)


		
func _on_species_hovered(info):
	if tooltip_panel:
		if info.is_empty():
			tooltip_panel.visible = false
		else:
			tooltip_panel.visible = true
			var txt = "Species #%s: %s\n" % [str(info.get("id", "?")), info.get("name", "Unknown")]
			txt += "Mass: %d\n" % int(info.get("mass", 0) * 1000)
			
			# Detailed 16-Gene Breakdown
			txt += "[Physiology]\n"
			txt += "  Archetype (Mu): %.2f\n" % info.get("mu", 0.0)
			txt += "  Stability (Sigma): %.2f\n" % info.get("sigma", 0.0)
			txt += "  Radius: %.2f | Visc: %.2f\n" % [info.get("radius", 0.0), info.get("viscosity", 0.0)]
			
			txt += "[Morphology]\n"
			txt += "  Shape A/B/C: %.2f / %.2f / %.2f\n" % [info.get("shape_a", 0.0), info.get("shape_b", 0.0), info.get("shape_c", 0.0)]
			txt += "  Inertia: %.2f\n" % info.get("inertia", 0.0)
			
			txt += "[Behavior]\n"
			txt += "  Aff: %.2f | Rep: %.2f\n" % [info.get("affinity", 0.0), info.get("repulsion", 0.0)]
			txt += "  Mob: %.2f | Tol: %.2f\n" % [info.get("mobility", 0.0), info.get("density_tol", 0.0)]
			
			txt += "[Senses]\n"
			txt += "  Sec: %.2f | Sens: %.2f\n" % [info.get("secretion", 0.0), info.get("sensitivity", 0.0)]
			txt += "  Hue Emit/Det: %.2f / %.2f" % [info.get("emission_hue", 0.0), info.get("detection_hue", 0.0)]
			
			if tooltip_label:
				tooltip_label.text = txt

	# Move tooltip to mouse
	if tooltip_panel.visible:
		var mpos = get_viewport().get_mouse_position()
		tooltip_panel.position = mpos + Vector2(16, 16)


func _on_gameplay_updated(biomass: float, deck_state: Array, selected: int):
	selected_slot = selected
	if biomass_label:
		biomass_label.text = "Biomass: %.1f" % biomass
	for i in range(deck_buttons.size()):
		if i >= deck_state.size():
			continue
		var card = deck_state[i]
		var cd = float(card.get("cooldown_remaining", 0.0))
		var txt = "%d:%s\n%s C%.0f" % [i + 1, card.get("name", "Card"), card.get("role", "Role"), float(card.get("cost", 0.0))]
		if cd > 0.0:
			txt += " [%.1fs]" % cd
		deck_buttons[i].text = txt
		deck_buttons[i].modulate = Color(1.0, 0.95, 0.7) if i == selected_slot else Color(1,1,1)

func _input(event):
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode >= KEY_1 and event.keycode <= KEY_4:
			var idx = int(event.keycode - KEY_1)
			selected_slot = idx
			sim.select_deck_slot(idx)
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		if not sim.camera:
			return
		var panel = $CanvasLayer/UI/Panel
		if panel and panel.get_global_rect().has_point(event.position):
			return
		var viewport_size = get_viewport().get_visible_rect().size
		var uv = sim.camera.screen_to_uv(event.position, viewport_size)
		var ok = sim.queue_deploy(uv)
		if deploy_hint_label:
			deploy_hint_label.text = "Deploy OK at (%.2f, %.2f)" % [uv.x, uv.y] if ok else "Cannot deploy: low biomass, cooldown, or out of bounds."

func _process(_delta):
	# Update tooltip pos if visible
	if tooltip_panel and tooltip_panel.visible:
		var mpos = get_viewport().get_mouse_position()
		tooltip_panel.position = mpos + Vector2(16, 16)
