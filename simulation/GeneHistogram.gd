extends Control

var gene_names = [
	# Physiology
	"Archetype", "Stability", "Radius", "Viscosity",
	# Morphology
	"Shape A", "Shape B", "Shape C", "Growth Rate",
	# Behavior
	"Affinity", "Repulsion", "DensityTol", "Mobility",
	# Senses
	"Secretion", "Sensitivity", "Emit Hue", "Detect Hue"
]

var gene_colors = [
	# Phy (Reds/Greens)
	Color(1.0, 0.4, 0.4), Color(0.4, 1.0, 0.4), Color(0.4, 0.4, 1.0), Color(0.6, 0.6, 0.6),
	# Mor (Purples/Cyans)
	Color(0.8, 0.2, 0.8), Color(0.6, 0.2, 0.6), Color(0.4, 0.2, 0.4), Color(0.2, 1.0, 0.2),
	# Beh (Oranges/Yellows)
	Color(1.0, 0.5, 0.0), Color(1.0, 0.2, 0.0), Color(1.0, 1.0, 0.2), Color(0.0, 1.0, 1.0),
	# Sen (Special)
	Color(1.0, 0.0, 1.0), Color(0.0, 1.0, 0.5), Color(1.0, 1.0, 1.0), Color(0.5, 0.5, 0.5)
]

var data = [] 

func _ready():
	custom_minimum_size = Vector2(0, 240) # 8 rows * 30px

func update_histograms(new_data):
	data = new_data
	queue_redraw()

func _draw():
	if data.size() != 16: return
	
	var w = size.x
	var h = size.y
	var col_count = 2
	var rows_per_col = 8
	var col_w = w / col_count
	var row_h = float(h) / float(rows_per_col)
	
	var label_width = 70.0
	var graph_width = col_w - label_width - 10.0 # 10px padding
	var bar_w = graph_width / 10.0
	
	var font = get_theme_default_font()
	
	for g in range(16):
		var col_idx = g / 8
		var row_idx = g % 8
		
		var x_base = col_idx * col_w
		var y_base = row_idx * row_h
		
		# Draw Label
		draw_string(font, Vector2(x_base, y_base + row_h - 8), gene_names[g], HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(0.8, 0.8, 0.8))
		
		# Draw Histogram
		# Access data[g] which is an array of 10 bins
		var bins = data[g]
		var max_val = 1
		for b in range(10):
			if bins[b] > max_val: max_val = bins[b]
			
		for b in range(10):
			var val = bins[b]
			var bar_h = (float(val) / float(max_val)) * (row_h - 4.0)
			var rect = Rect2(x_base + label_width + b * bar_w, y_base + row_h - 2 - bar_h, bar_w - 1, bar_h)
			
			var col = gene_colors[g]
			col.a = 0.5 + 0.5 * (float(val) / float(max_val))
			
			draw_rect(rect, col)
