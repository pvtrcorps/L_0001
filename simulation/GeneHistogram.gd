extends Control

var gene_names = ["Mu", "Sigma", "Radius", "Flow", "Affinity", "Lambda"]
# Colors matching the visualization shader logic roughly
# Mu=Hue (Spectrum), Sigma=Sat (Greenish?), Lambda=Val
var gene_colors = [
	Color(1.0, 0.2, 0.2), # Mu - Red
	Color(0.2, 1.0, 0.2), # Sigma - Green
	Color(0.2, 0.2, 1.0), # Radius - Blue
	Color(0.2, 1.0, 1.0), # Flow - Cyan
	Color(1.0, 0.2, 1.0), # Affinity - Magenta
	Color(1.0, 1.0, 0.2)  # Lambda - Yellow
]
var data = [] 

func _ready():
	custom_minimum_size = Vector2(0, 200)

func update_histograms(new_data):
	data = new_data
	queue_redraw()

func _draw():
	if data.size() < 60: return
	
	var w = size.x
	var h = size.y
	var row_h = h / 6.0
	var label_width = 60.0
	var graph_width = w - label_width
	var bar_w = graph_width / 10.0
	
	var font = get_theme_default_font()
	
	for g in range(6):
		var y_base = g * row_h
		
		# Draw Label
		draw_string(font, Vector2(0, y_base + row_h - 5), gene_names[g], HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.8, 0.8, 0.8))
		
		# Find max in this gene to normalize view
		var max_val = 1
		for b in range(10):
			var val = data[g*10 + b]
			if val > max_val: max_val = val
			
		# Draw Bars
		for b in range(10):
			var val = data[g*10 + b]
			var bar_h = (float(val) / float(max_val)) * (row_h - 4)
			# x pos: label_width + bucket_index * bar_w
			# y pos: bottom of row - bar_h
			var rect = Rect2(label_width + b * bar_w, y_base + row_h - 2 - bar_h, bar_w - 2, bar_h)
			
			# Modulate color alpha based on height for effect
			var col = gene_colors[g]
			col.a = 0.5 + 0.5 * (float(val) / float(max_val))
			
			draw_rect(rect, col)
