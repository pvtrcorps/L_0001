class_name SpeciesTracker
extends RefCounted

# Analyzing 64x64 grid (4096 samples)
const GRID_SIZE = 64
const CELL_FLOATS = 18 # Mass (1) + Genes (16) + Padding (1) = 18
const MASS_THRESHOLD = 0.05
const GENE_SIMILARITY_THRESHOLD = 0.25 # Slightly looser for high-D space

class Species:
	var id: int
	var mass: float = 0.0
	var area: int = 0
	
	# Storage for all 16 genes
	var genes_sum = {
		# Phys
		"mu": 0.0, "sigma": 0.0, "radius": 0.0, "viscosity": 0.0,
		# Morph
		"shape_a": 0.0, "shape_b": 0.0, "shape_c": 0.0, "growth_rate": 0.0,
		# Behavior
		"affinity": 0.0, "repulsion": 0.0, "density_tol": 0.0, "mobility": 0.0,
		# Senses
		"secretion": 0.0, "sensitivity": 0.0, "emission_hue": 0.0, "detection_hue": 0.0
	}
	
	var genes = {}
	var color: Color
	var name: String = "Unknown"
	
	func add_sample(sample_genes: Array, m: float):
		area += 1
		mass += m
		
		genes_sum["mu"] += sample_genes[0]
		genes_sum["sigma"] += sample_genes[1]
		genes_sum["radius"] += sample_genes[2]
		genes_sum["viscosity"] += sample_genes[3]
		
		genes_sum["shape_a"] += sample_genes[4]
		genes_sum["shape_b"] += sample_genes[5]
		genes_sum["shape_c"] += sample_genes[6]
		genes_sum["growth_rate"] += sample_genes[7]
		
		genes_sum["affinity"] += sample_genes[8]
		genes_sum["repulsion"] += sample_genes[9]
		genes_sum["density_tol"] += sample_genes[10]
		genes_sum["mobility"] += sample_genes[11]
		
		genes_sum["secretion"] += sample_genes[12]
		genes_sum["sensitivity"] += sample_genes[13]
		genes_sum["emission_hue"] += sample_genes[14]
		genes_sum["detection_hue"] += sample_genes[15]
		
	func finalize():
		if area == 0: return
		var n = float(area)
		
		for k in genes_sum.keys():
			genes[k] = genes_sum[k] / n
		
		# Calculate Color based on Hue Emission
		# If emission is low, fallback to physiology colors?
		# Actually, let's use Emission Hue primarily as it's the "Signal" color
		var hues = genes["emission_hue"]
		var sat = 0.8
		var val = 1.0
		
		# Use HSV for spectral accuracy
		color = Color.from_hsv(hues, sat, val)
		
		_generate_name()

	func _generate_name():
		var mu = genes["mu"]
		var rad = genes["radius"]
		var mob = genes["mobility"]
		var aff = genes["affinity"]
		
		# 1. Physiology (Mu) -> Noun
		var noun = "Proto"
		if mu < 0.2: noun = "Globus"
		elif mu < 0.4: noun = "Limbus"
		elif mu < 0.6: noun = "Vermes"
		elif mu < 0.8: noun = "Cellula"
		else: noun = "Structura"
		
		# 2. Size (Radius) -> Adjective 1
		var size_adj = ""
		if rad < 0.3: size_adj = "Micro "
		elif rad > 0.7: size_adj = "Mega "
		
		# 3. Behavior (Mobility/Affinity) -> Adjective 2
		var beh_adj = ""
		if mob > 0.7: beh_adj = "Velox" # Fast
		elif mob < 0.3: beh_adj = "Pigra" # Slow
		elif aff > 0.7: beh_adj = "Socialis" # Social
		elif aff < 0.3: beh_adj = "Solus" # Loner
		else: beh_adj = "Vagus" # Wandering
		
		name = size_adj + noun + " " + beh_adj

# Weighted genetic distance
static func get_fast_dist(g1: PackedFloat32Array, g2: PackedFloat32Array) -> float:
	var d = 0.0
	# Physiology (High weight)
	d += abs(g1[0] - g2[0]) * 2.0 # mu
	d += abs(g1[1] - g2[1]) * 1.5 # sigma
	d += abs(g1[2] - g2[2]) * 1.0 # radius
	d += abs(g1[3] - g2[3]) * 0.5 # viscosity
	d += abs(g1[4] - g2[4]) * 0.8 # shape_a
	d += abs(g1[8] - g2[8]) * 0.5 # affinity
	d += abs(g1[11] - g2[11]) * 1.0 # mobility
	
	# Emission Hue (Critical for speciation)
	# Handle circular distance for hue? [0,1]
	# Simple diff for now
	var hue_diff = abs(g1[14] - g2[14])
	if hue_diff > 0.5: hue_diff = 1.0 - hue_diff
	d += hue_diff * 2.0
	
	return d

static func get_gene_distance(g1: Dictionary, g2: Dictionary) -> float:
	# Keep consistent with fast_dist
	var d = 0.0
	d += abs(g1["mu"] - g2["mu"]) * 2.0
	d += abs(g1["sigma"] - g2["sigma"]) * 1.5
	d += abs(g1["radius"] - g2["radius"]) * 1.0
	d += abs(g1.get("viscosity",0.0) - g2.get("viscosity",0.0)) * 0.5
	d += abs(g1.get("shape_a",0.0) - g2.get("shape_a",0.0)) * 0.8
	d += abs(g1.get("affinity",0.0) - g2.get("affinity",0.0)) * 0.5
	d += abs(g1.get("mobility",0.0) - g2.get("mobility",0.0)) * 1.0
	var h1 = g1["emission_hue"]
	var h2 = g2["emission_hue"]
	var hd = abs(h1 - h2)
	if hd > 0.5: hd = 1.0 - hd
	d += hd * 2.0
	return d

func find_species(byte_data: PackedByteArray) -> Array:
	if byte_data.size() < GRID_SIZE * GRID_SIZE * CELL_FLOATS * 4:
		return []
		
	var floats = byte_data.to_float32_array()
	var species_list: Array[Species] = []
	var species_genes: Array[PackedFloat32Array] = []
	
	var count = GRID_SIZE * GRID_SIZE
	for i in range(count):
		var base = i * CELL_FLOATS
		var m = floats[base]
		if m <= MASS_THRESHOLD: continue
		
		# Collect all 16 genes
		var cg = PackedFloat32Array()
		cg.resize(16)
		for k in range(16):
			cg[k] = floats[base + 1 + k]
			
		var best_match_idx = -1
		var min_dist = GENE_SIMILARITY_THRESHOLD
		
		for j in range(species_genes.size()):
			var d = get_fast_dist(species_genes[j], cg)
			if d < min_dist:
				min_dist = d
				best_match_idx = j
				if d < 0.05: break
		
		if best_match_idx != -1:
			# Convert PackedFloat32Array to Array for helper
			var arr = []
			for k in range(16): arr.append(cg[k])
			species_list[best_match_idx].add_sample(arr, m)
		elif species_list.size() < 64:
			var s = Species.new()
			s.id = species_list.size() + 1
			
			var arr = []
			for k in range(16): arr.append(cg[k])
			s.add_sample(arr, m)
			
			species_list.append(s)
			species_genes.append(cg)
			
	# Finalize
	var final_list = []
	for s in species_list:
		if s.area > 0: # Ensure valid
			s.finalize()
			if s.mass > 1.0:
				final_list.append(s)
			
	final_list.sort_custom(func(a, b): return a.mass > b.mass)
	return final_list
