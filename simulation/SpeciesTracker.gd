class_name SpeciesTracker
extends RefCounted

# Analyzing 64x64 grid (4096 samples)
const GRID_SIZE = 64
const CELL_FLOATS = 10 # Struct size in floats (40 bytes)
const MASS_THRESHOLD = 0.05 # Minimum avg mass
const GENE_SIMILARITY_THRESHOLD = 0.1 # Max distance to be same species

class Species:
	var id: int
	var mass: float = 0.0
	var area: int = 0
	
	# Genetic Centroid (Running Average)
	var genes_sum = {
		"mu": 0.0, "sigma": 0.0, "radius": 0.0, 
		"flow": 0.0, "affinity": 0.0, "lambda": 0.0,
		"secretion": 0.0, "perception": 0.0
	}
	
	# Current Average Genes
	var genes = {}
	var color: Color
	var name: String = "Unknown"
	
	func add_sample(mu, sig, rad, flow, aff, lam, sec, per, m):
		area += 1
		mass += m
		genes_sum["mu"] += mu
		genes_sum["sigma"] += sig
		genes_sum["radius"] += rad
		genes_sum["flow"] += flow
		genes_sum["affinity"] += aff
		genes_sum["lambda"] += lam
		genes_sum["secretion"] += sec
		genes_sum["perception"] += per
		
	func finalize():
		if area == 0: return
		var n = float(area)
		genes = {
			"mu": genes_sum["mu"] / n,
			"sigma": genes_sum["sigma"] / n,
			"radius": genes_sum["radius"] / n,
			"flow": genes_sum["flow"] / n,
			"affinity": genes_sum["affinity"] / n,
			"lambda": genes_sum["lambda"] / n,
			"secretion": genes_sum["secretion"] / n,
			"perception": genes_sum["perception"] / n
		}
		
		# Calculate Color
		var h = genes["mu"]
		var s = 0.5 + genes["sigma"] * 0.5
		var v = 1.0 
		color = Color.from_hsv(h, s, v)
		
		_generate_name()

	func _generate_name():
		var mu = genes["mu"]
		var sig = genes["sigma"]
		var per = genes["perception"]
		var sec = genes["secretion"]
		var rad = genes["radius"]
		var aff = genes["affinity"]
		
		var prefix = ""
		if mu < 0.3: prefix = "Globus"
		elif mu < 0.6: prefix = "Amorph"
		else: prefix = "Vermes"
		
		# Adding size-based adjectives
		var size = ""
		if rad < 0.3: size = " Parvus"
		elif rad > 0.7: size = " Gigas"
		
		var suffix = ""
		if sig < 0.3: suffix = " Solidus"
		elif sig < 0.6: suffix = " Varius"
		else: suffix = " Nebula"
		
		# Armor/Social based suffix
		var type = ""
		if aff < 0.2: type = " Nudus"
		elif aff > 0.8: type = " Crustatus"
		
		var behavior = ""
		if sec > 0.6: behavior = " Pheromo"
		if per > 0.6: behavior = " Sensus"
		if sec > 0.6 and per > 0.6: behavior = " Socialis"
		
		name = prefix + size + suffix + type + behavior

# Optimized weighted distance
static func get_fast_dist(g1: PackedFloat32Array, g2: PackedFloat32Array) -> float:
	var d = 0.0
	d += abs(g1[0] - g2[0]) * 1.5 # mu
	d += abs(g1[1] - g2[1]) * 1.5 # sigma
	if d > GENE_SIMILARITY_THRESHOLD: return d # Fast exit
	d += abs(g1[2] - g2[2]) * 1.0 # radius
	d += abs(g1[4] - g2[4]) * 1.0 # affinity
	return d

# Dictionary-friendly version for UI/Inspection
static func get_gene_distance(g1: Dictionary, g2: Dictionary) -> float:
	var d = 0.0
	d += abs(g1["mu"] - g2["mu"]) * 1.5
	d += abs(g1["sigma"] - g2["sigma"]) * 1.5
	d += abs(g1["radius"] - g2["radius"]) * 1.0
	d += abs(g1["affinity"] - g2["affinity"]) * 1.0
	return d

func find_species(byte_data: PackedByteArray) -> Array:
	if byte_data.size() < GRID_SIZE * GRID_SIZE * CELL_FLOATS * 4:
		return []
		
	var floats = byte_data.to_float32_array()
	var species_list: Array[Species] = []
	
	# Cache gene arrays to avoid Dictionary hits in the hot loop
	var species_genes: Array[PackedFloat32Array] = []
	
	var count = GRID_SIZE * GRID_SIZE
	for i in range(count):
		var base = i * CELL_FLOATS
		var m = floats[base]
		if m <= MASS_THRESHOLD: continue
		
		# Current cell genes as array
		var cg = PackedFloat32Array([
			floats[base+1], floats[base+2], floats[base+3], floats[base+4],
			floats[base+5], floats[base+6], floats[base+7], floats[base+8]
		])
		
		var best_match_idx = -1
		var min_dist = GENE_SIMILARITY_THRESHOLD
		
		# Hot Loop: Compare against existing centroids
		for j in range(species_genes.size()):
			var d = get_fast_dist(species_genes[j], cg)
			if d < min_dist:
				min_dist = d
				best_match_idx = j
				if d < 0.02: break # "Close enough" exit
		
		if best_match_idx != -1:
			species_list[best_match_idx].add_sample(cg[0], cg[1], cg[2], cg[3], cg[4], cg[5], cg[6], cg[7], m)
		elif species_list.size() < 64: # Hard cap on species count
			var s = Species.new()
			s.id = species_list.size() + 1
			s.genes = { "mu": cg[0], "sigma": cg[1], "radius": cg[2], "flow": cg[3], "affinity": cg[4], "lambda": cg[5], "secretion": cg[6], "perception": cg[7] }
			s.add_sample(cg[0], cg[1], cg[2], cg[3], cg[4], cg[5], cg[6], cg[7], m)
			species_list.append(s)
			species_genes.append(cg)
	
	# Finalize and Filter
	var final_list = []
	for s in species_list:
		s.finalize()
		if s.mass > 1.0:
			final_list.append(s)
			
	final_list.sort_custom(func(a, b): return a.mass > b.mass)
	return final_list
