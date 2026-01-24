class_name SpeciesTracker
extends RefCounted

# Analyzing 64x64 grid (4096 samples)
const GRID_SIZE = 64
const CELL_FLOATS = 8 # Struct size in floats (32 bytes)
const MASS_THRESHOLD = 0.05 # Minimum avg mass
const GENE_SIMILARITY_THRESHOLD = 0.1 # Max distance to be same species

class Species:
	var id: int
	var mass: float = 0.0
	var area: int = 0
	
	# Genetic Centroid (Running Average)
	var genes_sum = {
		"mu": 0.0, "sigma": 0.0, "radius": 0.0, 
		"flow": 0.0, "affinity": 0.0, "lambda": 0.0
	}
	
	# Current Average Genes
	var genes = {}
	var color: Color
	var name: String = "Unknown"
	
	func add_sample(mu, sig, rad, flow, aff, lam, m):
		area += 1
		mass += m
		genes_sum["mu"] += mu
		genes_sum["sigma"] += sig
		genes_sum["radius"] += rad
		genes_sum["flow"] += flow
		genes_sum["affinity"] += aff
		genes_sum["lambda"] += lam
		
	func finalize():
		if area == 0: return
		var n = float(area)
		genes = {
			"mu": genes_sum["mu"] / n,
			"sigma": genes_sum["sigma"] / n,
			"radius": genes_sum["radius"] / n,
			"flow": genes_sum["flow"] / n,
			"affinity": genes_sum["affinity"] / n,
			"lambda": genes_sum["lambda"] / n
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
		var lam = genes["lambda"]
		var flow = genes["flow"]
		
		var prefix = ""
		if mu < 0.3: prefix = "Globus"
		elif mu < 0.6: prefix = "Amorph"
		else: prefix = "Vermes"
		
		var suffix = ""
		if sig < 0.3: suffix = "Solidus"
		elif sig < 0.6: suffix = "Varius"
		else: suffix = "Nebula"
		
		var adj = ""
		if flow > 0.7: adj = " Velox"
		if lam < 0.2: adj = " Tardus"
		
		name = prefix + " " + suffix + adj

# Returns list of Species objects
func find_species(byte_data: PackedByteArray) -> Array:
	if byte_data.size() < GRID_SIZE * GRID_SIZE * CELL_FLOATS * 4:
		return []
		
	var floats = byte_data.to_float32_array()
	var species_list: Array[Species] = []
	var next_id = 1
	
	# Loop all cells
	var count = GRID_SIZE * GRID_SIZE
	for i in range(count):
		var base = i * CELL_FLOATS
		var m = floats[base]
		
		if m > MASS_THRESHOLD:
			var mu = floats[base + 1]
			var sig = floats[base + 2]
			var rad = floats[base + 3]
			var flow = floats[base + 4]
			var aff = floats[base + 5]
			var lam = floats[base + 6]
			
			# Find in existing species clusters
			var found = false
			for s in species_list:
				# Calculate distance to current species average
				# Use Mu/Sigma as primary discriminators (could use all)
				var d_mu = abs(s.genes_sum["mu"]/s.area - mu)
				var d_sig = abs(s.genes_sum["sigma"]/s.area - sig)
				
				# Weighted distance? Or simple sum
				if (d_mu + d_sig) < GENE_SIMILARITY_THRESHOLD:
					s.add_sample(mu, sig, rad, flow, aff, lam, m)
					found = true
					break
			
			if not found:
				# New Species
				var s = Species.new()
				s.id = next_id
				next_id += 1
				s.add_sample(mu, sig, rad, flow, aff, lam, m)
				species_list.append(s)
	
	# Finalize
	for s in species_list:
		s.finalize()
	
	# Filter tiny clusters (noise)
	var final_list = []
	for s in species_list:
		if s.mass > 1.0: # Arbitrary mass threshold to ignore dust
			final_list.append(s)
			
	# Sort by mass
	final_list.sort_custom(func(a, b): return a.mass > b.mass)
	
	return final_list
