class_name SpeciesTracker
extends RefCounted

const GRID_SIZE = 64
const CELL_FLOATS = 18 # Mass + 16 genes + polarity
const MASS_THRESHOLD = 0.05
const GENE_SIMILARITY_THRESHOLD = 0.32

# Gene indices
const G_MU = 0
const G_SIGMA = 1
const G_RADIUS = 2
const G_VISCOSITY = 3
const G_SHAPE_A = 4
const G_SHAPE_B = 5
const G_SHAPE_C = 6
const G_ANISOTROPY = 7
const G_COMPACTNESS = 8
const G_REPULSION = 9
const G_PLASTICITY = 10
const G_MOBILITY = 11
const G_SECRETION = 12
const G_SENSITIVITY = 13
const G_EMIT_HUE = 14
const G_DETECT_HUE = 15

class Species:
	var id: int
	var mass: float = 0.0
	var area: int = 0

	var genes_sum = {
		# Physiology
		"mu": 0.0, "sigma": 0.0, "radius": 0.0, "viscosity": 0.0,
		# Morphology
		"shape_a": 0.0, "shape_b": 0.0, "shape_c": 0.0, "anisotropy": 0.0,
		# Body plan / Motor
		"compactness": 0.0, "repulsion": 0.0, "plasticity": 0.0, "mobility": 0.0,
		# Senses
		"secretion": 0.0, "sensitivity": 0.0, "emission_hue": 0.0, "detection_hue": 0.0,
		# State
		"polarity": 0.0
	}

	var genes = {}
	var color: Color
	var name: String = "Unknown"

	func add_sample(sample_genes: PackedFloat32Array, m: float, pol: float):
		area += 1
		mass += m

		genes_sum["mu"] += sample_genes[G_MU]
		genes_sum["sigma"] += sample_genes[G_SIGMA]
		genes_sum["radius"] += sample_genes[G_RADIUS]
		genes_sum["viscosity"] += sample_genes[G_VISCOSITY]

		genes_sum["shape_a"] += sample_genes[G_SHAPE_A]
		genes_sum["shape_b"] += sample_genes[G_SHAPE_B]
		genes_sum["shape_c"] += sample_genes[G_SHAPE_C]
		genes_sum["anisotropy"] += sample_genes[G_ANISOTROPY]

		genes_sum["compactness"] += sample_genes[G_COMPACTNESS]
		genes_sum["repulsion"] += sample_genes[G_REPULSION]
		genes_sum["plasticity"] += sample_genes[G_PLASTICITY]
		genes_sum["mobility"] += sample_genes[G_MOBILITY]

		genes_sum["secretion"] += sample_genes[G_SECRETION]
		genes_sum["sensitivity"] += sample_genes[G_SENSITIVITY]
		genes_sum["emission_hue"] += sample_genes[G_EMIT_HUE]
		genes_sum["detection_hue"] += sample_genes[G_DETECT_HUE]
		genes_sum["polarity"] += pol

	func finalize():
		if area == 0:
			return
		var inv_n = 1.0 / float(area)
		for k in genes_sum.keys():
			genes[k] = genes_sum[k] * inv_n

		color = Color.from_hsv(genes["emission_hue"], 0.85, 1.0)
		_generate_name()

	func _generate_name():
		var mu = genes["mu"]
		var anis = genes["anisotropy"]
		var mob = genes["mobility"]

		var noun = "Proto"
		if mu < 0.2: noun = "Globus"
		elif mu < 0.4: noun = "Limbus"
		elif mu < 0.6: noun = "Vermes"
		elif mu < 0.8: noun = "Cellula"
		else: noun = "Structura"

		var morph_adj = "Orbis"
		if anis > 0.7:
			morph_adj = "Filum"
		elif anis > 0.45:
			morph_adj = "Arcus"

		var beh_adj = "Vagus"
		if mob > 0.7:
			beh_adj = "Velox"
		elif mob < 0.3:
			beh_adj = "Pigra"

		name = "%s %s %s" % [noun, morph_adj, beh_adj]

static func get_fast_dist(g1: PackedFloat32Array, g2: PackedFloat32Array) -> float:
	var d = 0.0
	d += abs(g1[G_MU] - g2[G_MU]) * 2.0
	d += abs(g1[G_SIGMA] - g2[G_SIGMA]) * 1.5
	d += abs(g1[G_RADIUS] - g2[G_RADIUS]) * 1.0
	d += abs(g1[G_VISCOSITY] - g2[G_VISCOSITY]) * 0.5
	d += abs(g1[G_SHAPE_A] - g2[G_SHAPE_A]) * 0.7
	d += abs(g1[G_SHAPE_B] - g2[G_SHAPE_B]) * 0.7
	d += abs(g1[G_SHAPE_C] - g2[G_SHAPE_C]) * 0.7
	d += abs(g1[G_ANISOTROPY] - g2[G_ANISOTROPY]) * 1.3
	d += abs(g1[G_COMPACTNESS] - g2[G_COMPACTNESS]) * 1.0
	d += abs(g1[G_REPULSION] - g2[G_REPULSION]) * 0.8
	d += abs(g1[G_PLASTICITY] - g2[G_PLASTICITY]) * 1.0
	d += abs(g1[G_MOBILITY] - g2[G_MOBILITY]) * 1.0

	var hue_diff = abs(g1[G_EMIT_HUE] - g2[G_EMIT_HUE])
	if hue_diff > 0.5:
		hue_diff = 1.0 - hue_diff
	d += hue_diff * 2.0
	return d

static func get_gene_distance(g1: Dictionary, g2: Dictionary) -> float:
	var d = 0.0
	d += abs(g1.get("mu", 0.0) - g2.get("mu", 0.0)) * 2.0
	d += abs(g1.get("sigma", 0.0) - g2.get("sigma", 0.0)) * 1.5
	d += abs(g1.get("radius", 0.0) - g2.get("radius", 0.0)) * 1.0
	d += abs(g1.get("viscosity", 0.0) - g2.get("viscosity", 0.0)) * 0.5
	d += abs(g1.get("shape_a", 0.0) - g2.get("shape_a", 0.0)) * 0.7
	d += abs(g1.get("shape_b", 0.0) - g2.get("shape_b", 0.0)) * 0.7
	d += abs(g1.get("shape_c", 0.0) - g2.get("shape_c", 0.0)) * 0.7
	d += abs(g1.get("anisotropy", 0.0) - g2.get("anisotropy", 0.0)) * 1.3
	d += abs(g1.get("compactness", 0.0) - g2.get("compactness", 0.0)) * 1.0
	d += abs(g1.get("repulsion", 0.0) - g2.get("repulsion", 0.0)) * 0.8
	d += abs(g1.get("plasticity", 0.0) - g2.get("plasticity", 0.0)) * 1.0
	d += abs(g1.get("mobility", 0.0) - g2.get("mobility", 0.0)) * 1.0

	var h1 = g1.get("emission_hue", 0.0)
	var h2 = g2.get("emission_hue", 0.0)
	var hd = abs(h1 - h2)
	if hd > 0.5:
		hd = 1.0 - hd
	d += hd * 2.0
	return d

var _temp_gene_buffer := PackedFloat32Array()

func _init():
	_temp_gene_buffer.resize(16)

func find_species(byte_data: PackedByteArray) -> Array:
	if byte_data.size() < GRID_SIZE * GRID_SIZE * CELL_FLOATS * 4:
		return []

	var floats = byte_data.to_float32_array()
	var species_list: Array[Species] = []
	var species_genes: Array[PackedFloat32Array] = []

	var count = GRID_SIZE * GRID_SIZE
	var floats_size = floats.size()

	for i in range(count):
		var base = i * CELL_FLOATS
		if base + 17 >= floats_size:
			break

		var m = floats[base]
		if m <= MASS_THRESHOLD:
			continue

		for k in range(16):
			_temp_gene_buffer[k] = floats[base + 1 + k]
		var pol = floats[base + 17]

		# Ignore inert/dead matter: mass can exist with zeroed genome.
		var identity_strength = _temp_gene_buffer[G_MU] + _temp_gene_buffer[G_SIGMA] + _temp_gene_buffer[G_RADIUS]
		if identity_strength < 0.01:
			continue

		var best_match_idx = -1
		var min_dist = GENE_SIMILARITY_THRESHOLD

		for j in range(species_genes.size()):
			var d = get_fast_dist(species_genes[j], _temp_gene_buffer)
			if d < min_dist:
				min_dist = d
				best_match_idx = j
				if d < 0.05:
					break

		if best_match_idx != -1:
			species_list[best_match_idx].add_sample(_temp_gene_buffer, m, pol)
		elif species_list.size() < 64:
			var s = Species.new()
			s.id = species_list.size() + 1

			var new_gene_snapshot = _temp_gene_buffer.duplicate()
			s.add_sample(new_gene_snapshot, m, pol)

			species_list.append(s)
			species_genes.append(new_gene_snapshot)

	var final_list = []
	for s in species_list:
		if s.area > 0:
			s.finalize()
			if s.mass > 1.0:
				final_list.append(s)

	final_list.sort_custom(func(a, b): return a.mass > b.mass)
	return final_list
