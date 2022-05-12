import numpy as np

def funny_1B_force(bead1, test_parameter):

	return np.zeros(3)

def angry_bonded_2B_force(bead1, bead2, pointer, test_parameter):

	return np.ones(3)

def invalid_function(bead1, bead2, pointer):

	return None

def angry_bonded_2B_energy(bead1, bead2, pointer, test_parameter):

	return -1.0

def funny_1B_energy(bead1, test_parameter):

	return 100.0