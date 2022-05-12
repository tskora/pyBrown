import numpy as np

def funny_force(bead1, bead2, pointer, test_parameter):

	return np.zeros(3)

def angry_force(bead1, bead2, pointer, test_parameter):

	return np.ones(3)

def invalid_function(bead1, bead2, pointer):

	return None

def angry_energy(bead1, bead2, pointer, test_parameter):

	return -1.0

def funny_energy(bead1, bead2, pointer, test_parameter):

	return 100.0