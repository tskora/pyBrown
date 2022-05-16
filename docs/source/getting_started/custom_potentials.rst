.. _custom-potentials:

Custom potentials
-------------------

You can define your own potentials and use them in ``pyBrown`` simulation. However, you have to align with particular naming scheme. There are two questions you have to answer at first: whether the potential is **bonded** (only bonded beads interact, eg. harmonic bonds, angle potentials) and **how-many-body** the potential is. In your file you have to define two functions: one for the force, and one for the energy per every interaction you want to define. The naming scheme is: ``some_name(_bonded)_NB_force``/``some_name(_bonded)_NB_energy``, where ``_bonded`` is used only whene the potential is bonded. Instead of N use how-many-body the potential is. Instead of ``some_name`` you can use any name you wish. For example look at the file ``src/bdsim/tests/foo.py``:

.. code:: python
    
    import numpy as np

	def funny_1B_force(bead1, test_parameter):

		return np.array([0.0, 1.0, 2.0])

	def angry_bonded_2B_force(bead1, bead2, pointer, test_parameter):

		return np.ones(3)

	def invalid_function(bead1, bead2, pointer):

		return None

	def angry_bonded_2B_energy(bead1, bead2, pointer, test_parameter):

		return -1.0

	def funny_1B_energy(bead1, test_parameter):

		return 100.0

``invalid_function`` does not follow the naming scheme, so it will be ignored. Then, ``funny_1B_force`` and ``funny_1B_energy`` define **nonbonded**, **1-body** interaction. Finally, ``angry_bonded_2B_force`` and ``angry_bonded_2B_energy`` define **bonded**, **2-body** interaction.

Return
*******

Energy function has to return a ``float`` value. Force function has to return a three-dimenstional ``numpy.ndarray`` of floats.

Arguments
**********

When it comes to arguments, it depends on the how-many-body the interaction is and on the additional parameters you want to provide to the interactions (see :ref:`preparing-input`). The scheme looks like that:

``1B_force(bead1, ...)``

``2B_force(ead1, bead2, pointer, ...)``

``3B_force(bead1, bead2, bead3, pointer12, pointer23, ...)``

``4B_force(bead1, bead2, bead3, bead4, pointer12, pointer23, pointer34, ...)``

``1B_energy(bead1, ...)``

``2B_energy(bead1, bead2, pointer, ...)``

``3B_energy(bead1, bead2, bead3, pointer12, pointer23, ...)``

``4B_energy(bead1, bead2, bead3, bead4, pointer12, pointer23, pointer34, ...)``

where instead of ... you have to put **all** the auxialiary interactions keyword you proviade via ``.json`` input file.