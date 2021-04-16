Preparing input
----------------

Input for ``pyBrown`` consists of two files:

- ``.json`` -- file providing simulation parameters,
- ``.str`` -- file providing initial configuration of beads and their parameters.

Keywords
*********

Physical constants
^^^^^^^^^^^^^^^^^^^

- ``"T": float`` -- Temperature (*K*), **required**
- ``"viscosity": float`` -- Viscosity (*P*), **required**

Randomness
^^^^^^^^^^^

- ``"seed": int`` -- seed for pseudorandom number generation algorithm, default ``np.random.randint(2**32 - 1)``

Input/Output
^^^^^^^^^^^^^

- ``"input_str_filename": string`` -- filename of input ``.str`` file (*see above*), **required**
- ``"output_xyz_filename": string`` -- filename to which write the trajectory, **required**
- ``"xyz_write_freq": int`` -- , default: ``1``
- ``"output_rst_filename": string`` -- filename to which write the restart
- ``"rst_write_freq": int`` -- 

- ``"debug": boolean`` -- switching on/off the debug printout, default: ``false``
- ``"verbose": boolean`` -- switching on/off the verbose printout, default: ``false``
- ``"progress_bar": boolean`` -- switching on/off the progress bar, default: ``false``

Simulation box
^^^^^^^^^^^^^^^

- ``"box_length": float`` -- length of the cubic box side (*Å*), **required**
- ``"dt": float`` -- timestep (*ps*), **required**
- ``"number_of_steps": int`` -- number of simulation steps, **required**

Hydrodynamic interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``"hydrodynamics": option`` -- options: ``"nohi"``, ``"rpy"``, ``"rpy_smith"``, ``"rpy_lub"``, ``rpy_smith_lub``, default: ``"nohi"``
- ``"propagation_scheme": option`` -- options: ``"ermak"``, ``"midpoint"``, default: ``"ermak"``
- ``"diff_freq": int`` -- , default: ``1``
- ``"lub_freq": int`` -- , default: ``1``
- ``"chol_freq": int`` -- , default: ``1``

- ``"ewald_alpha": float`` -- , default: ``np.sqrt(np.pi)``
- ``"ewald_real": int`` -- , default: ``0``
- ``"ewald_imag": int`` -- , default: ``0``

External force
^^^^^^^^^^^^^^^

- ``"external_force": [float, float, float]`` -- default: ``[0.0, 0.0, 0.0]``

Other
^^^^^^

- ``"check_overlaps": boolean`` -- default: ``true``

- ``"immobile_labels": [string, ..., string]`` -- default: ``[]``

Keyword blocks
***************

*some more exotic options are activated by keywords which are dictionaries itself*

Beads
******

Structure file consists of lines with a following structure::

``sub label index x y z a q 2r e m``

Only the ``sub`` string is identical in every line. All the others should be replaced with values specific to a bead.

- ``label (string)`` -- bead label
- ``index (int)`` -- bead number
- ``x y z (float float float)`` -- ``x``, ``y`` and ``z`` coordinates of the bead center (*Å*)
- ``a (float)`` -- hydrodynamic radius (*Å*)
- ``q (float)`` -- electric charge (*e*)
- ``2r`` -- double the hard core radius (*Å*)
- ``e`` -- Lennard-Jones energy (*?*),
- ``m`` -- bead mass (*?*)

.. warning::
    Hard-core radii, masses, charges and Lennard-Jones energies are not yet implemented.

.. warning::
    Bonds between beads are not yet implemented.