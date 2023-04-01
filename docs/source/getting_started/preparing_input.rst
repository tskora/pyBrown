.. _preparing-input:

Preparing input
----------------

Input for ``pyBrown`` consists of two text files:

- ``.str`` -- providing initial configuration of beads and their properties,
- ``.json`` -- providing simulation parameters.

Below we ellaborate on their structure and meaning.

.. note::
    Consult :ref:`units` used in ``pyBrown`` before writing ``.str`` and ``.json`` files.

``.str`` file
**************

Structure (``.str``) file contains the information about the physical system under study. First and foremost, It introduces initial positions of beads and their sizes (both hydrodynamic and hard-core). Moreover, it contains information about masses qnd charges of beads. Last but not least, it encodes bonded and nonbonded interaction parameters.

Structure file consists of bead-defining lines starting with ``sub``, ``bond`` or ``angle``. Their structure is discussed below.

Subunit
^^^^^^^^

``sub label index x y z a q 2r ε m``

Only the ``sub`` string has to be identical in every bead-defining line. All the other entries should be replaced with values specific to a given type of bead. Below their meanings are explained:

- ``label`` -- label (name) of the bead (``string``)
- ``index`` -- an unique number assigned to the bead (``int``)
- ``x y z`` -- ``x``, ``y`` and ``z`` cartesian coordinates of the bead center (``float``, *Å*)
- ``a`` -- the hydrodynamic radius of the bead (``float``, *Å*)
- ``q`` -- the electric charge of the bead (``float``, *e*)
- ``2r`` -- double the hard-core (Lennard-Jones) radius (``float``, *Å*)
- ``ε`` -- Lennard-Jones well depth (``float``, *?*)
- ``m`` -- the mass of the bead (``float``, *?*)

.. todo::
    Masses and charges are not yet consumed by ``pyBrown`` -- change it.

Bond
^^^^^

``bond index1 index2 r0 rmax k``

Only the ``bond`` string has to be identical in every bond-defining line. All the other entries should be replaced with values specific to a given pair of beads. Below their meanings are explained:

- ``index1 index2`` -- indices of the beads connected with the bond (see above) (``int``)
- ``r0`` -- equilibrium distance between the beads (``float``, *Å*)
- ``rmax`` -- maximal possible distance between the beads (``float``, *Å*) **NOT IMPLEMENTED YET, just write any number**
- ``k`` -- force constant of the bond (``float``, *? Å*:sup:`-2`)

For each bonded pair the energy is incremented by a harmonic term:

.. math::
    E = \frac{1}{2} k ( r - r_0 )^2 .

.. todo::
    For now only harmonic bond works. In future implement more general bonded potential allowing for setting the max bond lengths.

Angle
^^^^^^

``angle angle index1 index2 index3 φ0 k``

Only the ``angle angle`` string has to be identical in every angle-defining line. All the other entries should be replaced with values specific to a given triple of beads. Below their meanings are explained:

- ``index1 index2 index3`` -- indices of the beads defining the angle (see above) (``int``)
- ``φ0`` -- equilibrium distance between the beads (``float``, :math:`^\circ`)
- ``k`` -- force constant of the angle potential (``float``, *? rad*:sup:`-2`)

For each angle the energy is incremented by a harmonic term:

.. math::
    E = \frac{1}{2} k ( \phi - \phi_0 )^2 .

.. todo::
    Implement cos angle potential *via* ``angle cos`` lines.

.. todo::
    Implement definition of dihedral potential *via* ``dihe`` lines.

``.json`` file for ``BD.py``
*****************************

Simulation parameters are provided in a standard `JSON <https://www.json.org/json-en.html>`_ data format. It consists of multiple lines in a form of ``keyword: value`` pairs enclosed by a curly bracket. Some keywords are obligatory, but a majority is not -- for them default values will be loaded, if needed. ``pyBrown`` will inform you if it does not recognize some keywords in input ``.json`` file.

The complete list of keywords is provided below.

Input/Output
^^^^^^^^^^^^^

- ``"input_str_filename": string`` -- the name of the input ``.str`` file (*see above*), **required**
- ``"output_xyz_filename": string`` -- the name of the output ``.xyz`` file to which ``pyBrown`` writes the trajectory, **required**
- ``"output_enr_filename": string`` -- the name of the output ``.enr`` file to which ``pyBrown`` writes the energy
- ``"output_rst_filename": string`` -- the name of the binary ``.rst`` file from which``pyBrown`` can restart simulation
- ``"filename_range": [int, int]`` -- provided that all your input and output files have names of a following form: ``"name_{}.str``, ``name_{}.xyz`` etc., ``pyBrown`` will iteratively substitute numbers from filename range instead of ``{}``, resulting with sequence of runs for consecutive jobs,

.. note::
    
    If for example input file looks in a following way:
    
    .. code:: JSON

        {
            "input_str_filename": "test_{}.str",
            "output_xyz_filename": "test_{}.xyz",
            "output_rst_filename": "test_{}.rst",
            "filename_range": [1, 3],
            "some_other_keywords": "some values"
        }

    then ``pyBrown`` will load ``test_1.str``, run the simulation writing restart to ``test_1.rst`` and trajectory to ``test_1.xyz``. After completing, it will load ``test_2.str`` and write to ``test_2.rst`` and ``test_2.xyz``. After completing the second job, ``pyBrown`` will end the run. If we swap "filename_range" value for ``[1, 11]`` 10 jobs will be ran consecutively.

.. warning::

    If you use ``"filename_range"`` keyword and manually set ``"seed"`` keyword, seed will be **the same** for all jobs.

- ``"xyz_write_freq": int`` -- the frequency of writing to the ``.xyz`` file (every ... timesteps), default: ``1``
- ``"enr_write_freq": int`` -- the frequency of writing to the ``.enr`` file (every ... timesteps)
- ``"rst_write_freq": int`` -- the frequency of writing to the ``.rst`` file (every ... timesteps)

- ``"debug": boolean`` -- switching on/off the debug printout, default: ``false``
- ``"verbose": boolean`` -- switching on/off the verbose printout, default: ``false``
- ``"progress_bar": boolean`` -- switching on/off the progress bar, default: ``false``

Simulation box
^^^^^^^^^^^^^^^

- ``"box_length": float`` -- length of the cubic simulation box (*Å*), **required**

.. todo::
    Turning on and off periodic boundary conditions would be nice.

- ``"ewald_alpha": float`` -- parameter controling the convergence of Ewald summation, default: ``np.sqrt(np.pi)``
- ``"ewald_real": int`` -- the maximal magnitude of the real lattice vectors in Ewald summation of the diffusion tensor, default: ``0``
- ``"ewald_imag": int`` -- the maximal magnitude of the reciprocal lattice vectors in Ewald summation of the diffusion tensor, default: ``0``

Forces
^^^^^^^

- ``"energy_unit": string`` -- units in which energy and force (in case of force it is that unit per angstrom) are provided in input files, options: ``"joule"``, ``"kcal/mol"``, ``"eV"``, default: ``"joule"``
- ``"lennard_jones_6": bool`` -- switching on/off the Lennard-Jones :math:`\propto r^{-6}` attraction between beads (multiplicative coefficients and Lennard-Jones radii of every bead are defined in ``.str`` input file, see :ref:`preparing-input`), default: ``false``
- ``"lennard_jones_12": bool`` -- switching on/off the Lennard-Jones :math:`\propto r^{-12}` repulsion between beads (multiplicative coefficients and Lennard-Jones radii of every bead are defined in ``.str`` input file, see :ref:`preparing-input`), default: ``false``
- ``"lennard_jones_alpha": float`` -- Lennard-Jones interaction scaling, if set to ``4.0``, ``ε`` parameter from ``.str`` is equal to depth of combined LJ6-LJ12 potencial, default: ``4.0``
- ``"dlvo": bool`` -- switching on/off DLVO screened electrostatic interactions between beads (charges of every bead are defined in ``.str`` input file, see :ref:`preparing-input`), default: ``false``
- ``"dielectric_constant": float`` -- dielectric constant relative to vacuum, default: ``78.54``,
- ``"inverse_debye_length": float`` -- inverse Debye length (*Å^-1*) responsible for exponential screening of electrostatic interactions, default: ``0.1``
- ``"custom_interactions": bool`` -- switching on/off reading of energy and force from custom external file, default: ``false``
- ``"custom_interactions_filename": string`` -- the name of the input ``.py`` file from which custom expressions for energy and force are loaded
- ``"auxiliary_custom_interactions_keywords": {}`` -- extra parameters for the custom energy and force
- ``"external_force": [float, float, float]`` -- external force experienced universally by all beads, default: ``[0.0, 0.0, 0.0]``

Propagation
^^^^^^^^^^^^

- ``"dt": float`` -- the timestep (*ps*), **required**
- ``"number_of_steps": int`` -- the total number of simulation steps, **required**
- ``"propagation_scheme": option`` -- propagation algorithm for the trajectory generation, options: ``"ermak"``, ``"midpoint"``, default: ``"ermak"``
- ``"m_midpoint": int`` -- inverse of a fraction of the time step made in a prediction part of midpoint algorithm (setting to ``2`` means that half of a time step will be made in a prediction part), default: ``100``
- ``"check_overlaps": boolean`` -- whether to check overlaps in every simulation step, default: ``true``
- ``"overlap_treshold": float`` -- how small distance is treated as overlap, default: ``0.0``
- ``"overlap_radius": string`` -- which radius decides that there is an overlap, options: ``hydrodynamic`` or ``hard_core`` (Lennard-Jones), default: ``"hydrodynamic"``

.. warning::

    If you turn on lubrication interactions, ``"overlap_treshold"`` should be slightly larger than ``0.0`` because small separations will lead to very small eigenvalues of diffusion matrix, and consequently to breakdown of the Choleski decomposition.

- ``"max_move_attempts": int`` -- maximal number of move attempts, if exceeded ``pyBrown`` will stop, default: ``1000000``
- ``"immobile_labels": [string, ..., string]`` -- label of beads which are to be immobile in simulation, default: ``[]``
- ``"seed": int`` -- seed for pseudorandom number generation algorithm, default ``np.random.randint(2**32 - 1)``

Hydrodynamic interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``"hydrodynamics": option`` -- the method used to compute diffusion tensor, options: ``"nohi"``, ``"rpy"``, ``"rpy_smith"``, ``"rpy_lub"``, ``rpy_smith_lub``, default: ``"nohi"``
- ``"m_midpoint": int`` -- the inverse of the timestep fraction made in the first stage of the midpoint propagation scheme
- ``"diff_freq": int`` -- the frequency of computing far field diffusion tensor (every ... timesteps), default: ``1``
- ``"lub_freq": int`` -- the frequency of computing near field resistance tensor and total diffusion matrix (every ... timesteps), default: ``1``
- ``"chol_freq": int`` -- the frequency of performing Cholesky decomposition of diffusion tensor (every ... timesteps), default: ``1``
- ``"lubrication_cutoff": float`` -- cutoff for lubrication interactions expressed as a ratio of distance between bead surfaces and a sum of their hydrodynamic radii (:math:`s_\mathrm{cutoff} = \frac{r_{ij} - a_i - a_j}{a_i + a_j}`), default: ``1``

.. note::

    Setting ``"lubrication_cutoff"`` to ``2`` means that lubrication correction won't be calculated for beads with surfaces separated by a distance equal to double of the sum of their hydrodynamic radii.

- ``"cichocki_correction": bool`` -- switching on/off the operation of removing collective movements of bead pairs from the lubrication correction, default: ''true''


Physical conditions
^^^^^^^^^^^^^^^^^^^^

- ``"T": float`` -- temperature (*K*), **required**
- ``"viscosity": float`` -- viscosity (*P*), **required**

Keyword blocks
^^^^^^^^^^^^^^^

Some more specific options are activated by keywords which are of ``JSON`` structure themselves. Such a keyword simultaneously turns on some functionality and specifies all the additional parameters regarding that functionality.

- ``"external_force_region": {...}`` -- restrict external force to the selected region of the box
   - ``"x": [float, float]`` -- ``x`` range defining the region
   - ``"y": [float, float]`` -- ``y`` range defining the region
   - ``"z": [float, float]`` -- ``z`` range defining the region

- ``"measure_flux": {...}`` -- measure the flux through a defined plane
   - ``"flux_normal": [float, float, float]`` -- normal to the plane (defines the direction of positive flux)
   - ``"flux_plane_point": [float, float, float]`` -- any point on the plane,
   - ``"output_flux_filename": string`` -- the name of the output ``.flx`` file to which ``pyBrown`` writes the net flux

- ``"measure_concentration": {...}`` -- measure the concentration in a selected region
   - ``"x": [float, float]`` -- ``x`` range defining the region
   - ``"y": [float, float]`` -- ``y`` range defining the region
   - ``"z": [float, float]`` -- ``z`` range defining the region
   - ``"output_concentration_filename"`` -- the name of the output ``.con`` file to which ``pyBrown`` writes the concentration in selected region

``.json`` file for ``BD-NAM.py``
*********************************
