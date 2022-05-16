pyBrown
========

    *These motions were such as to satisfy me ... that they arose neither from currents in the fluid, nor from its gradual evaporation, but belonged to the particle itself*
    - `Robert Brown <https://en.wikipedia.org/wiki/Robert_Brown_(botanist,_born_1773)>`_

.. rubric:: Overview

``pyBrown`` is a Brownian and Stokesian dynamics package and simulation tool. It is written mainly in ``python 3``, with the most computationally demanding functions written in ``C``. Brownian and Stokesian dynamics are computational methods to simulate systems driven by thermal fluctuations. Hydrodynamic interactions are taken into account using far-field *Rotne-Prager-Yamakawa* approximation but correcting for near-field *lubrication* effects is available as well.

.. rubric:: Quickstart

To configure, type the following command in the command line:

.. code-block:: console

    $ ./configure --prefix=DIR --with-lapack=LAPACK_LIBS

where ``DIR`` is the installation directory (``/usr/local/`` by default) and ``LAPACK_LIBS`` is lapack libraries to use (e.g. ``--with-lapack="-llapack"``).

Then to compile and install proceed with:

.. code-block:: console

    $ make
    $ make install

You can check :ref:`installation` for more details.

To ensure that all ``python`` packages needed by ``pyBrown`` are present on your computer you can run

.. code-block:: console

    $ pip3 install -r requirements.txt

If you want to run unit tests, go to a project directory ``src/bdsim/tests/`` and type:

.. code-block:: console

    $ make test

To run simulation, prepare ``.str`` and ``.json`` input files (see :ref:`preparing-input`) and type:

.. code-block:: console

    $ BD.py INPUT_JSON_FILENAME

where instead of ``INPUT_JSON_FILENAME`` substitute the name of your input ``.json`` file.

Provided that you have the restart ``.rst`` file and the trajectory ``.xyz``, you can restart the simulation from the last save typing:

.. code-block:: console

    $ BD-restart.py RESTART_FILENAME

where instead of ``RESTART_FILENAME`` substitute the name of your restart file. For more details regarding running simulations check :ref:`running-simulation`.

.. _units:

.. rubric:: Units

+-------------------+----------------------------+
| Physical property |             Unit           |
+===================+============================+
|    Temperature    |         kelvin (*K*)       |
+-------------------+----------------------------+
|     Viscosity     |          poise (*P*)       |
+-------------------+----------------------------+
|       Time        |       picosecond (*ps*)    |
+-------------------+----------------------------+
|      Distance     |        angstrom (*Å*)      |
+-------------------+----------------------------+

You can choose the energy units by ``"energy_unit"`` keyword in input ``.json`` file. Default is "joule" (*J*), other options are "kcal/mol" (kilocalorie per mol, *kcal/mol*) and "eV" (electronvolt, *eV*). Force unit is then energy unit per angstrom.

.. warning::
    The work on ``pyBrown`` is still in progress. Some functionalities may be temporally unavailable.

.. todo::
    - propagation schemes different than Ermak and Midpoint,
    - predefined 1-body force and energy,
    - dihedral bonded interactions

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/installation
   getting_started/preparing_input
   getting_started/running_simulation
   getting_started/custom_potentials

.. toctree::
   :maxdepth: 1
   :caption: API

   modules/bead
   modules/box
   modules/diffusion
   modules/input
   modules/interactions
   modules/output
   modules/reactions

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/authors
   reference/license

.. rubric:: Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
