pyBrown
========

    *These motions were such as to satisfy me ... that they arose neither from currents in the fluid, nor from its gradual evaporation, but belonged to the particle itself*
    - `Robert Brown <https://en.wikipedia.org/wiki/Robert_Brown_(botanist,_born_1773)>`_

.. rubric:: Overview

``pyBrown`` is a Brownian and Stokesian dynamics package and simulation tool. It is written mainly in ``python 3``, with the most computationally demanding functions written in ``C``. Brownian and Stokesian dynamics are computational methods to simulate systems driven by thermal fluctuations. Hydrodynamic interactions are taken into account using far-field *Rotne-Prager-Yamakawa* approximation but correcting for near-field lubrication effects is available as well.

.. rubric:: Quickstart

To configure, type the following command in the command line:

.. code-block:: console

    $ ./configure --prefix=DIR --with-lapack=LAPACK_LIBS

where ``DIR`` is the installation directory (``/usr/local/`` by default) and ``LAPACK_LIBS`` is lapack libraries to use (e.g. ``--with-lapack="-llapack"``).

Then to compile and install proceed with:

.. code-block:: console

    $ make
    $ make install

To ensure that all ``python`` packages needed by ``pyBrown`` are present on your computer you can run

.. code-block:: console

    $ pip3 install -r requirements.txt

If you want to run unit tests, go to a project directory ``src/bdsim/tests/`` and type:

.. code-block:: console

    $ make test

To run simulation, prepare ``.str`` and ``.json`` input files and type:

.. code-block:: console

    $ BD.py INPUT_JSON_FILENAME

where instead of ``INPUT_JSON_FILENAME`` substitute the name of your input ``.json`` file.

If you want to restart your simulation, provided that you have restart file, type:

.. code-block:: console

    $ BD-restart.py RESTART_FILENAME

where instead of ``RESTART_FILENAME`` substitute the name of your restart file.

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
|       Force       | joule per angstrom (*J/Å*) |
+-------------------+----------------------------+

.. warning::
    The work on ``pyBrown`` is still in progress. Some functionalities may be temporally unavailable.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/installation
   getting_started/preparing_input
   getting_started/running_simulation

.. toctree::
   :maxdepth: 1
   :caption: API

   modules/bead
   modules/box
   modules/diffusion
   modules/input
   modules/output

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/authors
   reference/license

.. rubric:: Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
