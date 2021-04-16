pyBrown
========

Overview
---------

``pyBrown`` is a Brownian and Stokesian dynamics package and simulation tool. It is written in mainly in python, with the most demanding computationally functions written in C.

Quickstart
-----------

To configure, type the following command in the command line:

.. code-block:: console

    $ ./configure --prefix=DIR --with-lapack=LAPACK_LIBS

where ``DIR`` is the installation directory (``/usr/local`` by default) and ``LAPACK_LIBS`` is lapack libraries to use (e.g. ``--with-lapack="-l lapack"``).

Then to compile and install proceed with:

.. code-block:: console

    $ make
    $ make install

If you want to run unit tests, go to a project directory ``src/bdsim/tests`` and type:

.. code-block:: console

    $ make test

.. warning::
    The work on ``pyBrown`` is still in progress. Some functionalities may be temporally unavailable.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting_started/installation
   getting_started/preparing_input

.. toctree::
   :maxdepth: 1
   :caption: Reference

   reference/authors
   reference/license

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
