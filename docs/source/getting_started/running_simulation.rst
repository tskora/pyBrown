Running simulation
-------------------

Brownian/Stokesian dynamics using ``BD.py``
*********************************************

To run the simulations defined in ``inp.json`` on a bead system defined in ``inp.str`` (and addressed in  ``inp.json``)

.. code-block:: console

    $ BD.py inp.json

Restarting simulation using ``BD-restart.py``
**********************************************

It is often the case that simulation is interrupted. ``pyBrown`` ensures that the hours or days of computing are not lost. With a frequency defined by user in ``inp.json`` (keyword ``"rst_write_freq"``) ``BD.py`` pickles its state to the file again defined by user in ``inp.json`` (keyword ``"rst_Write_freq"``). You can restart the simulation in a following way:

.. code-block:: console

    $ BD-restart.py my_restart.rst

.. note::
    If you are really unlucky, the simulation may be interrupted in the very moment of writing restart. To ensure that the whole simulation will not be damaged, extra copy is made everytime restart file is created. it has exactly the same filename apart of extra ``2`` character at the end. For instance, if restart file has a name ``my_restart.rst``, backup restart file has a name ``my_restart.rst2``. If your primary restart file does not work, try using the backup one.

Using input patch
*******************

Sometimes it is useful to change some values provided in initial ``.json`` when restarting the simulation. It is very common to do that with ``"number_of_steps"`` keyword. To do that, you can provide additional optional ``.json`` file to ``BD-restart.py``.

.. code-block:: console

    $ BD-restart.py my_restart.rst inp_patch.json

In ``inp_patch.json`` provide only those ``keyword: value`` pairs which you want to change. The rest will stay as it was configured in initial ``inp.json`` file.

.. warning::
    In text above it is assumed that the ``pyBrown`` was properly installed. If not, you can still use it but always providing the path (absolute or relative) to the ``BD.py`` or ``BD-restart.py`` script, eg.

    .. code-block:: console

        $ /path/to/script/BD.py inp.json