Installation
------------

.. note::
   We recommend installing ``fancy_gym`` into a virtual environment as
   provided by `venv <https://docs.python.org/3/library/venv.html>`__. 3rd
   party alternatives to venv like `Poetry <https://python-poetry.org/>`__
   or `Conda <https://docs.conda.io/en/latest/>`__ can also be used.

Installation from PyPI (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ``fancy_gym`` via

.. code:: bash

   pip install fancy_gym

We have a few optional dependencies. If you also want to install those
use

.. code:: bash

   # to install all optional dependencies
   pip install 'fancy_gym[all]'

   # or choose only those you want
   pip install 'fancy_gym[dmc,box2d,mujoco-legacy,jax,testing]'

Pip can not automatically install up-to-date versions of metaworld,
since they are not avaible on PyPI yet. Install metaworld via

.. code:: bash

   pip install metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@d155d0051630bb365ea6a824e02c66c068947439#egg=metaworld

Installation from master
~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository

.. code:: bash

   git clone git@github.com:ALRhub/fancy_gym.git

2. Go to the folder

.. code:: bash

   cd fancy_gym

3. Install with

.. code:: bash

   pip install -e .

We have a few optional dependencies. If you also want to install those
use

.. code:: bash

   # to install all optional dependencies
   pip install -e '.[all]'

   # or choose only those you want
   pip install -e '.[dmc,box2d,mujoco-legacy,jax,testing]'

Metaworld has to be installed manually with

.. code:: bash

   pip install metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@d155d0051630bb365ea6a824e02c66c068947439#egg=metaworld