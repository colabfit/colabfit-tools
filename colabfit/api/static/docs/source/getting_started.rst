===============
Getting started
===============

Installing colabfit-tools
=========================

Currently, installation is only supported using :code:`pip` to install directly
from the private GitHub repository.

Using pip
^^^^^^^^^

.. code-block:: console

    $ pip install git+https://<PAT>@github.com/colabfit/colabfit-tools.git

Note that since :code:`colabfit-tools` is currently still a private project,
:code:`<PAT>` must either be your a Personal Access Token that has appropriate
permissions.

Installing Mongo
================

See the `official MongoDB documentation
<https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/>`_ for
complete installation instructions. For convenience, the shell commands to
download, install, and start a Mongo server have been included below (for Ubuntu
20.04).

.. code-block:: shell

    # Instructions copied from MongoDB setup tutorial
    wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -
    echo "deb [ arch=amd64,arm64  ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
    sudo apt-get update
    sudo apt-get install -y mongodb-org
    sudo systemctl start mongod
    sudo systemctl status mongod

To enable access control (user/password authentication), see `the following
section of the MongoDB documentation
<https://docs.mongodb.com/manual/tutorial/enable-authentication/>`_.

First steps
===========

Start your local Mongo server and confirm that it's running.

.. code-block:: console

    $ sudo systemctl start mongod
    $ sudo systemctl status mongod

Open a connection to the Mongo server from inside your Python script.

.. code-block:: python

    from colabfit.tools.database import MongoDatabase

    client = MongoDatabase('my_database')

Build a Configuration just like you would build an `ASE Atoms object
<https://wiki.fysik.dtu.dk/ase/ase/atoms.html>`_

.. code-block:: python

    import numpy as np
    from colabfit.tools.configuration import Configuration

    atoms = Configuration(symbols='H2O', positions=np.random.random((3, 3)))

And finally, add the Configuration into the Database. Note that this command
will not work if you haven't first :ref:`installed Mongo <Installing Mongo>`.

.. code-block:: python

    client.insert_data(
        [atoms],
        generator=False,
        verbose=True
    )

Use :code:`mongosh` for external verification that the data was added to your local
database.

.. code-block:: console
    
    # In a Mongo terminal opened using the `mongosh` command-line-tool
    $ show dbs
    $ use my_database
    $ my_database.configurations.findOne()

Next steps
==========

* Take a look at the :ref:`Overview` to see how the Database is structured.
* Review the :ref:`Basics of Configurations` to better understand how data is
  stored when it is first loaded in.
* Follow the :ref:`Basic example`
* Continue with the :ref:`QM9 example` and/or the :ref:`Si PRX GAP example`
* Look at :ref:`Mongo usage` if you are unfamiliar with Mongo syntax.