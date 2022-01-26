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

For installation without :code:`sudo` or :code:`apt-get`, it is suggested to use
:code:`conda``:

.. code-block:: shell

    # Install using conda
    conda install -c conda-forge mongodb

    # Start the server without using sudo or systemctl
    mongod --dbpath <path_to_folder_for_storing_mongo_data>

To confirm that this has been set up correctly, try opening a connection:

.. code-block:: python

    database = MongoDatabase('test')

The logs from the `mongod` command should show the new connection, with output
looking something like this:

.. code-block:: shell

    2022-01-20T10:41:14.785-0600 I NETWORK  [conn1] received client metadata from 127.0.0.1:59890 conn1: { driver: { name: "PyMongo", version: "4.0.1" }, os: { type: "Linux", name: "Linux", architecture: "ppc64le", version: "4.18.0-305.3.1.el8_4.ppc64le" }, platform: "CPython 3.7.10.final.0" }

**Note:** in order for the :class:`~colabfit.tools.database.MongoDatabase` to be
able to access the Mongo server, it must be able to open an SSH connection to
the machine where the :code:`mongod` command was run from. Refer to the `PyMongo
documentation <https://pymongo.readthedocs.io/en/stable/tutorial.html>`_ for
more details regarding setting up a connection to the Mongo server.

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
