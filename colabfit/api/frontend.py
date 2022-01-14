import os
from getpass import getpass
from collections import namedtuple

from flask import Blueprint, request, redirect, flash, render_template
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
from werkzeug.utils import secure_filename

from .forms import UploadForm
from .nav import nav

from ..tools.database import MongoDatabase, load_data
from .resources import (
    CollectionsAPI, DatasetsTable, PropertiesTable, ConfigurationsTable
)

# Prepare DatasetManager
# user = input("mongodb username: ")
# pwrd = getpass("mongodb password: ")

database = MongoDatabase('colabfit_database')
collections = CollectionsAPI(database)

ALLOWED_EXTENSIONS = {'extxyz', 'xyz',}
UPLOAD_FOLDER = './data/uploads'

frontend = Blueprint('frontend', __name__)


nav.register_element('frontend_top', Navbar(
    View('Flask-Bootstrap', '.index'),
    View('Home', '.index'),
    View('Forms Example', '.example_form'),
    View('Debug-Info', 'debug.debug_root'),
    Subgroup(
        'Docs',
        Link('Flask-Bootstrap', 'http://pythonhosted.org/Flask-Bootstrap'),
        Link('Flask-AppConfig', 'https://github.com/mbr/flask-appconfig'),
        Link('Flask-Debug', 'https://github.com/mbr/flask-debug'),
        Separator(),
        Text('Bootstrap'),
        Link('Getting started', 'http://getbootstrap.com/getting-started/'),
        Link('CSS', 'http://getbootstrap.com/css/'),
        Link('Components', 'http://getbootstrap.com/components/'),
        Link('Javascript', 'http://getbootstrap.com/javascript/'),
        Link('Customize', 'http://getbootstrap.com/customize/'), ),
    Text('Using Bootstrap-Flask {}'.format('1.8.0')), ))


@frontend.route('/')
@frontend.route('/index')
def index():
    return "Welcome to the ColabFit database. Try looking at `/collections?collection=datasets`!"


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Shows a long signup form, demonstrating form rendering.
@frontend.route('/publish/', methods=('GET', 'POST'))
def publish():
    form = UploadForm(csrf_enabled=False)

    if request.method == 'POST':
        filename = secure_filename(form.upload.name)
        filename = os.path.join(UPLOAD_FOLDER, filename)

        form.upload.data.save(filename)

        configurations = load_data(
            file_path='/home/jvita/scripts/colabfit/data/gubaev/AlNiTi/train_2nd_stage.cfg',
            file_format='cfg',
            name_field=None,
            elements=['Al', 'Ni', 'Ti'],
            default_name='train_2nd_stage',
            verbose=True,
        )

        # prop_table = PropertiesTable(
        #     [dict(name=p, units='Unspecified') for p in properties],
        #     border=True,
        # )

        co_table = ConfigurationsTable(
            [
                dict(
                    name=co.attributes.colabfit_description[0],
                    elements=', '.join(co.attributes.elements),
                    natoms=co.attributes.nsites,
                    labels=', '.join(co.attributes.colabfit_labels),
                )
                for co in configurations
            ],
            border=True,
        )

        return render_template(
            'publish.html',
            form=form,
            prop_table=prop_table,
            co_table=co_table,
        )

    return render_template('publish.html', form=form)


# @frontend.route('/api/configurations/')
# def api_configurations():

#     # This function was part of an attempt to be able to handle a paginated
#     # table of potentially millions of entries. It isn't ready yet

#     co_cursor = database.configurations.find({})

#     return {
#         'configurations': [
#             {
#                 'id':           co_doc['_id'],
#                 'elements':     co_doc['elements'],
#                 'atoms':        co_doc['nsites'],
#                 'nperiodic':    co_doc['nperiodic_dimensions'],
#                 'names':        co_doc['names'],
#                 'labels':       co_doc['labels'],
#             }
#         for co_doc in co_cursor]
#     }


# @frontend.route('/configurations/')
# def configurations():
#     return render_template(
#         'configurations.html',
#         title='Configurations',
#     )

@frontend.route('/configuration_sets/')
def configuration_sets():

    cs_cursor = database.configuration_sets.find({})

    ConfigurationSetWrapper = namedtuple(
        'ConfigurationSetWrapper',
        [
            'id', 'description', 'configurations', 'atoms',
            'elements', 'labels',
        ]
    )

    configuration_sets = (ConfigurationSetWrapper(
        id=cs_doc['_id'],
        description=cs_doc['description'],
        configurations=cs_doc['aggregated_info']['nconfigurations'],
        atoms=cs_doc['aggregated_info']['nsites'],
        elements=cs_doc['aggregated_info']['elements'],
        labels=cs_doc['aggregated_info']['labels'],
    ) for cs_doc in cs_cursor)

    return render_template(
        'configuration_sets.html',
        title='Configuration Sets',
        configuration_sets=configuration_sets
    )


@frontend.route('/datasets/')
def datasets():

    ds_cursor = database.datasets.find({})

    DatasetWrapper = namedtuple(
        'DatasetWrapper',
        [
            'name', 'authors', 'links', 'elements', 'properties',
            'configurations', 'atoms'
        ]
    )

    datasets = (DatasetWrapper(
        name=ds_doc['name'],
        authors=ds_doc['authors'],
        links=ds_doc['links'],
        elements=ds_doc['aggregated_info']['elements'],
        properties=sum(ds_doc['aggregated_info']['property_types_counts']),
        configurations=ds_doc['aggregated_info']['nconfigurations'],
        atoms=ds_doc['aggregated_info']['nsites'],
    ) for ds_doc in ds_cursor)

    return render_template(
        'datasets.html',
        title='Datasets',
        datasets=datasets
    )


@frontend.route('/query/')
def query():
    return collections.get()
