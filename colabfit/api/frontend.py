import os
from xml.dom.expatbuilder import parseString
import markdown
from getpass import getpass
from ast import literal_eval
from collections import namedtuple
from tkinter import W

from flask import Blueprint, request, render_template, send_from_directory
from flask_nav.elements import Navbar, View, Subgroup, Link, Text, Separator
from werkzeug.utils import secure_filename

from .forms import UploadForm, QueryForm
from .nav import nav

import kim_edn
from kim_property.definition import PROPERTY_ID as VALID_KIM_ID

from ..tools.database import MongoDatabase, load_data
from ..tools.property_settings import PropertySettings
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
    full_form = UploadForm(csrf_enabled=False)

    if request.method == 'POST':

        tmp = os.path.join(UPLOAD_FOLDER, full_form.name.data)
        if os.path.isdir(tmp):
            base_folder = tmp
            # counter = 1
            # base_folder = '{}({})'.format(tmp, counter)

            # while os.path.isdir(base_folder):
            #     counter += 1
            #     base_folder = '{}({})'.format(tmp, counter)
        else:
            base_folder = tmp
            os.mkdir(base_folder)

        data_filenames = []
        if full_form.data_upload.data:
            for f in full_form.data_upload.data:
                if not f.filename:
                    continue

                filename = secure_filename(f.filename)
                filename = os.path.join(base_folder, filename)

                data_filenames.append(filename)

                f.save(filename)

        definition_files = {}
        if full_form.definitions_upload.data:
            for f in full_form.definitions_upload.data:
                if not f.filename:
                    continue

                filename = secure_filename(f.filename)
                filename = os.path.join(base_folder, filename)

                f.save(filename)

                definition = kim_edn.load(filename)['definition']
                definition_files[definition['property-id']] = filename

        property_map = {}
        property_settings = {}
        configuration_sets = []
        configuration_labels = []

        pmc_pname = None
        psc_pname = None
        for k, v in request.form.items():
            print(k, v)
            if 'pmc0' in k:
                pmc_pname = v
                pid_dict = property_map.setdefault(pmc_pname, {})
                print('PNAME CHANGE:', pmc_pname)
            elif 'pmc1' in k:
                pmc_field = v
                field_dict = pid_dict.setdefault(pmc_field, {})
                print('FIELD CHANGE:', pmc_field)
            elif 'pmc2' in k:
                field_dict['field'] = v
            elif 'pmc3' in k:
                pmc_units = v

                if pmc_units in ['None', '']:
                    pmc_units = None

                field_dict['units'] = v

            elif 'psc0' in k:
                psc_pname = v
                pso_dict = property_settings.setdefault(psc_pname, {})
            elif 'psc1' in k:
                pso_dict['method'] = v
            elif 'psc2' in k:
                pso_dict['description'] = v
            elif 'psc3' in k:
                pso_dict['labels'] = [
                    _.strip() for _ in v.split(',')
                ]
            elif 'psc4' in k:
                # The file upload doesn't work yet...
                pso_dict['files'] = None

            elif 'csc0' in k:
                cs_query = literal_eval(v)

                # If a string is passed, does regex matching on names
                if isinstance(cs_query, str):
                    cs_query = {'names': {'$regex': cs_query}}
            elif 'csc1' in k:
                cs_desc = v

                configuration_sets.append((cs_query, cs_desc))
            elif 'clc0' in k:
                cl_query = literal_eval(v)

                # If a string is passed, does regex matching on names
                if isinstance(cl_query, str):
                    cl_query = {'names': {'$regex': cl_query}}
            elif 'clc1' in k:
                cl_labels = [_.strip() for _ in v.split(',')]
                configuration_labels.append((cl_query, cl_labels))

        # Build PSO objects
        for pname, pdict in property_settings.items():
            property_settings[pname] = PropertySettings(
                method=pdict['method'],
                description=pdict['description'],
                labels=pdict['labels'],
                files=pdict['files'] if 'files' in pdict else None
            )

        print('PROPERTY MAP:', property_map)
        print('PROPERTY SETTINGS:', property_settings)
        print('CONFIGURATION SETS:', configuration_sets)
        print('CONFIGURATION LABELS:', configuration_labels)

        # configurations = load_data(
        #     file_path='/home/jvita/scripts/colabfit/data/gubaev/AlNiTi/train_2nd_stage.cfg',
        #     file_format='cfg',
        #     name_field=None,
        #     elements=['Al', 'Ni', 'Ti'],
        #     default_name='train_2nd_stage',
        #     verbose=True,
        # )

        # co_table = ConfigurationsTable(
        #     [
        #         dict(
        #             name=co.info['_name'],
        #             elements=sorted(list(set(co.get_chemical_symbols()))),
        #             natoms=len(co),
        #             labels=co.info['_labels']
        #         )
        #         for co in configurations
        #     ],
        #     border=True,
        # )

        template = \
"""
# Name

{}

# Authors

{}

# Links

{}

# Description

{}

# Storage format

|Elements|File|Format|Name field|
|---|---|---|---|
{}

# Properties

|Property|KIM field|ASE field|Units
|---|---|---|---|
{}

# Property settings

|Method|Description|Labels|Files|
|---|---|---|---|
{}

# Configuration sets

|Query|Description|
|---|---|
{}

# Configuration labels

|Query|Labels|
|---|---|
{}
"""

        if full_form.config_name_field.data:
            config_name = full_form.config_name_field.data
        else:
            config_name = '_name'

        formatting_arguments = []

        formatting_arguments.append(full_form.name.data)
        formatting_arguments.append(full_form.authors.data)
        formatting_arguments.append(full_form.links.data)
        formatting_arguments.append(full_form.description.data)

        # Storage format table
        formatting_arguments.append('\n'.join(
            '| {} | {} | {} | {} |'.format(
                full_form.elements.data,
                data_fname,
                'xyz',
                config_name,
            )
            for data_fname in data_filenames
        ))

        # Properties table
        tmp = []
        for pid, fdict in property_map.items():
            for f,v in fdict.items():
                if VALID_KIM_ID.match(pid) is None:
                    spoofed = 'tag:@,0000-00-00:property/' + pid
                else:
                    spoofed = pid

                tmp.append(
                    '| {} | {} | {} | {}'.format(
                        '[{}]({})'.format(pid, definition_files[spoofed]),
                        f,
                        v['field'],
                        v['units']
                    )
                )

        formatting_arguments.append('\n'.join(tmp))

        # Properties settings table
        tmp = []
        for pso in property_settings.values():
            tmp.append('| {} | {} | {} | {} |'.format(
                pso.method,
                pso.description,
                ', '.join(pso.labels),
                ', '.join('[{}]({})'.format(f, f) for f in pso.files)
            ))

        formatting_arguments.append('\n'.join(tmp))

        # ConfigurationSets table
        tmp = []
        for (cs_query, cs_desc) in configuration_sets:
            tmp.append('| `{}` | {} |'.format(
                cs_query,
                cs_desc,
            ))

        formatting_arguments.append('\n'.join(tmp))

        # Configuration labels table
        tmp = []
        for (cl_query, cl_labels) in configuration_labels:
            tmp.append('| `{}` | {} |'.format(cl_query, ', '.join(cl_labels)))

        formatting_arguments.append('\n'.join(tmp))

        html_file_name = os.path.join(base_folder, 'README.md')
        with open(html_file_name, 'w') as html:
            html.write(template.format(*formatting_arguments))

        return send_from_directory(base_folder, 'README.md')

    return render_template(
        'publish.html',
        full_form=full_form,
    )


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


# @frontend.route('/datasets/')
# def datasets():

#     ds_cursor = database.datasets.find({})

#     DatasetWrapper = namedtuple(
#         'DatasetWrapper',
#         [
#             'name', 'authors', 'links', 'elements', 'properties',
#             'configurations', 'atoms'
#         ]
#     )

#     datasets = (DatasetWrapper(
#         name=ds_doc['name'],
#         authors=ds_doc['authors'],
#         links=ds_doc['links'],
#         elements=ds_doc['aggregated_info']['elements'],
#         properties=sum(ds_doc['aggregated_info']['property_types_counts']),
#         configurations=ds_doc['aggregated_info']['nconfigurations'],
#         atoms=ds_doc['aggregated_info']['nsites'],
#     ) for ds_doc in ds_cursor)

#     return render_template(
#         'datasets.html',
#         title='Datasets',
#         datasets=datasets
#     )


@frontend.route('/query/', methods=('GET', 'POST'))
def query():
    form = QueryForm()

    if request.method == 'POST':
        # TODO: maybe configurations could return more info
        """
        count | labels | elements (ratios) | nsites |

        Also add ratios to CSs
        """
        if form.query.data:
            query = literal_eval(form.query.data)
        else:
            query = {}

        collection = getattr(database, form.collection.data)

        if form.collection.data  == 'configurations':
            extra = {
                'nconfigurations': collection.count_documents(query),
                'nsites':
                    next(collection.aggregate([
                        {'$match': query},
                        {'$group': {'_id': None, 'sum': {'$sum': '$nsites'}}},
                    ]))['sum'],
                'elements': collection.distinct('elements', query),
                'labels':   collection.distinct('labels', query),
            }
        elif form.collection.data  == 'properties':
            extra = {
                'nproperties':  collection.count_documents(query),
                'types':        collection.distinct('type', query),
                'methods':      collection.distinct('methods', query),
                'labels':       collection.distinct('labels', query),
            }
        else:
            extra = None

        return render_template(
            'query.html',
            form=form,
            collection=form.collection.data,
            data=collection.find(query)
                if form.collection.data in ['datasets', 'configuration_sets']
                else collection.count_documents(query),
            extra=extra,
        )


    return render_template(
        'query.html',
        form=form,
    )

