import os
import ast
import typing
from flask.templating import render_template
from pymongo import MongoClient

from flask import jsonify, request
from flask_restful import Resource
from flask_table import Table, Col

from colabfit import (
    _DATASETS_COLLECTION, _CONFIGSETS_COLLECTION,
    _PROPS_COLLECTION, _CONFIGS_COLLECTION
)
from colabfit.tools.dataset import Dataset
from colabfit.tools.configuration import Configuration
from colabfit.tools.property import Property
from colabfit.tools.configuration_set import ConfigurationSet

class CollectionsAPI(Resource):

    resource_class = {
        _DATASETS_COLLECTION: Dataset,
        _CONFIGSETS_COLLECTION: ConfigurationSet,
        _PROPS_COLLECTION: Property,
        _CONFIGS_COLLECTION: Configuration
    }

    def __init__(self, database):
        self.database = database

    def get(self):
        offset = 0
        limit = 10
        collection = None

        # elements, labels, chemical_systems, ...
        query = {}
        for arg, val in request.args.items():

            if arg == 'collection':
                collection = val
            elif arg == 'offset':
                offset = int(val)
            elif arg == 'limit':
                limit = int(val)
            elif arg == 'fields':
                continue
            elif arg == 'regex':
                continue
            elif arg == 'in':
                continue
            else:
                print(val)
                sub = ast.literal_eval(val)

                if ('regex' in request.args) and ('in' in request.args):
                    raise RuntimeError(
                        'Only one of `regex` or `in` can be specified.'
                    )

                if 'regex' in request.args:
                    sub = {'$regex': sub}
                elif 'in' in request.args:
                    sub = {'$in': sub}

                query[arg] = sub

        if collection is None:
            return 'Must specify a collection using ?collection='

        fields = request.args.getlist('fields')
        if len(fields) > 0:
            fields = {key: 1 for key in fields}
        else:
            fields = None

        collection_object = getattr(self.database, collection)
        print(collection_object)

        results = collection_object.find(
            query, fields
        ).skip(offset).limit(limit).sort('_id')

        if fields is None:
            results = [
                self.resource_class[collection](**_).dict() for _ in results
            ]
        else:
            results = list(results)

        return jsonify({'results': results, 'prev_url': '', 'next_url': ''})

class PropertiesTable(Table):
    # classes = ['table-wrapper-scroll-y', 'table-striped', 'table-bordered', 'table-condensed']
    classes = ['table', 'table-striped', 'table-bordered', 'table-condensed']
    # table-wrapper-scroll-y
    name = Col('Name')
    units = Col('Units')


class ConfigurationsTable(Table):
    classes = ['table', 'table-striped', 'table-bordered', 'table-condensed']
    # classes = ['table-wrapper-scroll-y', 'table-striped', 'table-bordered', 'table-condensed']
    name = Col('Name')
    elements = Col('Elements')
    natoms = Col('N_atoms')
    labels = Col('Labels')


class DefinitionsTable(Table):
    classes = ['table', 'table-striped', 'table-bordered', 'table-condensed']
    # classes = ['table-wrapper-scroll-y', 'table-striped', 'table-bordered', 'table-condensed']
    name = Col('Name')
    elements = Col('Elements')
    natoms = Col('N_atoms')
    labels = Col('Labels')



class DatasetsTable(Table):
    classes = ['table', 'table-striped', 'table-bordered', 'table-condensed']
    # classes = ['table-wrapper-scroll-y', 'table-striped', 'table-bordered', 'table-condensed']
    name            = Col('Name')
    authors         = Col('Authors')
    links           = Col('Links')
    elements        = Col('Elements')
    properties      = Col('Properties')
    configurations  = Col('Configurations')
    atoms           = Col('Atoms')