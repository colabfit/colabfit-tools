from wtforms import (
    SubmitField, StringField, MultipleFileField, validators,
    FieldList, FormField, SelectField
)

from flask_wtf import FlaskForm

class PropertyMapRow(FlaskForm):
    property_name   = StringField('Property name')#, [validators.InputRequired()])
    kim_field       = StringField('KIM field')#, [validators.InputRequired()])
    ase_field       = StringField('ASE field')#, [validators.InputRequired()])
    units           = StringField('Units')

class PropertySettingsRow(FlaskForm):
    property_name   = StringField('Property name')
    method          = StringField('Method')
    description     = StringField('Description')
    labels          = StringField('Labels')
    files           = MultipleFileField('Files')

class ConfigurationSetsRow(FlaskForm):
    query       = StringField('Mongo query')
    description = StringField('Description')

class PropertySettingsForm(FlaskForm):
    calculation_method      = StringField('Calculation method')
    calculation_description = StringField('Calculation description')
    calculation_labels      = StringField('Calculation labels')
    calculation_files       = MultipleFileField('Calculation files')

class UploadForm(FlaskForm):
    name = StringField(
        'Dataset name'
        # [validators.InputRequired()]
    )
    authors = StringField(
        'Authors',
        # [validators.InputRequired()]
    )

    links = StringField(
        'External links',
        [validators.Optional(strip_whitespace=True)]
    )

    description = StringField(
        'Description',
        # [validators.InputRequired(), validators.Length(min=6, max=1000)]
    )

    elements = StringField('Elements')

    data_upload = MultipleFileField(
        'Upload configurations',
        # [validators.regexp(u'^[^/\\]\.extxyz$')]
        # [validators.InputRequired()]
    )

    config_name_field = StringField(
        'Key for Configuration names'
    )

    definitions_upload = MultipleFileField(
        'Upload property definitions',
        # [validators.regexp(u'^[^/\\]\.extxyz$')]
        # [validators.InputRequired()]
    )

    submit = SubmitField("Submit")

class QueryForm(FlaskForm):
    collection = SelectField(
        'Collection',
        choices=[
            'datasets', 'configuration_sets', 'configurations', 'properties'
        ]
    )

    query = StringField('Query')
    search = SubmitField("Search")