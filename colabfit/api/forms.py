from wtforms import (
    SubmitField, StringField, FileField, validators
)

from flask_wtf import FlaskForm

class UploadForm(FlaskForm):
    authors = StringField(
        'Authors',
        [validators.InputRequired()]
    )

    links = StringField(
        'External links',
        [validators.Optional(strip_whitespace=True)]
    )

    description = StringField(
        'Dataset description',
        [validators.InputRequired(), validators.Length(min=6, max=1000)]
    )

    upload = FileField(
        'File upload',
        # [validators.regexp(u'^[^/\\]\.extxyz$')]
    )

    submit = SubmitField('Submit')