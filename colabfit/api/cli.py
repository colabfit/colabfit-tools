import re
import ast
import argparse
import numpy as np

from ase.io import read as ase_read

from colabfit.io import DataManager
from colabfit.converters import EXYZConverter, FolderConverter

parser = argparse.ArgumentParser()

parser.add_argument(
    "-c", "--command",
    help="What command to run. Must be 'publish' or 'search'",
    type=str,
    choices=['publish', 'search']
)

parser.add_argument(
    "-u", "--username",
    help="MongoDB username",
    type=str,
)

parser.add_argument(
    "-p", "--password",
    help="MongoDB password",
    type=str,
)

args = parser.parse_args()

def publish_prompt_from_file(filename, datamanager):

    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines()][::-1]
        print(lines)

    ds_name = lines.pop()

    authors = lines.pop()
    authors = [a.strip() for a in authors.split(',')]

    links = lines.pop()
    links = [l.strip() for l in links.split(' ')]

    dset_description = lines.pop()

    num_label_regexes = int(lines.pop())

    regex_labels = {}
    for _ in range(num_label_regexes):
        regex = lines.pop()
        labels = lines.pop()

        regex_labels[re.compile(regex)] = labels.split(' ')

    default_cs_desc = lines.pop()

    num_consets = int(lines.pop())

    regex_descriptions =  {re.compile(ds_name): default_cs_desc}

    for _ in range(num_consets):
        gname = lines.pop()
        gdesc = lines.pop()

        regex_descriptions[re.compile(gname)] = gdesc

    raw_format = lines.pop()

    if raw_format == 'xyz':

        name_field = lines.pop()

        calc_metadata = lines.pop()
        calc_metadata = [m.strip() for m in calc_metadata.split(';')]

        num_properties = int(lines.pop())

        property_mapping = {}
        property_units = {}
        for _ in range(num_properties):
            ptype = lines.pop()
            pname = lines.pop()
            units = lines.pop()

            property_mapping[pname] = ptype
            property_units[ptype] = units

            if ptype == 'stress':
                remap_virial = True if lines.pop().lower() == 'y' else False

        path = lines.pop()

        converter = EXYZConverter(
            base_path=path,
            property_mapping=property_mapping,
            calculation_metadata=calc_metadata,
            name_field=name_field,
            units=property_units,
            authors=authors,
            links=links,
            remap_virial=remap_virial
        )

    irv = True if lines.pop().lower() == 'y' else False
    icv = True if lines.pop().lower() == 'y' else False
    idv = True if lines.pop().lower() == 'y' else False
    chk = True if lines.pop().lower == 'y' else False
    
    tag = lines.pop()

    if len(tag) == 0:
        tag = None
    else:
        tag = int(tag)

    loaded_data = converter.load_data(ds_name=ds_name)

    reference_data, configurations = loaded_data

    datamanager.build_dataset_and_submit(
        ds_name=ds_name,
        reference_data=reference_data,
        configurations=configurations,
        regex_labels=regex_labels,
        regex_descriptions=regex_descriptions,
        increment_rd_version=irv,
        increment_cs_version=icv,
        increment_ds_version=idv,
        check_configurations=chk,
        dataset_tag=tag,
        dataset_description=dset_description,
        authors=authors,
        links=links,
        default_desc=default_cs_desc,
    )


def begin_publish_prompt():

    ds_name = input("Dataset name: ")
    print(ds_name)

    authors = input("Author names (separated by comma ','): ")
    authors = [a.strip() for a in authors.split(',')]
    print(authors)

    links = input("Reference links (separated by spaces ' '): ")
    links = [l.strip() for l in links.split(' ')]
    print(links)

    dset_description = input('Description: ')
    print(dset_description)

    num_label_regexes = int(input("Number of CO label regexes: "))
    print(num_label_regexes)

    regex_labels = {}
    for _ in range(num_label_regexes):
        regex = input("Regex: ")
        print(regex)
        labels = input("Labels (separated by spaces): ").split(' ')
        print(labels)

        regex_labels[re.compile(regex)] = labels

    default_cs_desc = input("Default configuration set description: ")
    print(default_cs_desc)

    num_consets = int(
        input("Number of configuration sets (not including default): ")
    )
    print(num_consets)

    regex_descriptions =  {re.compile(ds_name): default_cs_desc}

    for _ in range(num_consets):
        gname = input("Group name (regex): ")
        print(gname)
        gdesc = input("Group description: ")
        print(gdesc)

        regex_descriptions[re.compile(gname)] = gdesc

    raw_format = input("Data format ['xyz' or 'folder']: ")
    print(raw_format)

    if raw_format == 'xyz':

        name_field = input("atoms.info field containing configuration name (leave empty for random naming): ")
        print(name_field)

        calc_metadata = input("Calculation method metadata (separated by colon ';'): ")
        calc_metadata = [m.strip() for m in calc_metadata.split(';')]
        print(calc_metadata)

        num_properties = int(input("Number of properties: "))
        print(num_properties)

        property_mapping = {}
        property_units = {}
        for _ in range(num_properties):
            ptype = input("Property type ('energy', 'forces', ...): ")
            print(ptype)
            pname = input("Property name (<'info' or 'arrays'>.<property_name>): ")
            print(pname)
            units = input("Property units (ASE format): ")
            print(units)

            property_mapping[pname] = ptype
            property_units[ptype] = units

            if ptype == 'stress':
                remap_virial = True if input("Virial in 3x3 format? ['y' or 'n']: ").lower() == 'y' else False
                print(remap_virial)

        path = input("XYZ file path: ")
        print(path)

        converter = EXYZConverter(
            base_path=path,
            property_mapping=property_mapping,
            calculation_metadata=calc_metadata,
            name_field=name_field,
            units=property_units,
            authors=authors,
            links=links,
            remap_virial=remap_virial
        )

    elif raw_format == 'folder':
        num_properties = int(input("Number of properties (including 'configuration'): "))
        print(num_properties)

        file_names = {}
        file_load_fxns = {}
        property_units = {}
        for _ in range(num_properties):
            ptype = input("Property type ('energy', 'forces', ...): ")
            print(ptype)
            pfile = input("Property file name: ")
            print(pfile)
            if ptype != 'configuration':
                units = input("Property units (ASE format): ")
                print(units)
                property_units[ptype] = units

            largs = ast.literal_eval(input("Property load arguments (dict): "))
            print(largs)

            file_names[ptype] = pfile
            file_load_fxns[ptype] = (
                ase_read if ptype == 'configuration' else np.fromfile,
                largs,
            )

        path = input("Folder path: ")
        print(path)

        converter = FolderConverter(
            base_path=path,
            file_names=file_names,
            file_load_fxns=file_load_fxns,
            units=property_units,
            group_descriptions=regex_descriptions,
            authors=authors,
            links=links
        )

    irv = True if input("Increment RD versions? ['y' or 'n']: ").lower() == 'y' else False
    print(irv)
    icv = True if input("Increment CS versions? ['y' or 'n']: ").lower() == 'y' else False
    print(icv)
    idv = True if input("Increment DS versions? ['y' or 'n']: ").lower() == 'y' else False
    print(idv)
    chk = True if input("Check configurations? ['y' or 'n']: ").lower == 'y' else False
    print(chk)
    
    tag = input("Existing DS tag (Enter if None): ")
    print(tag)

    if len(tag) == 0:
        tag = None
    else:
        tag = int(tag)

    loaded_data = converter.load_data(ds_name=ds_name)

    reference_data, configurations = loaded_data

    datamanager = DataManager(user=args.username, password=args.password)

    datamanager.build_dataset_and_submit(
        ds_name=ds_name,
        reference_data=reference_data,
        configurations=configurations,
        regex_labels=regex_labels,
        regex_descriptions=regex_descriptions,
        increment_rd_version=irv,
        increment_cs_version=icv,
        increment_ds_version=idv,
        check_configurations=chk,
        dataset_tag=tag,
        dataset_description=dset_description,
        authors=authors,
        links=links,
        default_desc=default_cs_desc,
    )


def begin_search_prompt():
    pass


if __name__ == '__main__':

    if args.command is None:
        # import cProfile
        # from pstats import Stats
        # profiler = cProfile.Profile()
        # profiler.enable()

        inputs = [
            'publishing_inputs/Ta_Linear_JCP2015.txt',
            'publishing_inputs/Ta_PRM2019.txt',
            'publishing_inputs/WBe_PRB2019.txt',
            'publishing_inputs/W_PRB2019.txt',
            'publishing_inputs/Nb_PRM2019.txt',
            'publishing_inputs/V_PRM2019.txt',
            'publishing_inputs/Mo_PRM2019.txt',
            'publishing_inputs/InP_JPCA2020.txt',
            'publishing_inputs/MoNbTaVW_PRB2021.txt',
            'publishing_inputs/TiZrHfTa_APS2021.txt',
            'publishing_inputs/CuPd_CMS2019.txt',
            'publishing_inputs/CoNbV_CMS2019.txt',
            'publishing_inputs/AlNiTi_CMS2019.txt',
        ]

        datamanager = DataManager(args.username, args.password)

        for filename in inputs:
            print(filename)
            publish_prompt_from_file(filename, datamanager)

        # with open('profiling_stats.txt', 'w') as stream:
        #     stats = Stats(profiler, stream=stream)
        #     stats.strip_dirs()
        #     stats.sort_stats('time')
        #     stats.dump_stats('.prof_stats')
        #     stats.print_stats()
        # profiler.disable()
    else:
        if args.command == 'publish':
            begin_publish_prompt()
        elif args.command == 'search':
            begin_search_prompt()
