from ase import Atoms
from ase.io import iread
from colabfit.tools.configuration import AtomicConfiguration
from colabfit.tools.utilities import convert_stress
from pathlib import Path
import re

##############################################################
# Helper functions
##############################################################


def name_config_by_filepath(fp: Path, dataset_path: Path) -> str:
    """
    Generates a configuration name from the filepath with dataset path removed.
    Args:
        fp (Path): File path from which the configuration name is to be generated.
        dataset_path (Path): dataset_path will be removed from the beginning of the fp.
    Returns:
        str: The generated configuration name.
    """
    relative_path = fp.relative_to(dataset_path)
    name = "__".join(relative_path.parts)
    return name


def read_directory(directory: Path, parser, rglobstr="*.extxyz", **kwargs):
    """
    Read all files in a directory with a given parser.
    Args:
        directory (Path): Parent directory of data files.
        parser (function): Parser function to read data files.
        rglobstr (str): rglob string to search for data files.
        kwargs: Additional keyword arguments to pass to the parser.
    Returns:
        A generator of parsed configurations.
    """
    if isinstance(directory, str):
        try:
            directory = Path(directory)
            if not directory.exists():
                raise ValueError(f"{directory} does not exist")
        except Exception as e:
            raise ValueError(f"Could not convert {directory} to Path object") from e
    files = directory.rglob(rglobstr)
    for file in files:
        if kwargs:
            yield from parser(file, **kwargs)
        else:
            yield from parser(file)


##############################################################
# extxyz file parser
##############################################################


def read_extxyz(filepath: Path, dataset_path: Path):
    with open(filepath, "rt") as f:
        for i, config in enumerate(iread(f, format="extxyz", index=":")):
            config.info["_name"] = (
                name_config_by_filepath(filepath, dataset_path) + f"__index__{i}"
            )
            yield AtomicConfiguration.from_ase(config)


def read_extxyz_no_ix(filepath: Path, dataset_path: Path):
    "Returns configurations with no index in the configuration name"
    with open(filepath, "rt") as f:
        for i, config in enumerate(iread(f, format="extxyz", index=":")):
            config.info["_name"] = name_config_by_filepath(filepath, dataset_path)
            yield AtomicConfiguration.from_ase(config)


##############################################################
# MLIP .cfg file parser
##############################################################


def mlip_cfg_reader(symbol_map, filepath):
    with open(filepath, "rt") as f:
        energy = None
        forces = None
        coords = []
        cell = []
        symbols = []
        config_count = 0
        for line in f:
            if line.strip().startswith("Size"):
                size = int(f.readline().strip())
            elif line.strip().lower().startswith("supercell"):
                cell.append([float(x) for x in f.readline().strip().split()])
                cell.append([float(x) for x in f.readline().strip().split()])
                cell.append([float(x) for x in f.readline().strip().split()])
            elif line.strip().startswith("Energy"):
                energy = float(f.readline().strip())
            elif line.strip().startswith("PlusStress"):
                stress_keys = line.strip().split()[-6:]
                stress = [float(x) for x in f.readline().strip().split()]
                stress = convert_stress(stress_keys, stress)
            elif line.strip().startswith("AtomData:"):
                keys = line.strip().split()[1:]
                if "fx" in keys:
                    forces = []
                for i in range(size):
                    li = {
                        key: val for key, val in zip(keys, f.readline().strip().split())
                    }
                    symbols.append(symbol_map[li["type"]])
                    if "cartes_x" in keys:
                        coords.append(
                            [
                                float(c)
                                for c in [
                                    li["cartes_x"],
                                    li["cartes_y"],
                                    li["cartes_z"],
                                ]
                            ]
                        )
                    elif "direct_x" in keys:
                        coords.append(
                            [
                                float(c)
                                for c in [
                                    li["direct_x"],
                                    li["direct_y"],
                                    li["direct_z"],
                                ]
                            ]
                        )
                    if "fx" in keys:
                        forces.append([float(f) for f in [li["fx"], li["fy"], li["fz"]]])
            elif line.startswith("END_CFG"):
                if "cartes_x" in keys:
                    config = Atoms(positions=coords, symbols=symbols, cell=cell)
                elif "direct_x" in keys:
                    config = Atoms(scaled_positions=coords, symbols=symbols, cell=cell)
                config.info["energy"] = energy
                if forces:
                    config.info["forces"] = forces
                config.info["stress"] = stress  # Stress units appear to be kbar (?)
                config.info["_name"] = f"{filepath.stem}__index__{config_count}"
                config_count += 1
                yield AtomicConfiguration.from_ase(config)
                forces = None
                stress = []
                coords = []
                cell = []
                symbols = []
                energy = None


##############################################################
# VASP OUTCAR parser functions
##############################################################


vasp_coord_regex = re.compile(
    r"^\s+(?P<x>\-?\d+\.\d+)\s+(?P<y>\-?\d+\.\d+)\s+(?P<z>\-?\d+\.\d+)\s+(?P<fx>\-?"
    r"\d+\.\d+)\s+(?P<fy>\-?\d+\.\d+)\s+(?P<fz>\-?\d+\.\d+)"
)
param_re = re.compile(
    r"[\s+]?(?P<param>[A-Z_]+)(\s+)?=(\s+)?(?P<val>-?([\d\w\.\-]+)?\.?)"
    r"[\s;]?(?P<unit>eV)?\:?"
)
IGNORE_PARAMS = [
    "VRHFIN",
    "LEXCH",
    "EATOM",
    "TITEL",
    "LULTRA",
    "IUNSCR",
    "RPACOR",
    "POMASS",
    "RCORE",
    "RWIGS",
    "ENMAX",
    "RCLOC",
    "LCOR",
    "LPAW",
    "EAUG",
    "DEXC",
    "RMAX",
    "RAUG",
    "RDEP",
    "RDEPT",
]


def vasp_contcar_parser(fp):
    with open(fp, "r") as f:
        for i in range(5):
            _ = f.readline()
        line = f.readline()
        symbols = line.strip().split()
        counts = [int(x) for x in f.readline().strip().split()]
        symbol_arr = []
        for symbol, count in zip(symbols, counts):
            symbol_arr.extend([symbol] * count)
        return symbol_arr


def vasp_outcar_reader(symbols, fp):
    with open(fp, "r") as f:
        incar = dict()
        cinput = dict()
        in_latt = False
        in_coords = False
        lattice = []
        pos = []
        forces = []
        stress = None
        potcars = set()
        energy = None
        for line in f:
            # Prelim handling
            if line.strip() == "":
                continue
            # handle lattice
            elif "direct lattice vectors" in line:
                in_latt = True
                lattice = []
            elif in_latt is True:
                latt = line.strip().replace("-", " -").split()
                lattice.append([float(x) for x in [latt[0], latt[1], latt[2]]])
                if len(lattice) == 3:
                    in_latt = False
            elif "POTCAR:" in line:
                potcars.add(" ".join(line.strip().split()[1:]))
            elif "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in line:
                _ = f.readline()
                _, _, _, _, energy, units = f.readline().strip().split()
                if len(pos) > 0:
                    cinput["incar"] = incar
                    config = Atoms(positions=pos, symbols=symbols, cell=lattice)
                    config.info["input"] = cinput
                    config.info["input"]["potcars"] = list(potcars)
                    # config.info["outcar"] = outcar
                    if stress is not None:
                        config.info["stress"] = stress
                    config.info["forces"] = forces
                    config.info["energy"] = float(energy)
                    yield config
                    forces = []
                    stress = None
                    pos = []
                    energy = None
            elif "POSITION" in line:
                in_coords = True
            elif in_coords is True:
                if "--------" in line:
                    continue
                elif "total drift" in line:
                    in_coords = False
                    if energy is not None:
                        cinput["incar"] = incar
                        config = Atoms(positions=pos, symbols=symbols, cell=lattice)
                        config.info["input"] = cinput
                        config.info["input"]["potcars"] = list(potcars)
                        # config.info["outcar"] = outcar
                        config.info["forces"] = forces
                        if stress is not None:
                            config.info["stress"] = stress
                        config.info["energy"] = float(energy)
                        yield config
                        forces = []
                        stress = None
                        pos = []
                        energy = None
                    else:
                        continue
                else:
                    cmatch = vasp_coord_regex.search(line)
                    pos.append(
                        [float(p) for p in [cmatch["x"], cmatch["y"], cmatch["z"]]]
                    )
                    forces.append(
                        [float(p) for p in [cmatch["fx"], cmatch["fy"], cmatch["fz"]]]
                    )
            elif "Direction" in line and "XX" in line:
                stress_keys = [x.lower() for x in line.strip().split()[1:]]
            elif "in kB" in line:
                stress = [float(x) for x in line.strip().split()[2:]]
                stress = convert_stress(stress_keys, stress)

            else:
                continue
                # print("something went wrong")


def vasp_get_kpoints(fp):
    with open(fp, "r") as f:
        # f.readline() # if skipping first line
        kpoints = "".join(f.readlines())
    return kpoints


def vasp_parse_incar(fp):
    with open(fp, "r") as f:
        return f.read()


def file_finder(fp, file_glob, count=0):
    if count > 5:
        return None
    elif next(fp.glob(file_glob), None) is not None:
        # file_glob in [f.name for f in fp.glob("*")]:
        return next(fp.glob(file_glob))
    else:
        count += 1
        return file_finder(fp.parent, file_glob, count)


def vasp_outcar_wrapper(data_dir: Path, dataset_path, CO_METADATA=None):
    outcars = sorted(list(data_dir.rglob("OUTCAR")))
    for filepath in outcars:
        name = name_config_by_filepath(filepath, dataset_path)
        poscar = next(filepath.parent.glob(filepath.name.replace("OUTCAR", "POSCAR")))
        symbols = vasp_contcar_parser(poscar)
        kpoints_file = file_finder(filepath.parent, "KPOINTS")
        kpoints = vasp_get_kpoints(kpoints_file)
        incar_file = file_finder(filepath.parent, "INCAR")
        incar = vasp_parse_incar(incar_file)

        config_gen = vasp_outcar_reader(fp=filepath, symbols=symbols)
        for i, config in enumerate(config_gen):
            config.info["name"] = f"{name}_{i}"
            config.info["input"]["kpoints"] = kpoints
            config.info["input"]["incar"] = incar
            yield AtomicConfiguration.from_ase(config, CO_METADATA)
