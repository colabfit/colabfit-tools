from ase import Atoms
from colabfit.tools.configuration import AtomicConfiguration
from colabfit.tools.utilities import convert_stress


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
                        forces.append(
                            [float(f) for f in [li["fx"], li["fy"], li["fz"]]]
                        )
            elif line.startswith("END_CFG"):
                if "cartes_x" in keys:
                    config = Atoms(positions=coords, symbols=symbols, cell=cell)
                elif "direct_x" in keys:
                    config = Atoms(scaled_positions=coords, symbols=symbols, cell=cell)
                config.info["energy"] = energy
                if forces:
                    config.info["forces"] = forces
                config.info["stress"] = stress  # Stress units appear to be kbar (?)
                config.info["_name"] = f"{filepath.stem}_{config_count}"
                config_count += 1
                yield AtomicConfiguration.from_ase(config)
                forces = None
                stress = []
                coords = []
                cell = []
                symbols = []
                energy = None
