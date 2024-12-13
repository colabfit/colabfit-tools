energy_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2024-06-01:property/energy",
    "property-name": "energy",
    "property-title": "Energy from which atomic forces calculations have been derived",
    "property-description": (
        "The calculated energy property used to derive the value of the atomic forces "
        "property. Values are converted if necessary to units of eV, and normalized to "
        "per-configuration energy (i.e., energy reported in this column is not energy "
        "per atom)."
    ),
    "energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": True,
        "description": "The potential energy of the system.",
    },
    "per-atom": {
        "type": "bool",
        "has-unit": False,
        "extent": [],
        "required": False,
        "description": 'If True, "energy" is the total energy of the system divided by '
        "the number of atoms in the configuration.",
    },
    "reference-energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": 'If provided, then "energy" is the total energy (either '
        "of the whole system, or per-atom) "
        "minus the energy of a reference configuration (E = E_0 - E_reference). "
        'Note that "reference-energy" is just provided for documentation, and that '
        '"energy" should already have this value subtracted off. The reference '
        'energy must have the same units as "energy".',
    },
}

atomic_forces_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-05-30:property/atomic-forces",
    "property-name": "atomic-forces",
    "property-title": "Atomic forces from a static calculation",
    "property-description": (
        "The x, y, and z components of the force on each atom, calculated as a "
        "molecule’s energy differentiated with respect to atom positions and "
        "representing the negative gradient of the energy with respect to "
        "atom positions."
    ),
    "forces": {
        "type": "float",
        "has-unit": True,
        "extent": [":", 3],
        "required": True,
        "description": "The [x,y,z] components of the force on each particle.",
    },
}

cauchy_stress_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-05-30:property/cauchy-stress",
    "property-name": "cauchy-stress",
    "property-title": "Cauchy stress tensor from a static calculation",
    "property-description": "Full 3x3 Cauchy stress tensor from a calculation of a static configuration.",
    "stress": {
        "type": "float",
        "has-unit": True,
        "extent": [3, 3],
        "required": True,
        "description": "Cauchy stress tensor.",
    },
    "volume-normalized": {
        "type": "bool",
        "has-unit": False,
        "extent": [],
        "required": False,
        "description": "If True, the stress has been multiplied by the cell volume.",
    },
}

atomization_energy_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-11-18:property/atomization-energy",
    "property-name": "atomization-energy",
    "property-title": "Atomization energy from a static calculation",
    "property-description": "Formation energy from a calculation of a static "
    "configuration which uses atomic systems to compute reference energies. Energies "
    "must be specified to be per-atom if true. If the reference energies are known, "
    "these must be specified as well.",
    "energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "The atomization energy of the system.",
    },
    "per-atom": {
        "type": "bool",
        "has-unit": False,
        "extent": [],
        "required": False,
        "description": 'If True, "energy" is the atomization energy of the system '
        "divided by the number of atoms in the configuration.",
    },
    "reference-energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "If known, represents values which have been subtracted from "
        "the base energy, resulting in the atomization energy.",
    },
}

adsorption_energy_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-11-18:property/adsorption-energy",
    "property-name": "adsorption-energy",
    "property-title": "Adsorption energy from a static calculation",
    "property-description": "Adsorption energy from a calculation of a static "
    "configuration which uses non-atomic systems to compute reference energies. "
    "Energies must be specified to be per-atom if true. If the reference energies are "
    "known, this must be specified as well.",
    "energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "The adsorption energy of the system.",
    },
    "per-atom": {
        "type": "bool",
        "has-unit": False,
        "extent": [],
        "required": False,
        "description": 'If True, "energy" is the adsorption energy of the system '
        "divided by the number of atoms in the configuration.",
    },
    "reference-energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "If known, represents values which have been subtracted from "
        "the base energy, resulting in the adsorption energy.",
    },
}


formation_energy_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-11-18:property/formation-energy",
    "property-name": "formation-energy",
    "property-title": "Formation energy from a static calculation",
    "property-description": "Formation energy from a calculation of a static "
    "configuration which uses non-atomic systems to compute reference energies. "
    "Energies must be specified to be per-atom if true. If the reference energies are "
    "known, this must be specified as well.",
    "energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "The formation energy of the system.",
    },
    "per-atom": {
        "type": "bool",
        "has-unit": False,
        "extent": [],
        "required": False,
        "description": 'If True, "energy" is the formation energy of the system '
        "divided by the number of atoms in the configuration.",
    },
    "reference-energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "If known, represents values which have been subtracted from "
        "the base energy, resulting in the formation energy.",
    },
}


band_gap_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-11-18:property/band-gap",
    "property-name": "band-gap",
    "property-title": "Band gap energy from a static calculation",
    "property-description": "Band gap energy from a calculation of a static "
    "configuration.",
    "energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "The band gap energy of the system.",
    },
    "type": {
        "type": "string",
        "has-unit": False,
        "extent": [],
        "required": True,
        "description": "The type of band gap calculation: [direct, indirect, unknown].",
    },
}
