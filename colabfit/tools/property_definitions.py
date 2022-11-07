potential_energy_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-05-30:property/potential-energy",
    "property-name": "potential-energy",
    "property-title": "Potential energy from a static calculation",
    "property-description": "Potential energy from a calculation of a static configuration. Energies must be specified to be per-atom or supercell. If a reference energy has been used, this must be specified as well.",
    "energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "The potential energy of the system."
    },
    "per-atom": {
        "type": "bool",
        "has-unit": False,
        "extent": [],
        "required": True,
        "description": "If True, \"energy\" is the total energy of the system, and has NOT been divided by the number of atoms in the configuration."
    },
    "reference-energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "If provided, then \"energy\" is the energy (either of the whole system, or per-atom) LESS the energy of a reference configuration (E = E_0 - E_reference). Note that \"reference-energy\" is just provided for documentation, and that \"energy\" should already have this value subtracted off. The reference energy must have the same units as \"energy\"."
    }
}

atomic_forces_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-05-30:property/atomic-forces",
    "property-name": "atomic-forces",
    "property-title": "Atomic forces from a static calculation",
    "property-description": "Atomic forces from a calculation of a static configuration.",
    "forces": {
        "type": "float",
        "has-unit": True,
        "extent": [
            ":",
            3
        ],
        "required": False,
        "description": "The [x,y,z] components of the force on each particle."
    }
}

cauchy_stress_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-05-30:property/cauchy-stress",
    "property-name": "cauchy-stress",
    "property-title": "Cauchy stress tensor from a static calculation",
    "property-description": "Full 3x3 Cauchy stress tensor from a calculation of a static configuration.",
    "stress": {
        "type": "float",
        "has-unit": True,
        "extent": [
            3,
            3
        ],
        "required": False,
        "description": "Cauchy stress tensor."
    }
}

free_energy_pd = {
    "property-id": "tag:staff@noreply.colabfit.org,2022-05-30:property/free-energy",
    "property-name": "free-energy",
    "property-title": "Free energy from a static calculation",
    "property-description": "Free energy from a calculation of a static configuration. Energies must be specified to be per-atom or supercell. If a reference energy has been used, this must be specified as well.",
    "energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "The free energy of the system."
    },
    "per-atom": {
        "type": "bool",
        "has-unit": False,
        "extent": [],
        "required": True,
        "description": "If True, \"energy\" is the total energy of the system, and has NOT been divided by the number of atoms in the configuration."
    },
    "reference-energy": {
        "type": "float",
        "has-unit": True,
        "extent": [],
        "required": False,
        "description": "If provided, then \"energy\" is the energy (either of the whole system, or per-atom) LESS the energy of a reference configuration (E = E_0 - E_reference). Note that \"reference-energy\" is just provided for documentation, and that \"energy\" should already have this value subtracted off. The reference energy must have the same units as \"energy\"."
    }
}