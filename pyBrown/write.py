# pyBrown is a bundle of tools useful for Brownian and Stokesian dynamics
# simulations. Copyright (C) 2018  Tomasz Skora (tskora@ichf.edu.pl)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see https://www.gnu.org/licenses.

# TODO: correct writing in mc_fluct (sum of radii, not double the radius)
def write_structure(input_data, coordinates, output_str_filename):

    numbers_of_molecules = input_data["numbers_of_molecules"]

    hydrodynamic_radii = input_data["hydrodynamic_radii"]
    charges = input_data["charges"]
    lennard_jones_radii = input_data["lennard-jones_radii"]
    lennard_jones_energies = input_data["lennard-jones_energies"]
    masses = input_data["masses"]
    labels_of_molecules = input_data["labels_of_molecules"]

    bond_force_constants = input_data["bond_force_constants"]
    max_bond_lengths = input_data["max_bond_lengths"]
    bond_lengths = input_data["bond_lengths"]
    angle_force_constants = input_data["angle_force_constants"]

    if 'open_radii' in input_data.keys(): open_radii = input_data["open_radii"]
    if 'close_radii' in input_data.keys(): close_radii = input_data["close_radii"]
    packing_mode = input_data["packing_mode"]

    bonds = ""
    angles = ""

    with open(output_str_filename, "w") as write_file:

        count = 0

        for i, n_mol in enumerate(numbers_of_molecules):

            if len( hydrodynamic_radii[i] ) >= 3:
                if_bonds = True
                if_angles = True 
            elif len( hydrodynamic_radii[i] ) >= 2:
                if_bonds = True
                if_angles = False
            else:
                if_bonds = False
                if_angles = False

            for id_mol in range(n_mol):

                for j, r_bead in enumerate(hydrodynamic_radii[i]):

                    line = _bead_pattern(labels_of_molecules[i],
                                         count,
                                         coordinates[count],
                                         r_bead,
                                         charges[i][j],
                                         lennard_jones_radii[i][j],
                                         lennard_jones_energies[i][j],
                                         masses[i][j])
                    write_file.write(line + '\n')

                    if if_bonds and j < len(hydrodynamic_radii[i]) - 1:

                        if packing_mode == 'monte_carlo_fluct':
                            bonds += _bond_pattern(count, count + 1, 2*close_radii[i][j], 2*open_radii[i][j], bond_force_constants[i][j]) + '\n'
                        else:
                            bonds += _bond_pattern(count, count + 1, bond_lengths[i][j], max_bond_lengths[i][j], bond_force_constants[i][j]) + '\n'

                    if if_angles and j < len(hydrodynamic_radii[i]) - 2:
                        angles += _angle_pattern(count, count + 1, count + 2, 180.0, angle_force_constants[i][j]) + '\n'

                    count += 1

        write_file.write(bonds)
        write_file.write(angles)

#-------------------------------------------------------------------------------

def _bead_pattern(label, index, coords, rh, charge, rlj, elj, m):

    return "sub {} {} {:.2f} {:.2f} {:.2f} {} {} {} {} {}".format(
        label,
        index + 1,
        coords[0],
        coords[1],
        coords[2],
        rh,
        charge,
        2 * rlj,
        elj,
        m
    )

#-------------------------------------------------------------------------------

def _bond_pattern(index1, index2, r_eq, r_max, force_constant):

    return "bond {} {} {} {} {}".format(
        index1 + 1,
        index2 + 1,
        r_eq,
        r_max,
        force_constant
    )

#-------------------------------------------------------------------------------

def _angle_pattern(index1, index2, index3, phi_eq, force_constant):

    return "angle angle {} {} {} {} {}".format(
        index1 + 1,
        index2 + 1,
        index3 + 1,
        phi_eq,
        force_constant
    )
