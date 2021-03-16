# pyBD is a Brownian and Stokesian dynamics simulation tool
# Copyright (C) 2021  Tomasz Skora (tskora@ichf.edu.pl)
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

import numpy as np
import json

from pyBD.bead import Bead

#-------------------------------------------------------------------------------

class InputData:

    def __init__(self, input_filename, obligatory_keywords = [], defaults = {}):

        self._read_input_file(input_filename)

        self._complete_with_defaults(defaults)

        self._check_for_missing_keywords(obligatory_keywords)

    #---------------------------------------------------------------------------

    def __str__(self):

        string_representation = ''

        for keyword in self.input_data.keys():
            string_representation += '{}: {}\n'.format(keyword, self.input_data[keyword])

        return string_representation

    #---------------------------------------------------------------------------

    def __repr__(self):

        return self.__str__()

    #---------------------------------------------------------------------------

    def _read_input_file(self, input_filename):

        with open(input_filename, "r") as read_file:
            self.input_data = json.load(read_file)

    #---------------------------------------------------------------------------

    def _complete_with_defaults(self, defaults):

        for default_keyword in defaults.keys():

            if default_keyword not in self.input_data.keys():

                self.input_data[default_keyword] = defaults[default_keyword]

    #---------------------------------------------------------------------------

    def _check_for_missing_keywords(self, obligatory_keywords):

        for keyword in obligatory_keywords:
            assert keyword in self.input_data.keys(),\
                'Missing {} keyword in input JSON file.'.format(keyword)

#-------------------------------------------------------------------------------

def read_str_file(input_str_filename):

    with open(input_str_filename) as str_file:
        
        beads = []

        for line in str_file:

            if line.split()[0] == 'sub':
                label = line.split()[1]
                coords = np.array([ float(line.split()[i]) for i in range(3, 6) ])
                hydrodynamic_radius = float(line.split()[6])
                charge = float(line.split()[7])
                lennard_jones_radius = float(line.split()[8]) / 2
                lennard_jones_energy = float(line.split()[9])
                mass = float(line.split()[10])

                beads.append( Bead(coords, hydrodynamic_radius, label) )

    return beads

#-------------------------------------------------------------------------------