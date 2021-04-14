# pyBD

Brownian and Stokesian dynamics simulation software.
Copyright &copy;2021- Tomasz Skóra [tskora@ichf.edu.pl](mailto:tskora@ichf.edu.pl)

## Features

- [x] Brownian dynamics w/o hydrodynamic interactions
- [x] Brownian dynamics w hydrodynamic interactions
- [x] flux through an arbitrary plane (working on output)
- [x] immobile beads
- [ ] Stokesian dynamics (not optimized yet, but works)
- [ ] bead-bead interactions (inter- and intramolecular)
- [ ] various propagation schemes (EM, IGT, ...)

## Quick start

To compile, type following commands in a terminal:

```shell
./configure --prefix=DIR --with-lapack=LAPACK_LIBS
```
where DIR is the installation directory (/usr/local by default) 
and LAPACK_LIBS is lapack libraries to use (e.g. --with-lapack="-l lapack"

then proceed with 

```shell
make
make install
```

If you want tests, go to directory src/bdsims/tests and type 
```shell
make test
```

## Example input

```json
{
	"input_str_filename": "test_{}.str",
	"output_xyz_filename": "test_{}.xyz",
	"output_rst_filename": "test_{}.rst",
	"filename_range": [1, 2],
	"number_of_steps": 100000,
	"xyz_write_freq": 1000,
	"rst_write_freq": 100,
	"dt": 1.0,
	"T": 293.15,
	"viscosity": 0.01005,
	"box_length": 750.0,
	"hydrodynamics": "nohi",
	"seed": 0,
	"progress_bar": false,
	"measure_flux": {
		"normal": [1.0, 0.0, 0.0],
		"plane_point": [0.0, 0.0, 0.0]
	}
}
```

## Units

| Physical property | Units |
|---|---|
| Temperature | kelvin (*K*) |
| Viscosity | poise (*P*) |
| Time | picosecond (*ps*) |
| Distance | angstrom (*Å*) |
| Force | joule per angstrom (*J/Å*) |
