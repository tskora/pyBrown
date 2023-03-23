from setuptools import setup, find_packages, Extension

with open('README.md') as f:
    long_description = f.read()

extensions = [Extension("libpybrown",
                       ["pybrown/cbead.c", "pybrown/diff_tensor.c"],
                       libraries=['blas','lapack', 'm'],
                       include_dirs=['pybrown/'],
              )
]
setup(name='pybrown',
      version='0.1.0',
      description='Brownian dynamics simulation package',
      author='Tomasz Skora',
      author_email='tskora93@gmail.com',
      long_description=long_description,
      long_description_content_type='text/markdown',
      project_urls = {
          'Documentation': 'https://tskora.github.io/pyBrown',
          'Source': 'https://github.com/tskora/pyBrown'
      },
      license='GNU GPL',
      install_requires = ['click','scipy','numpy','tqdm'],
      packages=['pybrown'],
      zip_safe=False,
      ext_modules=extensions,
      entry_points={
        'console_scripts': [
            'pybrown_bd = pybrown.BD:main',
            'pybrown_bd_restart = pybrown.BD_restart:main',
            'pybrown_bd_nam = pybrown.BD_NAM:main',
            'pybrown_bd_nam_restart = pybrown.BD_NAM_restart:main',
        ],
      }
)