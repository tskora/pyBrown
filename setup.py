from setuptools import setup, find_packages, Extension

with open('README.md') as f:
    long_description = f.read()

extensions = [Extension("libpyBrown",
                       ["pyBrown/cbead.c", "pyBrown/diff_tensor.c"],
                       libraries=['blas','lapack', 'm'],
                       include_dirs=['pyBrown/'],
              )
]
setup(name='pyBrown',
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
      packages=['pyBrown'],
      zip_safe=False,
      ext_modules=extensions,
      entry_points={
        'console_scripts': [
            'pybrown-bd = pyBrown.BD:main',
            'pybrown-bd-restart = pyBrown.BD_restart:main',
            'pybrown-bd-nam = pyBrown.BD_NAM:main',
            'pybrown-bd-nam-restart = pyBrown.BD_NAM_restart:main',
        ],
      }
)