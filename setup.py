import setuptools

with open("Readme.md", "r") as f:
    long_description = f.read()

version_short = '0.1'
version_long = '0.1.0'

if __name__ == '__main__':
    setup(name='ccfitB',
          version=version_long,
          description='Cole-Cole fit routines',
          author='B. Mary',
          license='GPL-3',
          author_email='benjamin.mary@unipd.it',
          packages=setuptools.find_packages(),
          scripts=['src/cc_fit.py', ],
          install_requires=['numpy', 'scipy', 'matplotlib','pandas', 'seaborn', 'openpyxl'],
          )
