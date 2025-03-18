import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required.")
    sys.exit(1)

version = '0.0.1.dev'

setup(name="HYDRA",
      description="HYbrid Discontinous galerkin solveR for flow Analysis.",
      version=version,
      author="Bouteiller Paul",
      author_email="paul.bouteiller@cea.fr",
      packages = find_packages(),
)

