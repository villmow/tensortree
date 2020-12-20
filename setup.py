from setuptools import setup, find_namespace_packages

setup(name='torchtree',
      version='0.1a',
      python_requires='>=3.7',
      description='Machine learning with trees in PyTorch',
      author='Johannes Villmow',
      author_email='johannes.villmow@hs-rm.de',
      license='gnu',
      packages=find_namespace_packages(),
      install_requires=[
            "torch",
            "pytest"
      ],
      entry_points={
        'console_scripts': [
        ],
    },
)

