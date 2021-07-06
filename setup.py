from setuptools import setup, find_namespace_packages

setup(
    name='tensortree',
    packages=["tensortree"],
    version='0.2',
    python_requires='>=3.7',
    description='Represent trees with tensors and perform fast calculations and modifications.',
    author='Johannes Villmow',
    author_email='johannes.villmow@hs-rm.de',
    url='https://github.com/villmow/tensortree',
    download_url='https://github.com/villmow/tensortree/archive/refs/tags/v0.2.tar.gz',
    license='MIT',
    install_requires=[
        "numpy",
        "torch",
    ],
    keywords=['tree', 'tensor', 'pytorch', 'pytorch tree', 'tensor tree'],  # Keywords that define your package best
    classifiers=[
        'Development Status :: 4 - Beta',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)

