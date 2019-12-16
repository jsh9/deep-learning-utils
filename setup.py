from setuptools import setup

setup(
    name='deep_learning_utils',
    version='v0.0.1',
    description='deep-learning-utils',
    author='Jian Shi',
    license='GPL v3',
    url='https://github.com/jsh9/deep-learning-utils',
    packages=['deep_learning_utils'],
    classifiers=['Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
    ],
    install_requires=['numpy>=1.11.0',
                      'scipy>=1.1.0',
                      'torch>=1.3.0',
                      'torchvision>=0.4.1',
                      'typeguard>=2.7.0'
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
