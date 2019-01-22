from setuptools import setup

setup(
    name='pcs_aero',
    version='0.1.1',
    description='Drag and lift coefficient calculation in a D2Q9 LBM models.',
    author='Sebastian Melzer',
    author_email='pcs_aero@skydivizer.com',
    packages=['pcs_aero'],
    install_requires=[
        'numpy', 'numexpr', 'matplotlib', 'scikit_image', 'PyOpenGL'
    ])