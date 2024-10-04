from setuptools import setup, find_packages

setup(
    name='Concord',
    version='0.8.1',
    description='CONCORD: Contrastive Learning for Cross-domain Reconciliation and Discovery',
    packages=find_packages(include=['Concord','Concord.*']),
    author='Qin Zhu',
    author_email='qin.zhu@ucsf.edu',
    url='https://github.com/qinzhu/Concord',
    install_requires=[
        # dependencies here
    ],
    python_requires='>=3.7',
)
