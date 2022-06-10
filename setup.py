from setuptools import find_packages, setup


setup(
    name='s_enformer',
    version='1.0.0',
    description='Just like Enformer, but with sparse-attention',
    python_requires='>=3.8',
    packages=find_packages(
        include=['s_enformer', 's_enformer.*']
    ),
    install_requires=[
        'dm-sonnet==2.0.0',
        'jupyter==1.0.0',
        'matplotlib==3.1.3',
        'numpy>=1.11.0',
        'pandas==1.2.3',
        'seaborn==0.11.1',
        'tensorflow-gpu==2.9.0',
        'tensorflow-hub==0.11.0',
        'tqdm==4.64.0',
    ],
    dependency_links=[],
    include_package_data=True,
)
