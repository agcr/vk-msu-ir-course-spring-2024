from setuptools import find_packages, setup

with open('requirements.txt', 'r') as fin:
    reqs = fin.read().splitlines()

setup(
    name="ranking-lib",
    packages=find_packages(),
    include_package_data=True,
    version="0.1.0",
    author="Me",
    license="MIT",
    install_requires=reqs,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)