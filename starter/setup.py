import setuptools

setuptools.setup(
    name="starter",
    version="0.0.0",
    description="Starter code.",
    author="Student",
    # Lower bounds match pinned versions in requirements.txt;
    # upper bounds exclude the next major to avoid breaking changes.
    install_requires=[
        "pandas>=2.3.2,<3.0",
        "numpy>=2.3.3,<3.0",
        "scikit-learn>=1.7.2,<2.0",
        "PyYAML>=6.0,<7.0",
    ],
)
