# bs-python-utils

[![Release](https://img.shields.io/github/v/release/bsalanie/bs-python-utils)](https://img.shields.io/github/v/release/bsalanie/bs-python-utils)
[![Build status](https://img.shields.io/github/actions/workflow/status/bsalanie/bs-python-utils/main.yml?branch=main)](https://github.com/bsalanie/bs-python-utils/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/bsalanie/bs-python-utils/branch/main/graph/badge.svg)](https://codecov.io/gh/bsalanie/bs-python-utils)
[![Commit activity](https://img.shields.io/github/commit-activity/m/bsalanie/bs-python-utils)](https://img.shields.io/github/commit-activity/m/bsalanie/bs-python-utils)
[![License](https://img.shields.io/github/license/bsalanie/bs-python-utils)](https://img.shields.io/github/license/bsalanie/bs-python-utils)

my Python utilities

- **Github repository**: <https://github.com/bsalanie/bs-python-utils/>
- **Documentation** <https://bsalanie.github.io/bs-python-utils/>

## Getting started with your project

First, create a repository on GitHub with the same name as this project, and then run the following commands:

``` bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:bsalanie/bs-python-utils.git
git push -u origin main
```

Finally, install the environment and the pre-commit hooks with 

```bash
make install
```

You are now ready to start development on your project! The CI/CD
pipeline will be triggered when you open a pull request, merge to main,
or when you create a new release.


## Releasing a new version

<!-- - Create an API Token on [Pypi](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting 
[this page](https://github.com/bsalanie/bs-python-utils/settings/secrets/actions/new). -->
- Create a [new release](https://github.com/bsalanie/bs-python-utils/releases/new) on Github. 
Create a new tag in the form ``*.*.*``.

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).