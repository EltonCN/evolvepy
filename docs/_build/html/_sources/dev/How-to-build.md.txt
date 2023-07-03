# How to build and release

To build and release EvolvePy, you need to follow these steps:

- Check if the new version release is ready with the other developers.
- Update setup.cfg (develop branch)
  - New version
  - New dependencies created
- Update documentation (develop branch)
  - Install all extra dependencies (`pip install .[extras]`)
  - In docs folder, `sphinx-apidoc --force ../src -o .`
  - In docs folder, `make html`
- Merge develop into main (all tests must pass)
- Build: `python -m build`
- Upload to PyPI:
  - Ask for be added to EvolvePy's project, if it hasn't already been
  - Upload: `python -m twine upload dist/*`, it will ask for your credentials