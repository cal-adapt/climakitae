name: Tests

on:
  push:
    branches: 
    - add_branch_here

jobs:
  run_tests:
    runs-on: ubuntu-latest
    defaults:
      run:
        shell: bash -l {0}
    steps:
    - uses: actions/checkout@v2
    - uses: conda-incubator/setup-miniconda@v2
      with:
         auto-update-conda: false
         activate-environment: pangeo-notebook
         environment-file: conda-linux-64.lock
    - name: Checkout project
      uses: actions/checkout@v2
    - name: Output conda info 
      run: conda info
    - name: Install climakitae 
      run: pip install .
    - name: Test with pytest
      run: pytest
