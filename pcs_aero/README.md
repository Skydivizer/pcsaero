## Synopsis

Drag and lift coefficient calculation in a D2Q9 LBM models.

## Installation
Simply install the package using pip.

```bash
pip install -e .
```

## Usage
See the example scripts in the scripts folder.

### Tips
Generate and simulate a model with `./dump.py file_name --your_flags`. This
generates a  model for reusage. Make sure to run the simulation for some time
using the `-T` flag.

Most other scripts can run on the dumped model, e.g.
*   `./gui.py -l FILE_NAME`
*   `./streamplot.py -l FILE_NAME`

The dump script can even be reused to advance the simulation to a further time
step, e.g.
`./dump.py file_name -l file_name -T DESIRED_TIME`

## Experiments
Reproducing the pictures and graphs in the paper / poster will take some time,
they can be created with the example scripts by using the following commands.

```bash

```