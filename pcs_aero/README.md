## Synopsis

Drag and lift coefficient calculation in a D2Q9 LBM models.

## Installation
Simply install the package using pip.

```bash
pip install -e .
```

If for some reason you want to get rid of this amazing package just uninstall
using pip.

```bash
pip uninstall pcs_aero
```

## Usage
See the scripts in the scripts folder.

### Tips
Generate and simulate a model with `./dump.py file_name --your_flags`. This
generates a model for reusage. Make sure to run the simulation for some time
using the `-T` flag.

Most other scripts can run on the dumped model, e.g.
*   `./gui.py -l file_name`
*   `./streamplot.py -l file_name plotje.pdf`

The dump script can even be reused to advance the simulation to a further time
step, e.g.
`./dump.py file_name -l file_name -T desired_time`

## Experiments
Reproducing the pictures and graphs in the paper / poster will take some time.
They can be created with the example scripts by using the following commands.

```bash
# Perform experiments
scripts/runner.py "scripts/configs/full.py"

# Graphs are created in the notebook
jupyter notebook scripts/graphs.ipynb

# Two high res simulations for some nice pictures
scripts/dump.py -R500 -r400 -T20 -or rect.dat
scripts/dump.py -R500 -r400 -T20 -ot -t230 tri.dat
scripts/streamplot.py -l rect.dat sp_rect.pdf
scripts/streamplot.py -l tri.dat sp_tri.pdf
scripts/screenshot.py -l rect.dat ss_rect.png
scripts/screenshot.py -l tri.dat ss_tri.png
```

For a faster process use the commands below, but be aware this is at most an
approximation of the actual results. Also half of the shapes are skipped.
For more details compare the two configs.

```bash
# Perform experiments
scripts/runner.py "scripts/configs/minimal.py" 

# Graphs are created in the notebook
jupyter notebook scripts/graphs.ipynb

# Two not so high res simulations for some not so nice pictures
scripts/dump.py -R300 -r92 -T20 -or rect.dat
scripts/dump.py -R300 -r92 -T20 -ot -t230 tri.dat
scripts/streamplot.py -l rect.dat sp_rect.pdf
scripts/streamplot.py -l tri.dat sp_tri.pdf
scripts/screenshot.py -l rect.dat ss_rect.png
scripts/screenshot.py -l tri.dat ss_tri.png
```

Alernatively move the results folder to the scripts folder so that the notebook
reads the results delivered with this archive.
