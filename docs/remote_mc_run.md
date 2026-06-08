# Remote Monte Carlo run

Runbook for executing the heavy Monte Carlo on a second, more powerful Windows
machine and returning the results. Follow it top to bottom.

The Monte Carlo is parallel across realisations, and seeding is per-realisation,
so a parallel run matches a serial run bit for bit and is independent of the
worker count. The point of the second machine is simply more cores.

## One-time setup: a shared bare repo on the LAN

Both machines coordinate through a bare repository named `origin`, hosted on the
local network (not on P:, which holds only share-ready material). Do this once,
then skip to step 1 on later runs.

Pick a host on the LAN (for example the compute machine) and create the bare
repo in a folder you will share:

```powershell
git init --bare D:\git-remotes\weighting-wall.git
```

Share that folder so the other machine can reach it as a UNC path, then add it
as `origin` on both machines and seed it from the machine that has the work:

```powershell
# On each machine (use the host's real name and share path):
git remote add origin \\HOST\git-remotes\weighting-wall.git

# From the machine that currently has the commits:
git push -u origin master
```

If the other machine has no working copy yet, clone instead of adding a remote:
`git clone \\HOST\git-remotes\weighting-wall.git`.

SSH is an alternative to the file share if a host runs an OpenSSH server: create
the bare repo the same way and use `ssh://user@HOST/C:/git-remotes/weighting-wall.git`
as the remote URL.

## Conventions

- The P: network drive holds only clean, share-ready material. Do not write
  working data, trial runs, Monte Carlo outputs, or drafts to P:.
- If writing to P: at all, write only inside `P:\jthiebes`.

## 1. Update the working copy

```powershell
git pull
git rev-parse --short HEAD
git status
```

## 2. Build a pinned environment

Pin to the versions the analysis was developed on, so the curve-fit numerics --
and therefore the bias tail -- match. Use Python 3.13; install it if it is
absent.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install numpy==2.4.5 scipy==1.17.1 matplotlib==3.10.9 pytest==9.0.3
```

## 3. Verify the environment

All tests must pass before any long run.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## 4. Enable parallel Monte Carlo

Use all cores. Results are independent of the worker count.

```powershell
$env:WEIGHTING_WALL_WORKERS = 0
```

## 5. Run the task

Replace the commands below with the specific run required. Heavy defaults:

```powershell
Measure-Command { .venv\Scripts\python.exe scripts\per_profile_floor.py } | Select-Object TotalMinutes
Measure-Command { .venv\Scripts\python.exe scripts\dense_calibration_and_test.py } | Select-Object TotalMinutes
```

## 6. Return the results

`output/` is gitignored and P: is not used for trial runs, so return results on
a dedicated transfer branch.

```powershell
git checkout -b mc-run-<machine>-<yyyymmdd>
git add -f output\<files this run produced>
git commit -m "<one-line description of the run and its parameters>"
git push -u origin mc-run-<machine>-<yyyymmdd>
```

## Report

State: the commit pulled, the installed numpy / scipy / python versions, the
pytest result, the wall-clock for each run, the branch pushed, and the files
committed. Do not merge to `main`.
