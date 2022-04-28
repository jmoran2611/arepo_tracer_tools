import sys
import os
import glob
from tracer_reader import tracer_file
import numpy as np
import matplotlib.pyplot as plt
from gadget import gadget_readbinary
from read_params import read_params
from const import rsol, msol
from parallel_decorators import is_master, mpi_barrier, mpi_size, vectorize_parallel
from matplotlib import rcParams
import stat  # For the make-movie script
from make_movie import make_movie

"""

"""


@vectorize_parallel(method="MPI", use_progressbar=True)
def plot_tracer(
    NTimestep,
    pos,
    box=0.3,
    center=3.6e11,
    axes=[0, 1],
    trail_length=4,
    binary=False,
    centerofmass=False,
):
    if type(box) == int or type(box) == float:
        box = np.array([box, box, box])

    box = box * rsol
    if type(center) == int or type(center) == float:
        center = np.array([center, center, center])
    current_time = NTimestep * tracer.tracer_dt
    snap_index = round(current_time / params["TimeBetSnapshot"])
    binary_index = round(current_time / params["TimeBetStatistics"])
    box_center = np.ones(3) * params["BoxSize"] / 2.0
    if not centerofmass is False:
        center = centerofmass[NTimestep]
    if not binary is False:
        center = binary.poscomp[binary_index]
    center = centerofmass[0]
    # s = gadget_readsnap(snap_index, snappath=snap_path, snapbase=snap_base, hdf5=True, quiet=True, lazy_load=True)
    # center = s.centerofmass()
    plt.style.use("dark_background")
    if NTimestep < trail_length:
        i_in = 0
    else:
        i_in = NTimestep - trail_length
    if len(axes) > 2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        if not centerofmass is False:
            ax[0].scatter(
                centerofmass[NTimestep, 0],
                centerofmass[NTimestep, 1],
                marker="+",
                s=30,
                c="r",
                zorder=1000,
            )
            ax[1].scatter(
                centerofmass[NTimestep, 0],
                centerofmass[NTimestep, 2],
                marker="+",
                s=30,
                c="r",
                zorder=1000,
            )
        ax[0].scatter(
            box_center[0], box_center[1], marker="+", s=30, c="g", zorder=1000
        )
        ax[1].scatter(
            box_center[0], box_center[2], marker="+", s=30, c="g", zorder=1000
        )
        for tr in range(pos.shape[0]):
            ax[0].plot(
                pos[tr, i_in:NTimestep, 0], pos[tr, i_in:NTimestep, 1], c="w", lw=1
            )
            ax[1].plot(
                pos[tr, i_in:NTimestep, 0], pos[tr, i_in:NTimestep, 2], c="w", lw=1
            )
            ax[0].set_xlim(center[0] - box[0], center[0] + box[0])
            ax[0].set_ylim(center[1] - box[1], center[1] + box[1])
            ax[1].set_xlim(center[0] - box[0], center[0] + box[0])
            ax[1].set_ylim(center[2] - box[2], center[2] + box[2])
            ax[1].text(
                0.7,
                0.93,
                "Time: {:>7.02f}s".format(current_time),
                transform=ax[1].transAxes,
                fontname="Miriam Libre",
                color="white",
            )
    else:
        fig, ax = plt.subplots(1, 1)
        for tr in range(pos.shape[0]):
            ax.plot(
                pos[tr, i_in:NTimestep, axes[0]],
                pos[tr, i_in:NTimestep, axes[1]],
                c="w",
                lw=1,
            )
        ax.scatter(
            centerofmass[NTimestep, axes[0]],
            centerofmass[NTimestep, axes[1]],
            marker="+",
            s=30,
            c="r",
            zorder=1000,
        )
        ax.scatter(
            box_center[axes[0]],
            box_center[axes[1]],
            marker="+",
            s=30,
            c="g",
            zorder=1000,
        )
        ax.text(
            0.7,
            0.93,
            "Time: {:>7.02f}s".format(NTimestep * tracer.tracer_dt),
            transform=ax.transAxes,
            fontname="Miriam Libre",
            color="white",
        )
        ax.set_xlim(center[axes[0]] - box[axes[0]], center[axes[0]] + box[axes[0]])
        ax.set_ylim(center[axes[1]] - box[axes[1]], center[axes[1]] + box[axes[1]])
    fig.savefig(save_path + "/" + filename + "_%03d.png" % (NTimestep), dpi=300)
    plt.close(fig)


# Dictionary for values and the saved file name

axes = [0, 1, 2]
filename = "trail_plot"
if axes != [0, 1]:
    filename = filename + "ax1"
if len(axes) > 2:
    filename = "trail_plot_dual_cent"
# Plot range
boxsize = 0.1  # 0.9#0.3 # In rsol
center = [3.664e11, 3.6625e11, 3.65461144e11]
startfrom = 0
n_tracers = 1000
# n_tracer_file = n_tracers
n_tracer_file = 1000
plot_tidal_radius = True
WD_rad = 0.007604 * rsol  # Better model
# Scale bar
scale = boxsize / 5 * rsol
scale_center = 0.80

# Choose matplotlib font
rcParams["font.family"] = "Roboto"


# Get the simulation's folder as first argument
base_path = sys.argv[1]

# Snapshot location and names
if os.path.exists(base_path + "/energy.txt"):
    snap_path = base_path
else:
    snap_path = base_path + "/output"
snap_base = "snapshot_"
save_path = base_path + "/tracer_plots"

try:
    binary = gadget_readbinary(snap_path)
except:
    print("No binary.txt file found")
    binary = False

redo = False
snap_to_do = False
# Check if extra arguments were provided
if len(sys.argv) >= 3:
    for flag in sys.argv[2:]:
        if flag == "redo":
            redo = True
            continue
        try:
            if not os.path.isdir(flag):
                if is_master():
                    yn = input(
                        flag + " does not exist, do you want to create it? [Y/n]\n"
                    )
                    if yn == "Y":
                        os.mkdir(flag)
                    else:
                        sys.exit("Not creating", flag)
                mpi_barrier()
                save_path = flag
            else:
                save_path = flag
                if is_master():
                    print("Saving plots to", save_path)
        except:
            None
# Check if saving folder exists, if not, create it
if is_master():
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
mpi_barrier()

tracer = tracer_file(base_path, extended_output=True, Nspecies=5)
params = read_params(snap_path)
startfrom = int(startfrom / tracer.tracer_dt)
try:
    centerofmass = np.load(os.path.join(base_path, "tracer_centerofmass.npy"))
except:
    print("Center of mass of the simulation not found")
    centerofmass = False

if os.path.isfile(
    os.path.join(base_path, "followed_tracers" + str(n_tracer_file).zfill(2))
):
    tracer.read_followed_tracers(path=base_path, n=n_tracer_file)
else:
    if is_master():
        tracer.find_tracers_at_time(0)
        ids = tracer.find_tracers_in_box(
            [3e10, 3e10, 1e10], [3.65461144e11, 3.65461144e11, 3.65461144e11]
        )
        sample = np.random.choice(ids, size=n_tracers)
        print("Parsed data not found, following {:d} tracers...".format(n_tracers))
        tracer.follow_tracers(ids, vals=["pos", "vel", "temp"])
        tracer.save_followed_tracers(path=base_path, n=n_tracers)
    mpi_barrier()
    tracer.read_followed_tracers(path=base_path, n=n_tracer_file)
times = tracer.access_val("time")
pos = tracer.access_val("pos")
pos = pos[:n_tracers, :, :]
# Check number of timesteps
NTimesteps = times.shape[1] - startfrom
# Check number of pre-existing plots in the save folder
N_figures = len(glob.glob(save_path + "/" + filename + "_*"))

# Redo the existing plots or skip them
if redo:
    NTimesteps = np.arange(NTimesteps) + startfrom
else:
    NTimesteps = np.arange(NTimesteps - N_figures) + N_figures + startfrom

if type(snap_to_do) == int:
    NTimesteps = np.array([snap_to_do])

if is_master():
    print("Plotting tracers")
    print("Running with", mpi_size(), "processes")
    print("Reading from = " + snap_path + "\nSaving to = " + save_path)
    print("Making", len(NTimesteps), " plots")


mpi_barrier()
# vrange = None
if NTimesteps.size > 0:
    if mpi_size() > len(NTimesteps):
        if is_master():
            print(
                "\nWARNING:\tNumber of MPI tasks > Plots to make.\nSwitching to non-parallel plotting\n"
            )
            plot_ = plot_tracer.__wrapped__
            plot_(
                NTimesteps,
                pos,
                box=boxsize,
                center=center,
                axes=axes,
                binary=binary,
                centerofmass=centerofmass,
            )
    else:
        plot_tracer(
            NTimesteps,
            pos,
            box=boxsize,
            center=center,
            axes=axes,
            binary=binary,
            centerofmass=centerofmass,
        )
mpi_barrier()

# Write a script to make a movie out of the plots and attempt to run it
if is_master():
    if type(snap_to_do) != int:
        make_movie(save_path, filename, startfrom=startfrom, framerate=40)
