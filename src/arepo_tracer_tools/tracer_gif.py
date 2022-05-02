#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import imageio
from arepo_tracer_tools import tracer_reader
import gadget_snap

def plot_gif(initial_time, final_time, boxsize, center, nparticles=1000):
    tf = tracer_reader.tracer_file(".", extended_output=False, Nspecies=55)

    # Find indices of n random particles
    tf.find_tracers_at_time(90)
    ids = tf.find_tracers_in_box([boxsize, boxsize, boxsize], [center, center, center])
    sample = np.random.choice(ids, size=nparticles)
    
    # Get boxsize of final timestep
    final_data = tf.find_tracers_at_time(final_time)
    boxsize_min = min(final_data["pos"][sample][:,0])
    boxsize_max = max(final_data["pos"][sample][:,0])

    kwargs_write = {'fps':10.0, 'quantizer':'nq'}
    imageio.mimsave('./tracers.gif', [plot_explosion(tf, sample, t, boxsize_min, boxsize_max) for t in np.arange(final_time, step=0.5)], fps=10)

    return


def plot_explosion(tf, sample, time, boxsize_min, boxsize_max):
    print("Plotting time %.1fs" % time)
    
    # Data for plotting
    data = tf.find_tracers_at_time(time)
    
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(data["pos"][sample][:,0], data["pos"][sample][:,1])
    ax.set(xlabel='x-position (cm)', ylabel='y-position (cm)')

    # IMPORTANT ANIMATION CODE HERE
    # Used to keep the limits constant
    ax.set_ylim(boxsize_min, boxsize_max)
    ax.set_xlim(boxsize_min, boxsize_max)

    # Used to return the plot as an image rray
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


if __name__ == "__main__":
    s = gadget_snap.gadget_snapshot("output/snapshot_812.hdf5", hdf5=True, lazy_load=True)
    center = s.centerofmass()[0]
    plot_gif(0, 90, 1.4e11, center)
