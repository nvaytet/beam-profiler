import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def find_current_position_from_time(current_time, target_pos, target_time):
    """
    Find the current (x or y) position corresponding to the timestamp from
    the other axis (y or x).
    """
    idx = np.searchsorted(target_time, current_time, side = 'left') - 1
    if idx < 0:
        idx = 0
    return target_pos[idx]


def get_intersect(A, B, C, D):
    """
    Find the point of intersection betweem 2 straight lines defined by two
    points lying on the lines. First line is (A-B), second is (C-D).
    """
    # a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * (A[0]) + b1 * (A[1])
    # a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * (C[0]) + b2 * (C[1])
    # Determinant
    det = a1 * b2 - a2 * b1
    # parallel line
    if det == 0:
        return None
    # Intersect point(x,y)
    E = np.array([((b2 * c1) - (b1 * c2)) / det,
                  ((a1 * c2) - (a2 * c1)) / det])
    # Compute scalar product to check that point lies between A & B
    scal = np.dot(E-A, E-B)
    if scal > 0:
        return None
    return E


def profile(infile=None, outfile="beam_profile.pdf", sampling=64,
            resolution=256, vmin=None, vmax=None, nc=17, debug=False):
    """
    :param infile: Input file containing event and linear axis data.
    :param outfile: Output file to save figure.
    :param sampling: Sampling spatial resolution for the slit positions.
    :param resolution: Resolution of final interpolated image.
    :param vmin: Minimum value for image colorscale.
    :param vmax: Maximum value for image colorscale.
    :param nc: Number of contour levels in image.
    :param debug: Print additional information if true.
    :return: A 2d numpy array containing the data from the image.
    """

    # Read events and slit time stamps from file
    with h5py.File(infile, "r") as f:
        event_time = f["/entry/monitor_1/events/event_time_zero"][...]
        ypos = f["/entry/instrument/linear_axis_2/value/value"][...]
        ypos_time = f["/entry/instrument/linear_axis_2/value/time"][...]
        zpos = f["/entry/instrument/linear_axis_1/value/value"][...]
        zpos_time = f["/entry/instrument/linear_axis_1/value/time"][...]

    # Go through the positions of each slit axis and find the corresponding
    # position of the other axis by searching for identical timestamps
    actual_pos_list = []
    actual_pos_times = []
    for i in range(len(ypos)):
        current_ypos = ypos[i]
        if len(zpos) > 0:
            current_zpos = find_current_position_from_time(ypos_time[i], zpos,
                                                           zpos_time)
        else:
            current_zpos = 0.0
        actual_pos_list.append([current_ypos, current_zpos])
        actual_pos_times.append(ypos_time[i])
    for j in range(len(zpos)):
        current_zpos = zpos[j]
        if len(ypos) > 0:
            current_ypos = find_current_position_from_time(zpos_time[j], ypos,
                                                           ypos_time)
        else:
            current_ypos = 0.0
        actual_pos_list.append([current_ypos, current_zpos])
        actual_pos_times.append(zpos_time[j])

    # Construct a list of time-sorted positions
    actual_pos_list = np.array(actual_pos_list)
    actual_pos_times = np.array(actual_pos_times)
    pos_time_sort = actual_pos_times.argsort()
    actual_pos_times = actual_pos_times[pos_time_sort]
    actual_pos_list = actual_pos_list[pos_time_sort,:]
    if debug:
        print("actual_pos_times", actual_pos_times, np.shape(actual_pos_times))
        print("actual_pos_list", actual_pos_list, np.shape(actual_pos_list))


    # Now create the pixels to sample the positions in 2D.
    # For now we make them square.
    nx = ny = sampling
    xmin = min(np.amin(actual_pos_list[:, 0]), np.amin(actual_pos_list[:, 1]))
    xmax = max(np.amax(actual_pos_list[:, 0]), np.amax(actual_pos_list[:, 1]))
    # add padding
    dx = (xmax - xmin) / float(nx)
    xmin -= 0.5 * dx
    xmax += 0.5 * dx
    xe = np.linspace(xmin, xmax, nx + 1)
    dx = xe[1] - xe[0]
    if debug:
        print("xmin, xmax:", xmin, xmax)
        print("dx:", dx)
        print("x edges:", xe)
    ymin = xmin
    ymax = xmax
    ye = xe
    dy = dx

    # For each event, find positions of linear axes according to the time
    # stamps using numpy linear interpolation
    event_pos_x = np.interp(event_time, actual_pos_times, actual_pos_list[:,0])
    event_pos_y = np.interp(event_time, actual_pos_times, actual_pos_list[:,1])

    # Bin how many events arrived in a given pixel
    counts, yedges, xedges = np.histogram2d(event_pos_y, event_pos_x,
                                            bins=(ye, xe))

    # Durations are differences between pinhole time stamps
    durations = np.ediff1d(actual_pos_times) / 1.0e9

    # Allocate array to store normalization by pinhole time spent in each pixel
    normalization = np.zeros([ny, nx])

    # We need to normalize by the time pinhole spent in a particular pixel.
    # The method is the following:
    # - go through all the actual_pos_times and find the pixels where each x,y
    #   position falls into.
    # - we need to find how much of the duration between 2 successive time
    #   stamps A & B with spatial coordinates (xA, yA) and (xB, yB) needs to be
    #   allocated to each pixel:
    #     1. if the 2 timestamps are in the same pixel, the duration is dumped
    #        in that pixel
    #     2. if the 2 timestamps are in different pixels:
    #        - find the x and y distances (DeltaX, DeltaY) in pixels between
    #          the 2 points
    #        - for all the pixels in the rectangle of size (DeltaX, DeltaY),
    #          check if any edge of that pixel intersects the segment (A-B).
    #        - if a pixel has 2 intersections, the compute the distance
    #          between the 2 intersections to get the fraction of the duration
    #          that it should get allocated
    #        - if a pixel has only one intersection, it must contain either A
    #          or B
    #        - we need to make sure we remove duplicates if there are more than
    #          2 intersections, this will happen if (A-B) goes right through
    #          the corner of a pixel
    for i in range(len(actual_pos_times) - 1):
        if debug:
            print("i:", i)
        p0 = actual_pos_list[i, :]
        p1 = actual_pos_list[i+1, :]

        # Find pixels in mesh
        ix0 = int((p0[0] - xmin) / dx)
        iy0 = int((p0[1] - ymin) / dy)
        ix1 = int((p1[0] - xmin) / dx)
        iy1 = int((p1[1] - ymin) / dy)
        # Check if we are in the same pixel
        if ix0 == ix1 and iy0 == iy1:
            if debug:
                print("Segment is contained in pixel:", ix0, iy0)
            normalization[iy0, ix0] += durations[i]
        else:
            a_to_b = np.linalg.norm(p1 - p0)
            dist_sum = 0.0
            # Find intersects for all 4 pixel edges
            # TODO: we are duplicating computations here as an intersection on
            # one pixel edge is also an intersection for the neighbouring
            # pixel. Computations could thus be halved by using a smarter
            # scheme.
            if debug:
                print("ix0, ix1, iy0, iy1:", ix0, ix1, iy0, iy1)
            for k in range(min(iy0, iy1), max(iy0, iy1) + 1):
                for j in range(min(ix0, ix1), max(ix0, ix1) + 1):
                    if debug:
                        print("j, k:", j, k)
                    intersects = []
                    x0 = xmin + dx * j
                    y0 = ymin + dy * k
                    x1 = x0 + dx
                    y1 = y0 + dy
                    pix_corners = [np.array([x0, y0]), np.array([x1, y0]),
                                   np.array([x1, y1]), np.array([x0, y1])]
                    for l in range(4):
                        inter = get_intersect(p0, p1, pix_corners[l],
                                              pix_corners[(l + 1) % 4])
                        if inter is not None:
                            intersects.append(inter)
                    if debug:
                        print("Number of intersects:", len(intersects))
                        print("Intersects:", intersects)
                    if len(intersects) > 0:
                        if len(intersects) > 2:
                            if not np.array_equal(intersects[0],
                                                  intersects[1]):
                                intersects = intersects[:2]
                            elif not np.array_equal(intersects[0],
                                                    intersects[2]):
                                intersects = [intersects[0], intersects[2]]
                            else:
                                raise RuntimeError("Found more than 2 "
                                                   "intersects and no "
                                                   "duplicates.")
                        # Case where pixel contains one of the points.
                        # In principle, this can only happen if
                        # j == ix0 and k == iy0 or j == ix1 and k == iy1
                        dist = 0.0
                        if len(intersects) == 1:
                            if j == ix0 and k == iy0:
                                dist = np.linalg.norm(intersects[0] - p0)
                            elif j == ix1 and k == iy1:
                                dist = np.linalg.norm(intersects[0] - p1)
                            else:
                                raise RuntimeError("Only one intersect but x "
                                                   "and y indices are not "
                                                   "either i0 or i1.")
                        elif len(intersects) == 2:
                            dist = np.linalg.norm(intersects[0] -
                                                  intersects[1])
                        dist_sum += dist
                        normalization[k, j] += durations[i] * dist / a_to_b
            if not np.isclose(dist_sum, a_to_b):
                print("Warning: distances did not match:",
                      dist_sum, a_to_b, dist_sum/a_to_b)
            if debug:
                print("dist_sum, a_to_b, dist_sum/a_to_b:",
                      dist_sum, a_to_b, dist_sum/a_to_b)



    # Select only points where there is data, and then interpolate between
    # them with griddata
    sel = np.where(normalization > 0)
    count_rates = counts[sel] / normalization[sel]
    # Create grid of pixel centers
    xpoints, ypoints = np.meshgrid(0.5 * (xe[1:] + xe[:-1]),
                                   0.5 * (ye[1:] + ye[:-1]))
    points = np.transpose([xpoints[sel], ypoints[sel]])
    # Create high-resolution image dimensions
    x = np.linspace(xmin, xmax, 256)
    y = np.linspace(ymin, ymax, 256)
    grid_x, grid_y = np.meshgrid(x, y)
    # Use scipy interpolation function to make image
    z = griddata(points, count_rates, (grid_x, grid_y), method='linear')

    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)
    levels = np.linspace(vmin, vmax, nc)

    # Plot the image
    fig, ax = plt.subplots(1, 1)
    img = ax.contourf(x, y, z, levels=levels)
    cb = plt.colorbar(img, ax=ax)
    ax.set_xlabel("Distance x [mm]")
    ax.set_ylabel("Distance y [mm]")
    cb.ax.set_ylabel("Count rates (s$^{-1}$)")
    fig.savefig(outfile, bbox_inches="tight")
