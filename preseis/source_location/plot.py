import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse


def source_plot_with_ellipses(spatdist, md, row, frac, stations=None):
    sd = ["x", "y", "z"]
    sd.remove(md)
    coor_sel = {"space": sd, "space_T": sd}

    sp = spatdist.sel(coor_sel).transpose("source", row, ...)

    posterior = np.exp(sp["logposterior"])
    g = (posterior.sum([md]) / posterior.sum([md]).max(sd)).plot(
        x=sd[0], y=sd[1], col="source", row=row, cmap="BuGn", add_colorbar=False
    )

    spstack = sp.stack({"flat": [row, "source"]})

    for i, ax in enumerate(g.axs.flat):
        sploc = spstack.isel({"flat": i})

        mn = sploc["location_mean"]
        cov = sploc["covariance_integral"]

        el = covariance_ellipse(mn, cov, frac, edgecolor="b")
        ax.add_artist(el)

        astats = sploc["station"].where(sploc["active_stations"], drop=True)
        if stations is not None:
            st = stations.sel({"station": astats})
            ax.scatter(
                st[sd[0]],
                st[sd[1]],
                color="red",
                marker="v",
                label="station",
            )

        source_loc = {
            "x": sploc["source_X"],
            "y": sploc["source_Y"],
            "z": sploc["source_Z"],
        }
        ax.scatter(
            source_loc[sd[0]],
            source_loc[sd[1]],
            marker="+",
            s=50,
            color="black",
            label="source",
        )

    plt.legend(loc="lower left", fancybox=True, fontsize=8)


def covariance_ellipse(centre, cov, fraction, **kwargs):
    """
    Adapted from https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.arctan2(vy, vx)

    # width and height of ellipse to draw based on
    # fraction inside
    ndof = len(eigvals)
    nstd = np.sqrt(chi2.ppf(fraction, ndof))
    width, height = 2 * nstd * np.sqrt(eigvals)

    return Ellipse(
        xy=centre,
        width=width,
        height=height,
        angle=np.degrees(theta),
        lw=1,
        facecolor="none",
        **kwargs
    )
