import matplotlib.pyplot as plt


def custcallback(error=None, x=None, nfev=None, stepsize=None, fev_list=None):
    """
    custom callback to follow optimizer progress
    """
    print(
        "x={}   error={}  function evaluated {} times step taken: {}".format(
            x,
            error,
            nfev,
            stepsize,
        ),
    )
    x_plot = [fev[0] for fev in fev_list]
    y_plot = [fev[1] for fev in fev_list]
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(x_plot, y_plot, "o", color="royalblue")
    ax.plot(x_plot, y_plot, "-", color="royalblue")
    for fev in fev_list:
        ax.annotate(fev[2], (fev[0], fev[1]), ha="center")

    # formatting
    ax.set_xlim(left=0)
    ax.set(ylabel="cost (log scale)")
    ax.set(xlabel="variable value")
    ax.set_xticks(x_plot)
    ax.set_yscale("log")

    plt.show()


def custmin(
    f,
    guess: float,
    nfev_max: int = 0,
    steps: list = [1],
    min_x: float = 1,  # some values can make sim error (eg. 0 counters)
    callback=custcallback,
    tol: float = 0,
):
    """
    custom minimizer for a convex function
    look for local minima for a succession of steps
    can use callback
    """

    nfev_max = float("inf") if nfev_max == 0 else nfev_max
    nfev = 0

    for idx, step in enumerate(steps):
        # initialize
        nfev_step = 0
        if idx == 0:
            stopped = False
            fev_list = []
            bestx, besty = guess, f(guess)
            status = "initialization"

            fev_list.append((bestx, besty, status))
            if callback is not None:
                callback(
                    error=besty,
                    x=bestx,
                    nfev=nfev,
                    stepsize=step,
                    fev_list=fev_list,
                )
        else:
            stopped = False
            # we keep previous step fev_list and bestx, besty
        nfev_step += 1
        nfev += 1

        # main big loop resulting in minimum
        direction_decided = False
        while not stopped and nfev < nfev_max:
            # check after one step
            testx = max(min_x, bestx + step)
            testy = f(testx)
            nfev_step += 1
            nfev += 1

            # if tol reached, finished
            if testy <= tol:
                besty = testy
                bestx = testx
                stopped = True
                status = "tolerance reached"
                fev_list.append((testx, testy, status))
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nfev=nfev,
                        stepsize=step,
                        fev_list=fev_list,
                    )
                break

            # if plateau, finished
            if testy == besty:
                stopped = True
                status = "plateau found"
                fev_list.append((testx, testy, status))
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nfev=nfev,
                        stepsize=step,
                        fev_list=fev_list,
                    )
                break

            # if improved, we keep that direction
            if testy < besty:
                besty = testy
                bestx = testx
                direction_decided = True
                status = "one more step"
                fev_list.append((testx, testy, status))
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nfev=nfev,
                        stepsize=step,
                        fev_list=fev_list,
                    )

            # if not improved we stop and go to next stepsize
            if testy > besty and direction_decided:
                status = "local minima found"
                fev_list.append((testx, testy, status))
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nfev=nfev,
                        stepsize=step,
                        fev_list=fev_list,
                    )
                break

            # if not improved, we change direction
            if testy > besty and not direction_decided:
                step = -step
                direction_decided = True
                status = "change dir"
                fev_list.append((testx, testy, status))
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nfev=nfev,
                        stepsize=step,
                        fev_list=fev_list,
                    )

    return bestx, besty, status
