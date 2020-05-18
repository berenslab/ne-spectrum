import subprocess


def redo(deplist, exe="redo"):
    if isinstance(deplist, str):
        deplist = [deplist]
    return subprocess.call([exe] + deplist, close_fds=False)


def redo_ifchange(deplist):
    return redo(deplist, "redo-ifchange")
