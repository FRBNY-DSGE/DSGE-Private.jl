# DSGE.jl
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/FRBNY-DSGE/DSGE.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


## Using MATLAB.jl on the RAN

Run the following at the shell:
```
mkdir ~/local/bin
ln -s /usr/local/bin/matlab14a ~/local/bin/matlab
which matlab >/dev/null || echo 'add ~/local/bin to your $PATH'
```

And add the following to your .bashrc:
```
LD_LIBRARY_PATH=/apps/matlab14a/sys/os/glnxa64:$LD_LIBRARY_PATH
export MATLAB_HOME=/apps/matlab14a
```

Don't even worry about why this is necessary.
