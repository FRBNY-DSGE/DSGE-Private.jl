# DSGE.jl
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/FRBNY-DSGE/DSGE.jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


## Using Pkg behind the RAN firewall.

Run the following at the shell:
```
git config --global url."https://".insteadOf git://
echo 'unset SSH_ASKPASS' >> .bashrc
```

## Install DSGE.jl into the correct directory.
From Julia:
```
julia> Pkg.clone("https://user.name@github.com/FRBNY-DSGE/DSGE.jl.git")
```

Now, do all of your development in `~/.julia/v0.3/DSGE.jl`.

## Use MATLAB.jl on the RAN
And add the following to your .bashrc:
```
LD_LIBRARY_PATH=/apps/matlab14a/sys/os/glnxa64:$LD_LIBRARY_PATH
export MATLAB_HOME=/apps/matlab14a
```

This is because Matlab provides it's own version of some libraries that it
provides paths to at runtime that are not otherwise available on the system.

