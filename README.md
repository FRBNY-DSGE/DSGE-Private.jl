# dsgejl
FRBNY DSGE model

## Using MATLAB.jl on the RAN
Run the following:
```
mkdir ~/local/bin
echo $PATH | grep -q ~/local/bin && echo "okay"
ln -s /usr/local/bin/matlab14a ~/local/bin/matlab
echo 'LD_LIBRARY_PATH=/apps/matlab14a/sys/os/glnxa64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export MATLAB_HOME=/apps/matlab14a' >> ~/.bashrc
```

Don't even worry about why this is necessary.
