# Walk script port (item 9)

These scripts are copies of `estimation-AI/work/rtwalk104/scripts` and `rtwalk114/scripts` made on 20 Nov 2026. The originals and their running jobs were not changed.

The five scripts that used `SMC.INITIAL_C_BLOCKS[]` now set `:smc_initial_c_blocks` on the model passed to `DSGE.estimate`. `setup.jl` requires the merged mainline DSGE and SMC APIs and rejects the old pinned package clones. The existing `:smc_reuse_system_new_params` setting uses the implementation already merged into DSGE main. The wrappers accept `WALK_ENV_DIR` and `WALK_DEPOT_DIR` to select an isolated environment and cache.

The scripts still read their pinned inputs from the two original work directories. Before launching a migrated walk, provide an environment whose Manifest points to the mainline packages, set `WALK_ENV_DIR` and `WALK_DEPOT_DIR`, and run the walk's dry-run and smoke checks. Do not point a migrated job at an existing production step directory; its cloud files are from the pinned version. The production migration remains deferred in the team status note of 17 Nov 2026.
