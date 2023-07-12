function fixup_mct {
    local mct_path="${1}"

    # TODO make PR to fix
    if [[ ! -e "${mct_path}/mct/Makefile.bak" ]]
    then
        sed -i".bak" "s/\$(AR)/\$(AR) \$(ARFLAGS)/g" "${mct_path}/mct/Makefile"
    fi

    if [[ ! -e "${mct_path}/mpeu/Makefile.bak" ]]
    then
        sed -i".bak" "s/\$(AR)/\$(AR) \$(ARFLAGS)/g" "${mct_path}/mpeu/Makefile"
    fi
}

# Fixes mct/mpeu to use ARFLAGS environment variable
# CIME will eventually have this fixed, remove this function when it does
fixup_mct "/src/E3SM/externals/mct"
for number_of_processors in 16 180; do
    cd /tmp
    cp /fv3net/workflows/prognostic_scream_run/tests/example_configs/scream_ne30pg2.yaml ${number_of_processors}.yaml
    sed -i -e "s/number_of_processors: 16/number_of_processors: $(printf "%d" $number_of_processors)/g" ${number_of_processors}.yaml
    mkdir -p rundir
    scream_run write-rundir ${number_of_processors}.yaml rundir
done