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
for number_of_processers in 16 180; do
    cd /tmp
    cp /src/prognostic_scream_run/tests/example_configs/scream_ne30pg2.yaml ${number_of_processers}.yaml
    sed -i -e "s/number_of_processers: 16/number_of_processers: $(printf "%d" $number_of_processers)/g" ${number_of_processers}.yaml
    mkdir -p rundir
    write_scream_run_directory ${number_of_processers}.yaml rundir
done