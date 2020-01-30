import fv3gfs

if __name__ == "__main__":
    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        fv3gfs.step_dynamics()
        fv3gfs.step_physics()
        fv3gfs.save_intermediate_restart_if_enabled()
    fv3gfs.cleanup()
