import kfp
from kfp import dsl


@dsl.pipeline(name='my-pipeline')
def pipeline(grid_spec: str, rundir: str, output: str):

    compute_diags_op = dsl.ContainerOp(
        name="compute diagnostics",
        image="us.gcr.io/vcm-ml/fv3net:v0.2.0",
        command=["python", "workflows/prognostic_run_diags/save_prognostic_run_diags.py"],
        arguments=[
            "--grid-spec", grid_spec,
            rundir, output
        ]
    )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(pipeline, __file__ + '.yaml')
