import os
root = ""  # TODO: Change this to the project folder


def launch(abs_refac, surr_grad, dt):
    for i in range(3):
        id = f"shd_standard_{surr_grad}_{abs_refac}_{dt}_{i}"
        os.system(f"python {root}/scripts/supervised/train.py --root={root} --method=standard --abs_refac={abs_refac} --surr_grad={surr_grad} --name=shd --dt={dt} --id={id}")


launch(0, "mg", dt=2)
