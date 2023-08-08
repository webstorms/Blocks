import os
root = ""  # TODO: Change this to the project folder


def launch(abs_refac, surr_grad, dt, method):
    for i in range(3):
        id = f"shd_{method}_{surr_grad}_{abs_refac}_{dt}_{i}"
        os.system(f"python {root}/scripts/supervised/train.py --root={root} --method={method} --abs_refac={abs_refac} --surr_grad={surr_grad} --name=shd --dt={dt} --id={id}")


for abs_refac in [10, 20, 30, 40, 50]:
    launch(abs_refac, "mg", 1, "blocks")
