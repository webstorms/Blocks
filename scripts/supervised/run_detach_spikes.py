import os
root = ""  # TODO: Change this to the project folder


def launch(abs_refac, surr_grad, dt, detach_spike_grad):
    for i in range(3):
        id = f"shd_blocks_{surr_grad}_{abs_refac}_{dt}_{detach_spike_grad}_{i}"
        os.system(f"python {root}/scripts/supervised/train.py --root={root} --method=blocks --abs_refac={abs_refac} --surr_grad={surr_grad} --name=shd --dt={dt} --id={id} --detach_spike_grad={detach_spike_grad}")


launch(30, "mg", dt=2, detach_spike_grad=False)
launch(30, "fast_sigmoid", dt=2, detach_spike_grad=False)
launch(30, "box_car", dt=2, detach_spike_grad=False)
