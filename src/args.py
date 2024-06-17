import argparse
import motti


parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch_size", type=int, default=8
)
parser.add_argument(
    "--warmup_epochs", type=int, default=5
)
parser.add_argument(
    "--max_epochs", type=int, default=1,
)
parser.add_argument(
    "--from_epoch", type=int, default=0
)
parser.add_argument(
    "--num_workers", type=int, default=16
)
parser.add_argument(
    "--device_num", type=int, default=1
)
parser.add_argument(
    "--img_size", type=int, default=28,
)
parser.add_argument(
    "--ckpt", type=str, default=None,
)
parser.add_argument(
    "--ckpt_dir", type=str, default="."
)

parser.add_argument(
    "--log_dir", type=str, default="."
)

parser.add_argument(
    "--log_step", type=int, default=10
)

parser.add_argument(
    "--img_root", type=str, default="."
)

parser.add_argument(
    "--lr", type=float, default=float(1e-3)
)
parser.add_argument(
    "--fast", action="store_true", default=False,
)
parser.add_argument(
    "--save_training_output", action="store_true", default=False,
)
parser.add_argument(
    "--frozen", action="store_true", default=False,
)
parser.add_argument(
    "--accumulate_grad_batches", type=int, default=1
)
parser.add_argument(
    "--project", type=str, default="unnamed"
)
parser.add_argument(
    "--ps", type=str, default="postscript"
)
parser.add_argument(
    "-p", "--proportion", type=float, default=1.0
)
parser.add_argument(
    "--jitter_p", type=float, default=0.5
)
parser.add_argument(
    "--reuse", action="store_true", default=False,
)
opts, missing = parser.parse_known_args()

print(f"{opts = }")
print(f"{missing = }")
