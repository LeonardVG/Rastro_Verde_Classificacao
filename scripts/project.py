import sys

import dataset
import model

CMDS = {
    "DATASET": dataset,
    "MODEL": model
}


def help() -> None:
    print(f"usage: {sys.argv[0]} <command>")
    print("commands: \n")
    print(*[f'\t{i}' for i in CMDS.keys()], sep="\n")


def main() -> None:
    if len(sys.argv) == 1:
        help()
        return

    cmd = sys.argv.pop(1)
    if cmd.upper() in CMDS:
        CMDS[cmd.upper()].run()
    else:
        help()


if __name__ == '__main__':
    main()
