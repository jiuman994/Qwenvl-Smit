import os
import subprocess
import sys


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    requirements = os.path.join(here, "requirements.txt")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements])


if __name__ == "__main__":
    main()
