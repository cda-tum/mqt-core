# MQT-CATALYST Docker Image

This directory provides a docker image that can be used to install Catalyst and its dependencies on a
clean, isolated environment. The image is based on the Ubuntu Noble and installs Catalyst from source.

It also sets additionally required environment variables and installs Python 3.10.

## How to use the Image

If you haven't already, install Docker on your machine. E.g. on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install docker
```

To build the image, you can use the provided `build.sh` script:

```bash
cd docker
./build.sh
```

This will build the image as `mqt-catalyst:latest`. The process may take a while (~20 minutes), as it will download and compile
Catalyst from source.

Once the image is built, there are multiple ways to use it.

### Running through Terminal

The simplest way to run the image is through the terminal. You can use the provided `run.sh` script _from within this directory_:

```bash
./run.sh
```

to start a new container in your terminal. It will open the container's bash shell, where you can run commands. Once you're done,
the docker container will be stopped and removed. The run script will also mount the `mqt-mlir` directory into the container.
This way, you can edit code from your preferred IDE and immediately access it from the container.

Note that the container runs as `root`, so do not use any sensitive information in it, if you can avoid it. Also, files
created from inside the container may require root access to modify from the host machine.

### Running through VSCode

If you use Visual Studio Code, you can use the provided `devcontainer.json` file to open the project in a container.
This requires the `Dev Containers` extension. Doing this allows you to run the command (CTRL + P) `Dev Containers: Reopen in Container`
from this repository's root directory. This will open a new VSCode window with the project running inside the container.
This allows you to access the installed dependencies from your IDE for autocomplete purposes.

### Testing the installation

Once you're done, you can test the installation from this repositories root directory (if you connected through the terminal,
first change to the directory with `cd /home/mqt/mqt-mlir`):

```bash
python3 test/programs/bell-pair.py
```

or

```bash
python3 test/programs/control-flow.py
```

These python scripts create simple quantum programs that are executed on a simulated device. Also, running them
will generate their MLIR representation and print it to the console.
