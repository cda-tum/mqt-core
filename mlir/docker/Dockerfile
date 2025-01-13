FROM ubuntu:noble

# Install important programs
RUN apt-get update
RUN apt-get install -y git
RUN git clone --recurse-submodules --shallow-submodules https://github.com/PennyLaneAI/catalyst.git

RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y clang
RUN apt-get install -y make
RUN apt-get install -y cmake
RUN apt-get install -y nano
RUN apt-get install -y curl
RUN apt install lld ccache libomp-dev ninja-build -y

# Set the PATH environment variable
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin/
ENV LANG=C.UTF-8
ENV CC=clang
ENV CXX=clang++
ENV MLIR_DIR=/catalyst/mlir/llvm-project/build/lib/cmake/mlir/

WORKDIR /catalyst

# Install Catalyst
RUN pip config set global.break-system-packages true
RUN pip install -r requirements.txt --break-system-packages
RUN make all
WORKDIR /

RUN mkdir -p /home/mqt
COPY custom_bashrc /home/mqt/custom_bashrc
COPY start.sh /start.sh
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["source /start.sh"]
