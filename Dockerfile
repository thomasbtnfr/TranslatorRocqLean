FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

RUN apt-get update -yq \
&& apt-get install curl -yq \
&& yes '' | bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"  \
&& apt-get install unzip \
&& apt-get install bubblewrap -yq \
&& apt-get install make -yq \
&& apt-get install libgmp-dev -yq \
&& apt-get install git -yq

RUN apt-get update -yq \
 && apt-get install -yq --no-install-recommends \
       build-essential gcc g++ m4 pkg-config

# Install rocq
ENV OPAMROOT=/opt/opam
ENV PATH=$OPAMROOT/default/bin:$PATH

RUN mkdir -p /opt/opam && chmod -R 777 /opt/opam \
&& opam init --disable-sandboxing -yq \
&& opam switch create with-rocq 4.14.2 -yq \
&& eval $(opam env) \
&& opam pin add -y rocq-prover 9.0.0

RUN eval $(opam env)

# Install lean
ENV ELAN_HOME=/opt/.elan
ENV PATH=$ELAN_HOME/bin:$PATH
RUN mkdir -p $ELAN_HOME && chmod -R 777 $ELAN_HOME

RUN curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y

RUN lake +leanprover/lean4:nightly-2024-04-24 new DebugLean math \
&& cd DebugLean \
&& lake update \
&& lake build

COPY ./TranslatorRocqLean .

# Python environment
RUN pip install -e . \
&& pip install pyyaml \
&& git clone https://github.com/idriscnrs/idr_torch.git \
&& cd idr_torch \
&& pip install . \
&& pip install mlflow \
&& pip install torchmetrics \
&& pip install transformers \
&& pip install datasets \
&& pip install gradio \
&& pip install accelerate \
&& pip install trl \
&& pip install vllm \
&& cd .. \
&& git clone https://github.com/idriscnrs/idr_accelerate.git \
&& cd idr_accelerate \
&& pip install . \
&& pip install deepspeed

# Necessary for deepspeed (need CUDA Compiler and not only CUDA Runtime)
RUN conda install -c nvidia cuda-compiler

WORKDIR /TranslatorRocqLean
