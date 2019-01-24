# Dockerfile for testing the jet-hadron analysis
# We use the Overwatch base image so we don't have to deal with setting up ROOT.
# All we need to know is that the user is named "overwatch".
# Set the python version here so that we can use it to set the base image.
ARG PYTHON_VERSION=3.7.1
FROM rehlers/overwatch-base:py${PYTHON_VERSION}
LABEL maintainer="Raymond Ehlers <raymond.ehlers@cern.ch>, Yale University"

# Setup environment
ENV ROOTSYS="/opt/root"
ENV PATH="${ROOTSYS}/bin:/home/overwatch/.local/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROOTSYS}/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${ROOTSYS}/lib:${PYTHONPATH}"

# Setup the jet-hadron package
ENV JET_HADRON_ROOT /opt/jetHadron
# We intentionally make the directory before setting it as the workdir so the directory is made with user permissions
# (workdir always creates the directory with root permissions)
RUN mkdir -p ${JET_HADRON_ROOT}
WORKDIR ${JET_HADRON_ROOT}

# Copy the jet-hadron analysis into the image.
COPY --chown=overwatch:overwatch . ${JET_HADRON_ROOT}

# Necessary for iminuit, probfit
RUN pip install --user --upgrade --no-cache-dir numpy cython
# Install the jet-hadron analysis.
RUN pip install --user --upgrade --no-cache-dir -e .[tests,dev,docs]
