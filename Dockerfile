# Dockerfile for testing the jet-hadron analysis
# We use the Overwatch base image so we don't have to deal with setting up ROOT.
# All we need to know is that the user is named "overwatch".
FROM rehlers/overwatch-base:py3.6.7
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

# TEMP: As of 12 Dec 2018, lz4 will fail to install because there is an empty "pkgconfig" directory
#       in the $ROOTSYS/lib directory. pkgconfig is needed for uproot (and thus, pachyderm). Fixed in
#       the lz4 dev branch, but it needs to be tagged.
RUN pip install --user --upgrade --no-cache-dir pkgconfig
# Install the jet-hadron analysis.
RUN pip install --user --upgrade --no-cache-dir -e .[tests,dev,docs]
