#!/usr/bin/env python

# Contains Jet-Hadron analysis parameters
#
# As well as ways to access that information

from builtins import range

import math
import enum
import collections
import logging
logger = logging.getLogger(__name__)

# Bins
# eta is absolute value!
etaBins = [0, 0.4, 0.6, 0.8, 1.2, 1.5]
trackPtBins = [0.15, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0]
jetPtBins = [15.0, 20.0, 40.0, 60.0, 200.0]
phiBins = [-1.*math.pi/2., math.pi/2., 3.*math.pi/2]

########
# Utility functions
#######
def iterateOverPtBins(name, bins, config = None):
    """ Create a generator of the bins in a requested list.

    Bin skipping should be specified as:

    ```
    config = {
        "skipPtBins" : {
            "name" : [bin1, bin2]
        }
    }
    ```

    Args:
        name (str): Name of the skip bin entries in the config.
        bins (list): Bin edges for determining the bin indices.
        config (dict): Containing information regarding bins to skip, as specified above.
    """
    # Create a default dict if none is available
    if not config:
        config = {}
    skipPtBins = config.get("skipPtBins", {}).get(name, [])
    # Sanity check on skip pt bins
    for val in skipPtBins:
        if val >= len(bins) - 1:
            raise ValueError(val, "Pt bin to skip {val} is outside the range of the {name} list".format(val = val, name = name))

    for ptBin in range(0, len(bins)-1):
        if ptBin in skipPtBins:
            continue

        yield ptBin

def iterateOverJetPtBins(config = None):
    """ Iterate over the available jet pt bins. """
    return iterateOverPtBins(config = config, name = "jet", bins = jetPtBins)

def iterateOverTrackPtBins(config = None):
    """ Iterate over the available track pt bins. """
    return iterateOverPtBins(config = config, name = "track", bins = trackPtBins)

def iterateOverJetAndTrackPtBins(config = None):
    """ Iterate over all possible combinations of jet and track pt bins. """
    for jetPtBin in iterateOverJetPtBins(config):
        for trackPtBin in iterateOverTrackPtBins(config):
            yield (jetPtBin, trackPtBin)

def useLabelWithRoot(label):
    """ Function to automatically convert LaTeX to something that is mostly ROOT compatiable.

    Args:
        label (str): Label to be converted.
    Returns:
        str: Convert label.
    """
    # Remove "$" and map "\" -> "#""
    return label.replace("$", "").replace("\\", "#")

#########
# Parameter information (access and display)
#########
class aliceLabelType(enum.Enum):
    """ ALICE label types. """
    workInProgress = "ALICE Work in Progress"
    preliminary = "ALICE Preliminary"
    final = "ALICE"
    thesis = "This thesis"

    def __str__(self):
        """ Return the value. This is just a convenience function.

        Note that this is backwards of the usual convention of returning the name, but the value is
        more meaningful here. The name can always be accessed with `.name`. """
        return str(self.value)

    def str(self):
        """ Helper for __str__ to allow it to be accessed the same as the other str functions. """
        return self.__str__()

def aliceLabel(labelType):
    """ Determine the ALICE label based on the label type.

    Args:
        labelType (JetHParams.aliceLabelType): The type of ALICE label desired.
    Returns:
        str: The ALICE label.
    """
    # Convert if necessary
    if isinstance(labelType, str):
        labelType = aliceLabelType[labelType]

    # Determine label
    labels = {aliceLabelType.workInProgress : "ALICE Work in Progress",
              aliceLabelType.preliminary : "ALICE Preliminary",
              aliceLabelType.final : "ALICE",
              aliceLabelType.thesis : "This thesis"}

    return labels[labelType]

def systemLabel(collisionSystem, eventActivity = None, energy = 2.76):
    """ Generates the collision system, event activity, and energy label.

    Args:
        collisionSystem (JetHParams.collisionSystem): The collision system.
        eventActivity (JetHParams.eventActivity): The event activity selection.
        energy (float): The collision energy
    Returns:
        str: Label for the entire system, combining the avaialble information.
    """
    # Use as proxy of CollisionSystem so we don't need to import JetHUtils
    # NOTE: Usually, "Pb--Pb" is used in latex, but ROOT won't render it properly...
    systems = {"pp" : "pp",
               "PbPb" : r"Pb\mbox{-}Pb"}
    if eventActivity is None:
        eventActivity = collisionSystem
    eventActivities = {"inclusive" : "",
                     "central" : r",\:0\mbox{-}10\mbox{\%}",
                     "semiCentral" : r",\:30\mbox{-}50\mbox{\%}"}
    # Adding for backwards compatibility
    # TODO: Remove this values
    eventActivities["pp"] = eventActivities["inclusive"]
    eventActivities["PbPb"] = eventActivities["central"]
    logger.debug("eventActivity: {}".format(eventActivities[eventActivity]))

    systemLabel = r"$\mathrm{%(system)s}\:\sqrt{s_{\mathrm{NN}}} = %(energy)s\:\mathrm{TeV}%(eventActivity)s$" % {"energy" : energy,
            "eventActivity" : eventActivities[eventActivity],
            "system" : systems[collisionSystem]}

    logger.debug("systemLabel: {}".format(systemLabel))

    return systemLabel

def generatePtRangeString(arr, binVal, lowerLabel, upperLabel, onlyShowLowerValueForLastBin = False):
    """ Generate string to describe pt ranges for a given list.

    Args:
        arr (list): Bin edges for use in determining the values lower and upper values
        binVal (int): Generate the range for this bin
        lowerLabel (str): Subscript label for pT
        upperLabel (str): Superscript labe for pT
        onlyShowLowerValueForLastBin (bool): If True, skip show the upper value.
    Returns:
        str: The pt range label
    """
    # Cast as string so we don't have to deal with formatting the extra digits
    lower = "%(lower)s < " % {"lower" : arr[binVal]}
    upper = " < %(upper)s" % {"upper": arr[binVal+1]}
    if onlyShowLowerValueForLastBin and binVal == len(arr) - 2:
        upper = ""
    ptRange = r"$%(lower)sp_{%(lowerLabel)s}^{%(upperLabel)s}%(upper)s\:\mathrm{GeV/\mathit{c}}$" % {"lower" : lower, "upper": upper, "lowerLabel" : lowerLabel, "upperLabel" : upperLabel}

    return ptRange

def generateJetPtRangeString(jetPtBin):
    """ Generate a label for the jet pt range based on the jet pt bin.

    Args:
        jetPtBin (int): Jet pt bin
    Returns:
        str: Jet pt range label
    """
    return generatePtRangeString(arr = jetPtBins,
            binVal = jetPtBin,
            lowerLabel = r"\mathrm{T \,unc,jet}",
            upperLabel = r"\mathrm{ch+ne}",
            onlyShowLowerValueForLastBin = True)

def generateTrackPtRangeString(trackPtBin, ptBins = None):
    """ Generate a label for the track pt range based on the track pt bin.

    Args:
        trackPtBin (int): Track pt bin.
        ptBins (list): Track pt bins. Defaults to the default jet-h track pt bins if not specified.
    Returns:
        str: Track pt range label.
    """
    return generatePtRangeString(arr = ptBins if ptBins is not None else trackPtBins,
            binVal = trackPtBin,
            lowerLabel = r"\mathrm{T}",
            upperLabel = r"\mathrm{assoc}")

def jetPropertiesLabel(jetPtBin):
    """ Return the jet finding properties based on the jet pt bin.

    Args:
        jetPtBin (int): Jet pt bin
    Returns:
        tuple: (jetFinding, constituentCuts, leadingHadron, jetPt)
    """
    jetFinding = r"$\mathrm{anti\mbox{-}k}_{\mathrm{T}}\;R=0.2$"
    constituentCuts = r"$p_{\mathrm{T}}^{\mathrm{ch}}\:\mathrm{\mathit{c},}\:\mathrm{E}_{\mathrm{T}}^{\mathrm{clus}} > 3\:\mathrm{GeV}$"
    leadingHadron = r"$p_{\mathrm{T}}^{\mathrm{lead,ch}} > 5\:\mathrm{GeV/\mathit{c}}$"
    jetPt = generateJetPtRangeString(jetPtBin)
    return (jetFinding, constituentCuts, leadingHadron, jetPt)

class collisionEnergy(enum.Enum):
    """ Define the available collision system energies. """
    twoSevenSix = 2.76
    fiveZeroTwo = 5.02

    def __str__(self):
        """ Returns a string of the value. """
        return str(self.value)

    def str(self):
        """ Helper for __str__ to allow it to be accessed the same as the other str functions. """
        return self.__str__()

class collisionSystem(enum.Enum):
    """ Define the collision system """
    NA = -1
    pp = 0
    # We want to alias this value, but also it should generally be treated the same as pp
    embedPP = 0
    pPb = 1
    PbPb = 2

    #def __str__(self):
    #    """ Return the name of the value without the appended "k". This is just a convenience function """
    #    return str(self.name.replace("k", "", 1))
    def __str__(self):
        return self.name

    def str(self):
        """ Helper for __str__ to allow it to be accessed the same as the other str functions. """
        return self.__str__()

    def filenameStr(self):
        """ """
        return self.name

class eventActivity(enum.Enum):
    """ Define the event activity.

    Object value are of the form (index, (centLow, centHigh)), where index is the expected
    enumeration index, and cent{low,high} define the low and high values of the centrality.
    -1 is defined as the full range!
    """
    inclusive = (0, (-1, -1))
    central = (1, (0, 10))
    semiCentral = (2, (30, 50))

    def __init__(self, index, activityRange):
        self.index = index
        self.activityRange = activityRange

    def getRange(self):
        """ """
        return self.activityRange

    def __str__(self):
        """ """
        return str(self.name)

    def str(self):
        """ Helper function to return str by calling explicitly """
        return self.__str__()

class leadingHadronBiasType(enum.Enum):
    """ Leading hadron bias type """
    NA = -1
    track = 0
    cluster = 1
    both = 2

    #def __str__(self):
    #    """ Return the name of the value without the appended "k". This is just a convenience function """
    #    return str(self.name.replace("k", "", 1))

    def __str__(self):
        """ Return the type and value, such as "cluster6" or "track5". """
        return "{name}{value}".format(name = self.name, value = self.value)

    def str(self):
        """ Helper function to return str by calling explicitly """
        return self.__str__()

class leadingHadronBias(enum.Enum):
    NA = -1
    track = 5
    clusterSemiCentral = 6
    clusterCentral = 10

    # TODO: Implement the value of the leading hadron bias, which depends
    #       on the type, collision energy, and event activity
    def get(name, collisionEnergy, eventActivity):
        pass

selectedAnalysisOptions = collections.namedtuple("selectedAnalysisOptions", ["energy",
            "collisionSystem",
            "eventActivity",
            "leadingHadronBiasType"])

class eventPlaneAngle(enum.Enum):
    """ Selects the event plane angle in the sparse. """
    kAll = 0
    kInPlane = 1
    kMidPlane = 2
    kOutOfPlane = 3

    def baseString(self):
        """ Turns kOutOfPlane into "OutOfPlane" """
        return self.name.replace("k", "", 1)

    def __str__(self):
        """ Turns kOutOfPlane into "outOfPlane" """
        tempStr = self.filenameStr()
        tempStr = tempStr[:1].lower() + tempStr[1:]
        return tempStr

    def str(self):
        """ Helper for __str__ to allow it to be accessed the same as the other str functions. """
        return self.__str__()

    def filenameStr(self):
        """ Turns kOutOfPlane into "eventPlaneOutOfPlane" """
        return "eventPlane{}".format(self.baseString())

    def displayStr(self):
        """ Turns kOutOfPlane into "Out Of Plane". """
        tempStr = self.filenameStr()
        tempList = re.findall('[A-Z][^A-Z]*', tempStr)
        return " ".join(tempList)

class qVector(enum.Enum):
    """ Selection based on the Q vector. """
    all = 0
    top10 = 1
    bottom10 = 2

    def __str__(self):
        """ TODO: Returns the selection range. """
        return self.name

    def str(self):
        """ Helper for __str__ to allow it to be accessed the same as the other str functions. """
        return self.__str__()

    def filenameStr(self):
        """ Helper class that returns a filename self value. """
        return "qVector{}".format(self.str().capitalize())
