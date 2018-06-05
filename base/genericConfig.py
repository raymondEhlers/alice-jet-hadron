#!/usr/bin/env python

# Analysis configuration base module. For usage information,
# see JetHConfig

# py2/3
import future.utils
from future.utils import iteritems
from future.utils import itervalues

import string
import ruamel.yaml

import logging
logger = logging.getLogger(__name__)

def loadConfiguration(filename):
    """ Load an analysis configuration from a file.

    Args:
        filename (str): Filename of the YAML configuration file.
    Returns:
        dict-like: dict-like object containing the loaded configuration
    """
    # Initialize the YAML object in the roundtrip mode
    # NOTE: "typ" is a not a typo. It stands for "type"
    yaml = ruamel.yaml.YAML(typ = "rt")

    with open(filename, "r") as f:
        config = yaml.load(f)

    return config

def overrideOptions(config, selectedOptions, setOfPossibleOptions, configContainingOverride = None):
    """ Determine override options for a particluar configuration, searching following the order specified
    in selectedOptions.

    For the example config,
    ```
    config:
        value: 3
        override:
            2.76:
                track:
                    value: 5
    ```
    value will be assigned the value 5 if we are at 2.76 TeV with a track bias, regardless of the event
    activity or leading hadron bias. The order of this configuration is specified by the order of the
    selectedOptions passed. The above example configuration is from the jet-hadron analysis.

    Args:
        config (CommentedMap): The dict-like configuration from ruamel.yaml which should be overridden.
        selectedOptions (tuple): The selected analysis options. They will be checked in the order with which
            they are passed, so make certain that it matches the order in the configuration file!
        setOfPossibleOptions (tuple of enums): Possible options for the override value categories.
        configContainingOverride (CommentedMap): The dict-like config containing the override options in a map called
            "override". If it is not specified, it will look for it in the main config.
    Returns:
        dict-like object: The updated configuration
    """
    if configContainingOverride is None:
        configContainingOverride = config
    overrideOptions = configContainingOverride.pop("override")
    overrideDict = fillOverrideOptions(selectedOptions, overrideOptions, setOfPossibleOptions)
    logger.debug("overrideDict: {}".format(overrideDict))

    # Set the configuration values to those specified in the override options
    # Cannot just use update() on config because we need to maintain the anchors.
    for k, v in iteritems(overrideDict):
        # Check if key is there and if it is not None! (The second part is imporatnt)
        if k in config:
            try:
                # We can't check for the anchor - we just have to try to access it.
                # However, we don't actually care about the value. We just want to
                # preserve it if it is exists.
                config[k].anchor
                logger.debug("type: {}, k: {}".format(type(config[k]), k))
                if isinstance(config[k], list):
                    # Clear out the existing list entries
                    del config[k][:]
                    if isinstance(overrideDict[k], future.utils.string_types):
                        # We have to treat str carefully because it is an iterable, but it will be expanded as
                        # individual characters if it's treated the same as a list, which is not the desired
                        # behavior! If we wrap it in [], then it will be treated as the only entry in the list
                        config[k].append(overrideDict[k])
                    else:
                        # Here we just assign all entries of the list to all entries of overrideDict[k]
                        config[k].extend(overrideDict[k])
                elif isinstance(config[k], dict):
                    # Clear out the existing entries because we are trying to replace everything
                    # Then we can simply update the dict with our new values
                    config[k].clear()
                    config[k].update(overrideDict[k])
            except AttributeError:
                # If no anchor, just overwrite the value at this key
                config[k] = v
        else:
            raise KeyError(k, "Trying to override key \"{}\" that it is not in the config.".format(k))
    
    return config

def simplifyDataRepresentations(config):
    """ Convert one entry lists to the scalar value

    This step is necessary because anchors are not kept for scalar values - just for lists and dicts.
    Now that we are done with all of our anchor refernces, we can convert these single entry lists to
    just the scalar entry, which is more usable.
    
    Args:
        config (CommentedMap): The dict-like configuration from ruamel.yaml which should be simplified.
    Returns:
        dict-like object: The updated configuration
    """
    for k,v in iteritems(config):
        if v and isinstance(v, list) and len(v) == 1:
            logger.debug("v: {}".format(v))
            config[k] = v[0]

    return config

def fillOverrideOptions(selectedOptions, overrideOptions, setOfPossibleOptions = ()):
    """ Reusrively extract the dict described in overrideOptions().

    In particular, this searches for selected options in the overrideOptions dict.
    It stores only the override options that are selected.

    Args:
        selectedOptions (tuple): The options selected for this analysis, in the order defined used
            with overrideOptions() and in the configuration file.
        overrideOptions (CommentedMap): dict-like object returned by ruamel.yaml which contains the options that 
            should be used to override the configuration options.
        setOfPossibleOptions (tuple of enums): Possible options for the override value categories.
    """
    overrideDict = {}
    for option in overrideOptions:
        # We need to cast the option to a string to effectively compare to the selected option,
        # since only some of the options will already be strings
        if str(option) in list(map(lambda opt: opt.str(), selectedOptions)):
            overrideDict.update(fillOverrideOptions(selectedOptions, overrideOptions[option], setOfPossibleOptions))
        else:
            logger.debug("overrideOptions: {}".format(overrideOptions))
            # Look for whether the key is one of the possible but unselected options.
            # If so, we haven't selected it for this analysis, and therefore they should be ignored.
            # NOTE: We compare both the names and value because sometimes the name is not sufficient,
            #       such as in the case of the energy (because a number is not allowed to be a field name.)
            foundAsPossibleOption = False
            for possibleOptions in setOfPossibleOptions:
                # Same type of comparison as above, but for all possible options instead of the selected options.
                if str(option) in list(map(lambda opt: opt.str(), possibleOptions)):
                    foundAsPossibleOption = True
                # Below is more or less equivalent to the above (although .str() hides the details or whether
                # we should compare to the name or the value in the enum and only compares against the designated value).
                #for possibleOpt in possibleOptions:
                    #if possibleOpt.name == option or possibleOpt.value == option:
                        #foundAsPossibleOption = True

            if not foundAsPossibleOption:
                # Store the override value, since it doesn't correspond with a selected option or a possible option
                # and therefore must be an option that we want to override.
                logger.debug("Storing override option \"{}\", with value \"{}\"".format(option, overrideOptions[option]))
                overrideDict[option] = overrideOptions[option]
            else:
                logger.debug("Found option \"{}\" as possible option, so skipping!".format(option))

    return overrideDict

class formattingDict(dict):
    """ Dict to handle missing keys when formatting a string. It returns the missing key
    for later use in formatting. See: https://stackoverflow.com/a/17215533 """
    def __missing__(self, key):
        return "{" + key + "}"

def applyFormattingDict(obj, formatting):
    """ Recursively apply a formatting dict to all strings in a configuration.

    Note that it skips applying the formatting if the string appears to contain latex (specifically,
    if it contains an "$"), since the formatting fails on nested brackets.

    Args:
        obj (dict): Some configuration object to recursively applying the formatting to.
        formatting (dict): String formatting options to apply to each configuration field.
    Returns:
        dict: Configuration with formatting applied to every field.
    """
    #logger.debug("Processing object of type {}".format(type(obj)))

    if isinstance(obj, str):
        # Apply the formatting options to the string.
        # We explicitly allow for missing keys. They will be kept so they can be filled later.
        # see: https://stackoverflow.com/a/17215533
        # If a more sophisticated solution is needed,
        # see: https://ashwch.github.io/handling-missing-keys-in-str-format-map.html
        # Note that we can't use format_map becuase it is python 3.2+ only.
        # The solution below works in py 2/3
        if not "$" in obj:
            obj = string.Formatter().vformat(obj, (), formattingDict(**formatting))
        #else:
        #    logger.debug("Skipping str {} since it appears to be a latex string, which may break the formatting.".format(obj))
    elif isinstance(obj, dict):
        for k, v in iteritems(obj):
            # Using indirect access to ensure that the original object is updated.
            obj[k] = applyFormattingDict(v, formatting)
    elif isinstance(obj, list):
        for i, el in enumerate(obj):
            # Using indirect access to ensure that the original object is updated.
            obj[i] = applyFormattingDict(el, formatting)
    elif isinstance(obj, int) or isinstance(obj, float):
        # Skip over this, as there is nothing to be done - we just keep the value.
        pass
    else:
        # This may or may not be expected, depending on the particular value.
        logger.info("NOTE: Unrecognized type {} of obj {}".format(type(obj), obj))

    return obj

