"""Module for generating template matching distributions."""

import Levenshtein


def _general_templates(premise, hypothesis):
    """Generate uninformative explanations based on general templates.

    Args:
        premise (str): The premise string.
        hypothesis (str): The hypothesis string.

    Returns:
        dict: The uninformative explanations from the general templates.
    """

    premise = premise.lower()
    hypothesis = hypothesis.lower()
    patterns = {
        "P"         : f"{premise}",
        "H"         : f"{hypothesis}",
        "H_P"       : f"{hypothesis} {premise}",
        "P_H"       : f"{premise} {hypothesis}",
        "S1_P_S2_H" : f"sentence 1 states {premise}. sentence 2 is stating {hypothesis}",
        "S2_H_S1_P" : f"sentence 2 states {hypothesis}. sentence 1 is stating {premise}",
        "H_EXISTS"  : f"there is {hypothesis}",
        "P_EXISTS"  : f"there is {premise}"
    }
    return patterns


def _contradiction_templates(premise, hypothesis):
    """Generate uninformative explanations based on contradiction templates.

    Args:
        premise (str): The premise string.
        hypothesis (str): The hypothesis string.

    Returns:
        dict: The uninformative explanations from the contradiction templates.
    """

    premise = premise.lower()
    hypothesis = hypothesis.lower()
    patterns = {
        "S1_P_WHILE_S2_H"           : f"in sentence 1 {premise} while in sentence 2 {hypothesis}",
        "CAN_EITHER_BE_P_OR_H"      : f"it can either be {premise} or {hypothesis}",
        "NOT_H_IF_P"                : f"it cannot be {hypothesis} if {premise}",
        "EITHER_P_OR_H"             : f"either {premise} or {hypothesis}",
        "EITHER_H_OR_P"             : f"either {hypothesis} or {premise}",
        "P_AND_OTHER_H"             : f"{premise} and other {hypothesis}",
        "H_AND_OTHER_P"             : f"{hypothesis} and other {premise}",
        "H_AFTER_P"                 : f"{hypothesis} after {premise}",
        "P_NOT_SAME_AS_H"           : f"{premise} is not the same as {hypothesis}",
        "H_NOT_SAME_AS_P"           : f"{hypothesis} is not the same as {premise}",
        "P_CONTRADICTS_H"           : f"{premise} is contradictory to {hypothesis}",
        "H_CONTRADICTS_P"           : f"{hypothesis} is contradictory to {premise}",
        "P_CONTRADICTS_H_ALT"       : f"{premise} contradicts {hypothesis}",
        "H_CONTRADICTS_P_ALT"       : f"{hypothesis} contradicts {premise}",
        "P_CANNOT_ALSO_BE_H"        : f"{premise} cannot also be {hypothesis}",
        "H_CANNOT_ALSO_BE_P"        : f"{hypothesis} cannot also be {premise}",
        "EITHER_P_OR_H_NOT_BOTH"    : f"either {premise} or {hypothesis} not both at the same time",
        "P_OR_H_NOT_BOTH"           : f"{premise} or {hypothesis} not both at the same time"
    }
    return patterns


def _neutral_templates(premise, hypothesis):
    """Generate uninformative explanations based on neutral templates.

    Args:
        premise (str): The premise string.
        hypothesis (str): The hypothesis string.

    Returns:
        dict: The uninformative explanations from the neutral templates.
    """

    premise = premise.lower()
    hypothesis = hypothesis.lower()
    patterns = {
        "JUST_BECAUSE_P_NOT_MEAN_H" : f"just because {premise} doesn't mean {hypothesis}",
        "CANNOT_INFER_H"            : f"cannot infer the {hypothesis}",
        "CANNOT_ASSUME_H"           : f"one cannot assume {hypothesis}",
        "CANNOT_INFER_THAT_H"       : f"one cannot infer that {hypothesis}",
        "CANNOT_ASSUME_H_ALT"       : f"cannot assume {hypothesis}",
        "P_NOT_MEAN_H"              : f"{premise} does not mean {hypothesis}",
        "DONT_KNOW_H"               : f"we don't know that {hypothesis}",
        "FACT_P_NOT_MEAN_H"         : f"the fact that {premise} doesn't mean {hypothesis}",
        "FACT_P_NOT_IMPLY_H"        : f"the fact that {premise} does not imply {hypothesis}",
        "FACT_P_NOT_ALWAYS_MEAN_H"  : f"the fact that {premise} does not always mean {hypothesis}",
        "FACT_P_NOT_ALWAYS_IMPLY_H" : f"the fact that {premise} doesn't always imply {hypothesis}"
    }
    return patterns


def _entailment_templates(premise, hypothesis):
    """Generate uninformative explanations based on entailment templates.

    Args:
        premise (str): The premise string.
        hypothesis (str): The hypothesis string.

    Returns:
        dict: The uninformative explanations from the entailment templates.
    """

    premise = premise.lower()
    hypothesis = hypothesis.lower()
    patterns = {
        "P_IMPLIES_H"           : f"{premise} implies {hypothesis}",
        "IF_P_THEN_H"           : f"if {premise} then {hypothesis}",
        "P_WOULD_IMPLY_H"       : f"{premise} would imply {hypothesis}",
        "H_REPHRASING_OF_P"     : f"{hypothesis} is a rephrasing of {premise}",
        "P_REPHRASING_OF_H"     : f"{premise} is a rephrasing of {hypothesis}",
        "BOTH_SENTENCES_H"      : f"in both sentences {hypothesis}",
        "P_WOULD_BE_H"          : f"{premise} would be {hypothesis}",
        "P_CAN_BE_SAID_AS_H"    : f"{premise} can also be said as {hypothesis}",
        "H_CAN_BE_SAID_AS_P"    : f"{hypothesis} can also be said as {premise}",
        "H_LESS_SPECIFIC_P"     : f"{hypothesis} is a less specific rephrasing of {premise}",
        "CLARIFIES_H"           : f"this clarifies that {hypothesis}",
        "IF_P_IT_MEANS_H"       : f"if {premise} it means {hypothesis}",
        "H_IN_BOTH_SENTENCES"   : f"{hypothesis} in both sentences",
        "H_IN_BOTH"             : f"{hypothesis} in both",
        "H_SAME_AS_P"           : f"{hypothesis} is same as {premise}",
        "P_SAME_AS_H"           : f"{premise} is same as {hypothesis}",
        "P_SYNONYM_OF_H"        : f"{premise} is a synonym of {hypothesis}",
        "H_SYNONYM_OF_P"        : f"{hypothesis} is a synonym of {premise}"
    }
    return patterns


def template_matching(premises, hypotheses, labels, explanations, cutoff=13):
    """Match explanations against all templates.
     
    Every premise and hypothesis pair that corresponds to an explanation
    is substituted into all general, contradiction, neutral and entailment templates,
    which are then used to determine whether the explanation is considered
    uninformative or not, and counted towards the template matching distribution.

    Args:
        premises (List[str]): A list of premise strings.
        hypotheses (List[str]): A list of hypothesis strings.
        labels (List[str]): A list of label strings.
        explanations (List[str]): A list of explanation strings.
        cutoff (int, optional): Levenshtein distance. Defaults to 13.

    Returns:
        dict: A template counter dict, distribution dict and its respective indices dict.
    """

    counter = {
        'general': 0,
        'entailment': 0,
        'neutral': 0,
        'contradiction': 0
    }
    
    # initialize template bins for counting specific template matches
    bins = {
        'general': { i: 0 for i in _general_templates('', '').keys() },
        'contradiction': { i: 0 for i in _contradiction_templates('', '').keys()},
        'neutral': { i: 0 for i in _neutral_templates('', '').keys() },
        'entailment': { i: 0 for i in _entailment_templates('', '').keys() }
    }
    
    # index list for each template match
    indices = {
        'general': { i: [] for i in _general_templates('', '').keys() },
        'contradiction': { i: [] for i in _contradiction_templates('', '').keys()},
        'neutral': { i: [] for i in _neutral_templates('', '').keys() },
        'entailment': { i: [] for i in _entailment_templates('', '').keys() }
    }
    
    for idx, (premise, hypothesis, label, sentence) in enumerate(zip(premises, hypotheses, labels, explanations)):
        # first check general
        is_counted = False
        for loc, pattern in _general_templates(premise, hypothesis).items():
            difference = Levenshtein.distance(sentence.lower(), pattern)
            if difference < cutoff:
                bins['general'][loc] += 1
                indices['general'][loc].append(idx)
                if not is_counted:
                    counter['general'] += 1
                    is_counted = True
        
        # now check specifics
        is_counted = False
        if label == 'contradiction':
            for loc, pattern in _contradiction_templates(premise, hypothesis).items():
                difference = Levenshtein.distance(sentence.lower(), pattern)
                if difference < cutoff:
                    bins['contradiction'][loc] += 1
                    indices['contradiction'][loc].append(idx)
                    if not is_counted:
                        counter['contradiction'] += 1
                        is_counted = True
        elif label == 'neutral':
            for loc, pattern in _neutral_templates(premise, hypothesis).items():
                difference = Levenshtein.distance(sentence.lower(), pattern)
                if difference < cutoff:
                    bins['neutral'][loc] += 1
                    indices['neutral'][loc].append(idx)
                    if not is_counted:
                        counter['neutral'] += 1
                        is_counted = True
        elif label == 'entailment':
            for loc, pattern in _entailment_templates(premise, hypothesis).items():
                difference = Levenshtein.distance(sentence.lower(), pattern)
                if difference < cutoff:
                    bins['entailment'][loc] += 1
                    indices['entailment'][loc].append(idx)
                    if not is_counted:
                        counter['entailment'] += 1
                        is_counted = True
    
    return {'count': counter, 'distribution': bins, 'indices': indices}
