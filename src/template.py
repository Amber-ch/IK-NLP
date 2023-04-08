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
        0: f"{premise}",
        1: f"{hypothesis}",
        2: f"{hypothesis} {premise}",
        3: f"{premise} {hypothesis}",
        4: f"sentence 1 states {premise}. sentence 2 is stating {hypothesis}",
        5: f"sentence 2 states {hypothesis}. sentence 1 is stating {premise}",
        6: f"there is {hypothesis}",
        7: f"there is {premise}"
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
        0: f"in sentence 1 {premise} while in sentence 2 {hypothesis}",
        1: f"it can either be {premise} or {hypothesis}",
        2: f"it cannot be {hypothesis} if {premise}",
        3: f"either {premise} or {hypothesis}",
        4: f"either {hypothesis} or {premise}",
        5: f"{premise} and other {hypothesis}",
        6: f"{hypothesis} and other {premise}",
        7: f"{hypothesis} after {premise}",
        8: f"{premise} is not the same as {hypothesis}",
        9: f"{hypothesis} is not the same as {premise}",
        10: f"{premise} is contradictory to {hypothesis}",
        11: f"{hypothesis} is contradictory to {premise}",
        12: f"{premise} contradicts {hypothesis}",
        13: f"{hypothesis} contradicts {premise}",
        14: f"{premise} cannot also be {hypothesis}",
        15: f"{hypothesis} cannot also be {premise}",
        16: f"either {premise} or {hypothesis} not both at the same time",
        17: f"{premise} or {hypothesis} not both at the same time"
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
        0: f"just because {premise} doesn't mean {hypothesis}",
        1: f"cannot infer the {hypothesis}",
        2: f"one cannot assume {hypothesis}",
        3: f"one cannot infer that {hypothesis}",
        4: f"cannot assume {hypothesis}",
        5: f"{premise} does not mean {hypothesis}",
        6: f"we don't know that {hypothesis}",
        7: f"the fact that {premise} doesn't mean {hypothesis}",
        8: f"the fact that {premise} does not imply {hypothesis}",
        9: f"the fact that {premise} does not always mean {hypothesis}",
        10: f"the fact that {premise} doesn't always imply {hypothesis}"
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
        0: f"{premise} implies {hypothesis}",
        1: f"if {premise} then {hypothesis}",
        2: f"{premise} would imply {hypothesis}",
        3: f"{hypothesis} is a rephrasing of {premise}",
        4: f"{premise} is a rephrasing of {hypothesis}",
        5: f"in both sentences {hypothesis}",
        6: f"{premise} would be {hypothesis}",
        7: f"{premise} can also be said as {hypothesis}",
        8: f"{hypothesis} can also be said as {premise}",
        9: f"{hypothesis} is a less specific rephrasing of {premise}",
        10: f"this clarifies that {hypothesis}",
        11: f"if {premise} it means {hypothesis}",
        12: f"{hypothesis} in both sentences",
        13: f"{hypothesis} in both",
        14: f"{hypothesis} is same as {premise}",
        15: f"{premise} is same as {hypothesis}",
        16: f"{premise} is a synonym of {hypothesis}",
        17: f"{hypothesis} is a synonym of {premise}"
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
            similarity = Levenshtein.distance(sentence.lower(), pattern)
            if similarity < cutoff:
                bins['general'][loc] += 1
                indices['general'][loc].append(idx)
                if not is_counted:
                    counter['general'] += 1
                    is_counted = True
        
        # now check specifics
        is_counted = False
        if label == 'contradiction':
            for loc, pattern in _contradiction_templates(premise, hypothesis).items():
                similarity = Levenshtein.distance(sentence.lower(), pattern)
                if similarity < cutoff:
                    bins['contradiction'][loc] += 1
                    indices['contradiction'][loc].append(idx)
                    if not is_counted:
                        counter['contradiction'] += 1
                        is_counted = True
        elif label == 'neutral':
            for loc, pattern in _neutral_templates(premise, hypothesis).items():
                similarity = Levenshtein.distance(sentence.lower(), pattern)
                if similarity < cutoff:
                    bins['neutral'][loc] += 1
                    indices['neutral'][loc].append(idx)
                    if not is_counted:
                        counter['neutral'] += 1
                        is_counted = True
        elif label == 'entailment':
            for loc, pattern in _entailment_templates(premise, hypothesis).items():
                similarity = Levenshtein.distance(sentence.lower(), pattern)
                if similarity < cutoff:
                    bins['entailment'][loc] += 1
                    indices['entailment'][loc].append(idx)
                    if not is_counted:
                        counter['entailment'] += 1
                        is_counted = True
    
    return {'count': counter, 'distribution': bins, 'indices': indices}
