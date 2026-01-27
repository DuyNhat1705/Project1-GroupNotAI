def get_problem(name, **kwargs):
    """
    problem string name
    """
    #add problems list later: /*...*/ = "..."
    problems = {
        # Discrete

        # Continuous

    }

    if name not in problems:
        raise ValueError(f"Problem '{name}' not found. Available: {list(problems.keys())}")

    return problems[name](**kwargs)


def get_algorithm(name, **kwargs):
    """
    algorithm string name.
    """

    # add problems list later: /*...*/ = "..."
    algos = {
        # Classical

        # Nature-Inspired

    }

    if name not in algos:
        raise ValueError(f"Algorithm '{name}' not found. Available: {list(algos.keys())}")

    return algos[name](params=kwargs)