import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve_settlement(balances):
    people = list(balances)
    creditors = [person for person in people if balances[person] > 0]
    debtors = [person for person in people if balances[person] < 0]
    edges = [(debtor, creditor) for debtor in debtors for creditor in creditors]

    amount_offset = 0
    used_offset = len(edges)
    variable_count = len(edges) * 2

    objective = np.zeros(variable_count)
    objective[used_offset:] = 1.0

    lower_bounds = np.zeros(variable_count)
    upper_bounds = np.zeros(variable_count)
    integrality = np.zeros(variable_count)

    for i, (debtor, creditor) in enumerate(edges):
        max_amount = min(-balances[debtor], balances[creditor])
        upper_bounds[amount_offset + i] = max_amount
        upper_bounds[used_offset + i] = 1.0
        integrality[used_offset + i] = 1

    rows = []
    lower_constraints = []
    upper_constraints = []

    for debtor in debtors:
        row = np.zeros(variable_count)
        for i, (edge_debtor, _) in enumerate(edges):
            if edge_debtor == debtor:
                row[amount_offset + i] = 1.0
        rows.append(row)
        lower_constraints.append(-balances[debtor])
        upper_constraints.append(-balances[debtor])

    for creditor in creditors:
        row = np.zeros(variable_count)
        for i, (_, edge_creditor) in enumerate(edges):
            if edge_creditor == creditor:
                row[amount_offset + i] = 1.0
        rows.append(row)
        lower_constraints.append(balances[creditor])
        upper_constraints.append(balances[creditor])

    for i, (debtor, creditor) in enumerate(edges):
        max_amount = min(-balances[debtor], balances[creditor])
        row = np.zeros(variable_count)
        row[amount_offset + i] = 1.0
        row[used_offset + i] = -max_amount
        rows.append(row)
        lower_constraints.append(-np.inf)
        upper_constraints.append(0.0)

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.array(lower_constraints),
            np.array(upper_constraints),
        ),
        options={"presolve": True},
    )

    if not result.success:
        raise RuntimeError(result.message)

    payments = []
    for i, (debtor, creditor) in enumerate(edges):
        amount = result.x[amount_offset + i]
        if amount > 0.01:
            payments.append((debtor, creditor, amount))

    return result, payments


if __name__ == "__main__":
    mixed_twenty_person_balances = {
        "A": 115.0,
        "B": 90.0,
        "C": 75.0,
        "D": 65.0,
        "E": 55.0,
        "F": 45.0,
        "G": 35.0,
        "H": 30.0,
        "I": 25.0,
        "J": 40.0,
        "K": -100.0,
        "L": -85.0,
        "M": -80.0,
        "N": -70.0,
        "O": -60.0,
        "P": -55.0,
        "Q": -40.0,
        "R": -35.0,
        "S": -25.0,
        "T": -25.0,
    }

    result, payments = solve_settlement(mixed_twenty_person_balances)

    print("status:", result.message)
    print("optimal transaction count:", int(round(result.fun)))
    print("mip gap:", result.mip_gap)
    for debtor, creditor, amount in payments:
        print(f"{debtor} pays {creditor}: {amount:.2f}")
