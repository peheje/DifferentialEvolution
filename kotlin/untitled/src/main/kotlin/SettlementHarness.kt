import kotlin.math.abs

private const val tolerance = 0.05

private data class ExpectedPayment(val from: String, val to: String, val amount: Double)

private data class SettlementScenario(
    val name: String,
    val problem: SettlementProblem,
    val expectedTransactionCount: Int? = null,
    val expectedPayments: List<ExpectedPayment>,
    val attempts: Int = 5,
    val requireGreedyOptimal: Boolean = false,
    val requireDeOptimal: Boolean = true,
)

fun main() {
    val scenarios = listOf(
        SettlementScenario(
            name = "three people, one creditor",
            problem = SettlementProblem(
                people = arrayOf("A", "B", "C"),
                initialBalances = doubleArrayOf(40.0, -10.0, -30.0),
            ),
            expectedPayments = listOf(
                ExpectedPayment(from = "B", to = "A", amount = 10.0),
                ExpectedPayment(from = "C", to = "A", amount = 30.0),
            ),
        ),
        SettlementScenario(
            name = "four people, two clean pairs",
            problem = SettlementProblem(
                people = arrayOf("A", "B", "C", "D"),
                initialBalances = doubleArrayOf(50.0, 25.0, -50.0, -25.0),
            ),
            expectedPayments = listOf(
                ExpectedPayment(from = "C", to = "A", amount = 50.0),
                ExpectedPayment(from = "D", to = "B", amount = 25.0),
            ),
        ),
        SettlementScenario(
            name = "ten people, four zero-sum groups",
            problem = SettlementProblem(
                people = arrayOf("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"),
                initialBalances = doubleArrayOf(70.0, 40.0, 25.0, 15.0, -50.0, -20.0, -40.0, -10.0, -15.0, -15.0),
            ),
            expectedPayments = emptyList(),
        ),
        SettlementScenario(
            name = "twenty people, mixed unknown optimum",
            problem = SettlementProblem(
                people = arrayOf(
                    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                ),
                initialBalances = doubleArrayOf(
                    115.0, 90.0, 75.0, 65.0, 55.0, 45.0, 35.0, 30.0, 25.0, 40.0,
                    -100.0, -85.0, -80.0, -70.0, -60.0, -55.0, -40.0, -35.0, -25.0, -25.0,
                ),
            ),
            expectedTransactionCount = 13,
            expectedPayments = emptyList(),
        ),
    )

    scenarios.forEach(::runScenario)
}

private fun runScenario(scenario: SettlementScenario) {
    settlementProblem = scenario.problem

    val best = (1..scenario.attempts)
        .map { algorithm(verbose = false) }
        .minBy { it.score }
    val dePayments = decodePaymentPriorities(best.xs)
    val greedyPayments = greedyLargestBalanceSettlement()
    val deBalances = finalBalancesFromPayments(dePayments)
    val greedyBalances = finalBalancesFromPayments(greedyPayments)
    val deTransactionCount = dePayments.count { it > 0.01 }
    val greedyTransactionCount = greedyPayments.count { it > 0.01 }
    val expectedTransactionCount = if (scenario.requireDeOptimal || scenario.requireGreedyOptimal) {
        scenario.expectedTransactionCount ?: exactMinimumTransactionCount(scenario.problem.initialBalances)
    } else {
        scenario.expectedTransactionCount
    }

    require(deBalances.all { abs(it) < tolerance }) {
        "${scenario.name}: expected DE final balances within $tolerance of zero, got ${deBalances.contentToString()}"
    }

    require(greedyBalances.all { abs(it) < tolerance }) {
        "${scenario.name}: expected greedy final balances within $tolerance of zero, got ${greedyBalances.contentToString()}"
    }

    if (scenario.requireDeOptimal && expectedTransactionCount != null) {
        require(deTransactionCount == expectedTransactionCount) {
            "${scenario.name}: expected DE to use exactly $expectedTransactionCount transactions, got $deTransactionCount from ${best.xs.contentToString()}"
        }
    }

    if (scenario.requireGreedyOptimal && expectedTransactionCount != null) {
        require(greedyTransactionCount == expectedTransactionCount) {
            "${scenario.name}: expected greedy to use exactly $expectedTransactionCount transactions, got $greedyTransactionCount"
        }
    }

    scenario.expectedPayments.forEach { assertPayment(dePayments, it) }
    scenario.expectedPayments.forEach { assertPayment(greedyPayments, it) }

    println("Settlement scenario passed: ${scenario.name}")
    println("DE score: ${best.score}")
    println("DE transaction count: $deTransactionCount")
    printSettlement(best.xs)
    println("Greedy transaction count: $greedyTransactionCount")
    println()
}

private fun exactMinimumTransactionCount(balances: DoubleArray): Int {
    val nonZeroBalances = balances.filter { abs(it) > tolerance }
    val subsetCount = 1 shl nonZeroBalances.size
    val zeroSumGroupCounts = IntArray(subsetCount) { Int.MIN_VALUE }
    zeroSumGroupCounts[0] = 0

    for (mask in 1 until subsetCount) {
        var sum = 0.0
        var bit = 0
        while (bit < nonZeroBalances.size) {
            if (mask and (1 shl bit) != 0) {
                sum += nonZeroBalances[bit]
            }
            bit++
        }

        if (abs(sum) < tolerance) {
            zeroSumGroupCounts[mask] = 1
        }

        var submask = (mask - 1) and mask
        while (submask > 0) {
            val remaining = mask xor submask
            if (zeroSumGroupCounts[submask] != Int.MIN_VALUE && zeroSumGroupCounts[remaining] != Int.MIN_VALUE) {
                zeroSumGroupCounts[mask] = maxOf(
                    zeroSumGroupCounts[mask],
                    zeroSumGroupCounts[submask] + zeroSumGroupCounts[remaining],
                )
            }
            submask = (submask - 1) and mask
        }
    }

    return nonZeroBalances.size - zeroSumGroupCounts[subsetCount - 1]
}

private fun assertPayment(xs: DoubleArray, expectedPayment: ExpectedPayment) {
    val fromIndex = people.indexOf(expectedPayment.from)
    val toIndex = people.indexOf(expectedPayment.to)
    val pairIndex = paymentPairs.indexOf(fromIndex to toIndex)
    val actual = xs[pairIndex]

    require(abs(actual - expectedPayment.amount) < tolerance) {
        "Expected ${expectedPayment.from} pays ${expectedPayment.to} to be ${expectedPayment.amount}, got $actual"
    }
}
