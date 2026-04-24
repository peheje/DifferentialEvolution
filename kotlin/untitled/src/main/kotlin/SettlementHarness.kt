import kotlin.math.abs

private const val tolerance = 0.05

private data class ExpectedPayment(val from: String, val to: String, val amount: Double)

private data class SettlementScenario(
    val name: String,
    val problem: SettlementProblem,
    val expectedTransactionCount: Int,
    val expectedPayments: List<ExpectedPayment>,
    val attempts: Int = 5,
)

fun main() {
    val scenarios = listOf(
        SettlementScenario(
            name = "three people, one creditor",
            problem = SettlementProblem(
                people = arrayOf("A", "B", "C"),
                initialBalances = doubleArrayOf(40.0, -10.0, -30.0),
            ),
            expectedTransactionCount = 2,
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
            expectedTransactionCount = 2,
            expectedPayments = listOf(
                ExpectedPayment(from = "C", to = "A", amount = 50.0),
                ExpectedPayment(from = "D", to = "B", amount = 25.0),
            ),
        ),
    )

    scenarios.forEach(::runScenario)
}

private fun runScenario(scenario: SettlementScenario) {
    settlementProblem = scenario.problem

    val best = (1..scenario.attempts)
        .map { algorithm(verbose = false) }
        .minBy { it.score }
    val balances = finalBalances(best.xs)
    val transactionCount = best.xs.count { it > 0.01 }

    require(balances.all { abs(it) < tolerance }) {
        "${scenario.name}: expected all final balances within $tolerance of zero, got ${balances.contentToString()}"
    }

    require(transactionCount == scenario.expectedTransactionCount) {
        "${scenario.name}: expected exactly ${scenario.expectedTransactionCount} transactions, got $transactionCount from ${best.xs.contentToString()}"
    }

    scenario.expectedPayments.forEach { assertPayment(best.xs, it) }

    println("Settlement scenario passed: ${scenario.name}")
    println("Score: ${best.score}")
    printSettlement(best.xs)
    println()
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
