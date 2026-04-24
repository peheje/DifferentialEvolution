import kotlin.math.abs

private const val tolerance = 0.05

fun main() {
    val best = algorithm(verbose = false)
    val balances = finalBalances(best.xs)
    val transactionCount = best.xs.count { it > 0.01 }

    require(balances.all { abs(it) < tolerance }) {
        "Expected all final balances within $tolerance of zero, got ${balances.contentToString()}"
    }

    require(transactionCount == 2) {
        "Expected exactly 2 transactions, got $transactionCount from ${best.xs.contentToString()}"
    }

    assertPayment(best.xs, from = "B", to = "A", expected = 10.0)
    assertPayment(best.xs, from = "C", to = "A", expected = 30.0)

    println("Settlement harness passed")
    println("Score: ${best.score}")
    printSettlement(best.xs)
}

private fun assertPayment(xs: DoubleArray, from: String, to: String, expected: Double) {
    val fromIndex = people.indexOf(from)
    val toIndex = people.indexOf(to)
    val pairIndex = paymentPairs.indexOf(fromIndex to toIndex)
    val actual = xs[pairIndex]

    require(abs(actual - expected) < tolerance) {
        "Expected $from pays $to to be $expected, got $actual"
    }
}
