import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.random.Random
import kotlin.time.measureTime

data class SettlementProblem(
    val people: Array<String>,
    val initialBalances: DoubleArray,
) {
    val paymentPairs: Array<Pair<Int, Int>> = people.indices
        .filter { initialBalances[it] < 0.0 }
        .flatMap { from -> people.indices.filter { initialBalances[it] > 0.0 }.map { to -> from to to } }
        .toTypedArray()

    val maxPayment: Double = initialBalances.sumOf { if (it > 0.0) it else 0.0 }
}

val defaultSettlementProblem = SettlementProblem(
    people = arrayOf("A", "B", "C"),
    initialBalances = doubleArrayOf(40.0, -10.0, -30.0),
)
var settlementProblem = defaultSettlementProblem
val people get() = settlementProblem.people
val initialBalances get() = settlementProblem.initialBalances
val paymentPairs get() = settlementProblem.paymentPairs

const val min = 0.0
val max get() = settlementProblem.maxPayment
val argsize get() = paymentPairs.size
const val popsize = 200
const val generations = 20000
const val print = 2000
val optimizer = ::settleTrip

const val minimumTransactionAmount = 0.01
const val settledEnoughBalanceTolerance = 0.05
const val unsettledBalancePenalty = 1_000_000.0
const val realTransactionPenalty = 1_000.0
const val softTransactionPenalty = 10.0
const val softTransactionScale = 0.1
const val unsettledSoftTransactionPenalty = 100.0
const val moneyMovedPenalty = 0.001

fun main() {
    // source "$HOME/.sdkman/bin/sdkman-init.sh"
    // kotlinc Main.kt -include-runtime -d run.jar && java -jar run.jar && rm run.jar
    measureTime { algorithm() }.let { println("Elapsed: ${it.inWholeMilliseconds}ms") }
}

fun algorithm(verbose: Boolean = true): Agent {
    val pool = List(popsize) { createAgent() }
    val trials = Array(popsize) { DoubleArray(argsize) }
    val indices = (0 until popsize).toList()
    repeat(generations) { g ->
        if (verbose && g % print == 0) {
            val scores = pool.map { it.score }
            println("Generation $g")
            println("Mean ${scores.average()}")
            println("Min ${scores.min()}")
        }

        val crossover = Random.nextDouble(0.1, 1.0)
        val mutate = Random.nextDouble(0.2, 0.95)

        indices.parallelStream().forEach { i -> mate(pool, pool[i], trials[i], crossover, mutate) }
    }

    val best = pool.minBy { it.score }
    if (verbose) {
        println("Generation best ${best.xs.contentToString()}")
        println("score: ${best.score}")
        printSettlement(best.xs)
    }

    return best
}

class Agent(val xs: DoubleArray, var score: Double)

fun mate(
        pool: List<Agent>,
        agent: Agent,
        trial: DoubleArray,
        crossoverOdds: Double,
        mutateOdds: Double
) {

    val x0 = pool.random().xs
    val x1 = pool.random().xs
    val x2 = pool.random().xs

    for (i in 0 until argsize) {
        if (Random.nextDouble() < crossoverOdds) {
            trial[i] = (x0[i] + (x1[i] - x2[i]) * mutateOdds).coerceIn(min, max)
        } else {
            trial[i] = agent.xs[i]
        }
    }

    val trialScore = optimizer(trial)
    if (trialScore < agent.score) {
        System.arraycopy(trial, 0, agent.xs, 0, argsize)
        agent.score = trialScore
    }
}

fun f1(xs: DoubleArray): Double = xs.sumOf { it * it }

fun f2(xs: DoubleArray): Double {
    var s = 0.0
    var p = 1.0

    for (i in xs.indices) {
        s += abs(xs[i])
        p *= xs[i]
    }

    return abs(s) + abs(p)
}

fun findSqrt(xs: DoubleArray): Double {
    val c = xs[0]
    val t = c * c - 54
    return abs(t)
}

fun rastrigin(xs: DoubleArray): Double {
    val a = 10.0
    val sum = xs.sumOf { (it * it) - (a * cos(2.0 * PI * it)) }
    return a * xs.size + sum
}

fun settleTrip(xs: DoubleArray): Double {
    val payments = decodePaymentPriorities(xs)
    val totalUnsettledBalance = finalBalancesFromPayments(payments).sumOf { abs(it) }
    val numberOfRealTransactions = payments.count { it > minimumTransactionAmount }
    val smoothApproximationOfTransactionCount = payments.sumOf { it / (it + softTransactionScale) }
    val totalMoneyMoved = payments.sum()

    return if (totalUnsettledBalance > settledEnoughBalanceTolerance) {
        val balanceIsStillTheOnlyThingThatMatters = totalUnsettledBalance * unsettledBalancePenalty
        val gentleHintTowardFewerTransactions = smoothApproximationOfTransactionCount * unsettledSoftTransactionPenalty

        balanceIsStillTheOnlyThingThatMatters + gentleHintTowardFewerTransactions
    } else {
        val exactTransactionCountDominatesAfterSettlement = numberOfRealTransactions * realTransactionPenalty
        val smoothTieBreakerPrefersFewerSplitPayments = smoothApproximationOfTransactionCount * softTransactionPenalty
        val tinyTieBreakerPrefersMovingLessMoney = totalMoneyMoved * moneyMovedPenalty

        exactTransactionCountDominatesAfterSettlement +
                smoothTieBreakerPrefersFewerSplitPayments +
                tinyTieBreakerPrefersMovingLessMoney +
                totalUnsettledBalance
    }
}

fun finalBalances(xs: DoubleArray): DoubleArray {
    return finalBalancesFromPayments(decodePaymentPriorities(xs))
}

fun finalBalancesFromPayments(payments: DoubleArray): DoubleArray {
    val balances = initialBalances.copyOf()

    for (i in paymentPairs.indices) {
        val amount = payments[i]
        val (from, to) = paymentPairs[i]
        balances[from] += amount
        balances[to] -= amount
    }

    return balances
}

fun printSettlement(xs: DoubleArray) {
    val payments = decodePaymentPriorities(xs)
    println("Payments:")
    for (i in paymentPairs.indices) {
        val amount = payments[i]
        if (amount > 0.01) {
            val (from, to) = paymentPairs[i]
            println("${people[from]} pays ${people[to]}: ${String.format("%.2f", amount)}")
        }
    }

    println("Final balances: ${finalBalances(xs).contentToString()}")
}

fun createAgent(): Agent {
    val xs = DoubleArray(argsize) { Random.nextDouble(min, max) }
    return Agent(xs = xs, score = optimizer(xs))
}

fun decodePaymentPriorities(xs: DoubleArray): DoubleArray {
    val balances = initialBalances.copyOf()
    val payments = DoubleArray(argsize)

    for (paymentIndex in xs.indices.sortedByDescending { xs[it] }) {
        val (from, to) = paymentPairs[paymentIndex]
        val amount = minOf(-balances[from], balances[to])
        if (amount > minimumTransactionAmount) {
            payments[paymentIndex] = amount
            balances[from] += amount
            balances[to] -= amount
        }
    }

    return payments
}

fun greedyLargestBalanceSettlement(): DoubleArray {
    val remainingBalances = initialBalances.copyOf()
    val payments = DoubleArray(argsize)

    while (true) {
        val debtor = people.indices
            .filter { remainingBalances[it] < -minimumTransactionAmount }
            .maxByOrNull { -remainingBalances[it] }
        val creditor = people.indices
            .filter { remainingBalances[it] > minimumTransactionAmount }
            .maxByOrNull { remainingBalances[it] }

        if (debtor == null || creditor == null) break

        val amount = minOf(-remainingBalances[debtor], remainingBalances[creditor])
        val pairIndex = paymentPairs.indexOf(debtor to creditor)
        if (pairIndex < 0) error("No payment pair for ${people[debtor]} -> ${people[creditor]}")

        payments[pairIndex] += amount
        remainingBalances[debtor] += amount
        remainingBalances[creditor] -= amount
    }

    return payments
}
