import kotlin.math.PI
import kotlin.math.cos
import kotlin.random.Random
import kotlin.time.measureTime

const val min = -10.0
const val max = 10.0
const val argsize = 1000
const val popsize = 200
const val generations = 10_000
const val print = 1000
val optimizer = ::f1

fun main() {

    measureTime {
        algorithm()
    }.let { println("Elapsed: ${it.inWholeMilliseconds}ms") }

}

private fun algorithm() {
    var pool = List(popsize) { createAgent() }

    repeat(generations) { g ->
        if (g % print == 0) {
            val scores = pool.map { it.score }
            println("Generation $g")
            println("Mean ${scores.average()}")
            println("Min ${scores.min()}")
        }

        val crossover = Random.nextDouble(0.1, 1.0)
        val mutate = Random.nextDouble(0.2, 0.95)
        val next = pool.parallelStream().map { mate(pool, it, crossover, mutate) }
        pool = next.toList()
    }

    val best = pool.minBy { it.score }
    println("Generation best ${best.xs.contentToString()}")
    println("score: ${best.score}")
}

data class Agent(val xs: DoubleArray, val score: Double)

fun mate(pool: List<Agent>, agent: Agent, crossoverOdds: Double, mutateOdds: Double): Agent {

    val x0 = pool.random().xs
    val x1 = pool.random().xs
    val x2 = pool.random().xs

    val trial = DoubleArray(argsize) { i ->
        if (Random.nextDouble() < crossoverOdds) {
            (x0[i] + (x1[i] - x2[i]) * mutateOdds).coerceIn(min, max)
        } else {
            agent.xs[i]
        }
    }

    val trialScore = optimizer(trial)
    return if (trialScore < agent.score) {
        Agent(xs = trial, score = trialScore)
    } else {
        agent
    }
}

fun f1(xs: DoubleArray) = xs.sumOf { it * it }

fun rastrigin(xs: DoubleArray): Double {
    return xs.sumOf { (it * it) - (10.0 * cos(2.0 * PI * it)) }.let { 10.0 * xs.size * it }
}

fun createAgent(): Agent {
    val xs = DoubleArray(argsize) { Random.nextDouble(min, max) }
    return Agent(xs = xs, score = optimizer(xs))
}