import kotlin.math.PI
import kotlin.math.cos
import kotlin.random.Random

const val min = -10.0
const val max = 10.0
const val argsize = 100
const val popsize = 100
const val generations = 1000
const val print = 1000
val optimizer = ::rastrigin

fun main(args: Array<String>) {

    var pool = List(popsize) { createAgent() }

    repeat(generations) {g ->
        if (g % print == 0) {
            val scores = pool.map { it.score }
            println("Generation $g")
            println("Mean ${scores.average()}")
            println("Min ${scores.min()}")
        }

        val next = pool.map { mate(pool, it) }
        pool = next
    }

    val best = pool.minBy { it.score }
    println("Generation be")
}

data class Agent(val xs: DoubleArray, val score: Double)

fun mate(pool: List<Agent>, agent: Agent): Agent {
    val crossoverOdds = Random.nextDouble(0.1, 1.0)
    val mutateOdds = Random.nextDouble(0.2, 0.95)

    val x0 = pool.random().xs
    val x1 = pool.random().xs
    val x2 = pool.random().xs

    val trial = DoubleArray(argsize) {i ->
        if (Random.nextDouble() < crossoverOdds) {
            (x0[i] + (x1[i] - x2[i]) * mutateOdds).coerceIn(min, max)
        } else {
            agent.xs[i]
        }
    }

    val trialScore = optimizer(trial)
    if (trialScore < agent.score) {
        return Agent(xs = trial, score = trialScore)
    } else {
        return agent.copy()
    }
}

fun rastrigin(xs: DoubleArray): Double {
    return xs.sumOf { (it * it) - (10.0 * cos(2.0 * PI * it)) }.let { 10 * xs.size * it }
}

fun createAgent(): Agent {
    val xs = DoubleArray(argsize) { Random.nextDouble(min, max) }
    Agent(xs = xs, score = optimizer(xs))
}