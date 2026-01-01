import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.random.Random
import kotlin.time.measureTime

const val min = -10.0
const val max = 10.0
const val argsize = 1000
const val popsize = 200
const val generations = 20000
const val print = 20000
val optimizer = ::f1

fun main() {
    // kotlinc Main.kt -include-runtime -d run.jar && java -jar run.jar && rm run.jar
    measureTime { algorithm() }.let { println("Elapsed: ${it.inWholeMilliseconds}ms") }
}

fun algorithm() {
    val pool = List(popsize) { createAgent() }
    val trials = Array(popsize) { DoubleArray(argsize) }
    val indices = (0 until popsize).toList()
    repeat(generations) { g ->
        if (g % print == 0) {
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
    println("Generation best ${best.xs.contentToString()}")
    println("score: ${best.score}")
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

fun createAgent(): Agent {
    val xs = DoubleArray(argsize) { Random.nextDouble(min, max) }
    return Agent(xs = xs, score = optimizer(xs))
}
