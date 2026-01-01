import kotlin.time.measureTime

fun main() {
    val runs = 20
    val times = LongArray(runs)

    println("Starting $runs benchmark runs...")

    for (i in 0 until runs) {
        print("Run ${i + 1}/$runs... ")
        val duration = measureTime { algorithm() }
        times[i] = duration.inWholeMilliseconds
        println("Done in ${times[i]}ms")
    }

    val average = times.average()
    val min = times.minOrNull() ?: 0
    val max = times.maxOrNull() ?: 0

    println("\nBenchmark Results ($runs runs):")
    println("Average: ${String.format("%.2f", average)} ms")
    println("Min: $min ms")
    println("Max: $max ms")
}
