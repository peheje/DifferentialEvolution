# crystal build evo.cr --release

def calculate_score(xs : Array(Float64))
  # f(0) = 0, n params
  xs.sum { |x| x**2 }
end

start_dt = Time.utc

print_each = 20000
params = 1000
bounds = -10.0..10.0
generations = 20000
pop_size = 200
mutate_range = 0.2..0.95
crossover_range = 0.1..1.0

trial = Array.new(params, 0.0)
pop = Array.new(pop_size) { Array.new(params) { rand(bounds) } }
scores = pop.map { |xs| calculate_score(xs) }

generations.times do |g|
  crossover, mutate = rand(crossover_range), rand(mutate_range)

  pop_size.times do |i|
    x0, x1, x2, xt = pop.sample, pop.sample, pop.sample, pop[i]
    params.times { |j| trial[j] = rand < crossover ? (x0[j] + (x1[j] - x2[j]) * mutate).clamp(bounds) : xt[j] }
    trial_score = calculate_score(trial)
    pop[i], scores[i] = trial.dup, trial_score if trial_score < scores[i]
  end

  if g.remainder(print_each) == 0 || g == generations-1
    mean = scores.sum / pop_size
    puts "GEN #{g}"
    puts "MEAN #{mean}"
    puts "MIN #{scores.min}"
  end
end

best_score = scores.min
best_index = scores.index(best_score).not_nil!
best = pop[best_index]

end_dt = Time.utc
puts "score #{best_score}"
puts "execution time #{((end_dt - start_dt).total_milliseconds).to_i} ms"
