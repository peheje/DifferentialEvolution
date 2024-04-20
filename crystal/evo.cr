# crystal build evo.cr --release

def calculate_score(xs : Array(Float64))
  # f(0) = 0, n params
  xs.sum { |x| x**2 }
end

start_dt = Time.utc

print_each = 1000
params = 300
bounds = -10.0..10.0
generations = 5000
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
    #best = scores.min
    #puts "BEST #{best.xs}"
  end
end

end_dt = Time.utc
puts end_dt - start_dt
