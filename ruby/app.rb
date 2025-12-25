# app.rb

# --- Problem Definitions ---
module Problems
  module_function

  def constraints1(c)
    x = c[0]
    y = c[1]
    f1_val = (2 * x + 3 * y) * (x - y) - 2
    f2_val = 3 * x + y - 5
    (f1_val**2) + (f2_val**2)
  end

  def booth(x)
    # f(1.0,3.0)=0
    t1 = (x[0] + 2 * x[1] - 7.0)**2
    t2 = (2 * x[0] + x[1] - 5.0)**2
    t1 + t2
  end

  def beale(x)
    # f(3,0.5)=0
    term1 = (1.500 - x[0] + x[0] * x[1])**2
    term2 = (2.250 - x[0] + x[0] * x[1]**2)**2
    term3 = (2.625 - x[0] + x[0] * x[1]**3)**2
    term1 + term2 + term3
  end

  def matyas(x)
    # f(0,0)=0
    t1 = 0.26 * (x[0]**2 + x[1]**2)
    t2 = 0.48 * x[0] * x[1]
    t1 - t2
  end

  def f1(x)
    # f(0) = 0
    s = 0.0
    i = 0
    while i < x.size
      val = x[i]
      s += val * val
      i += 1
    end
    s
  end

  def f2(x)
    s = 0.0
    p = 1.0
    i = 0
    while i < x.size
      val = x[i]
      s += val.abs
      p *= val
      i += 1
    end
    s.abs + p.abs
  end

  def f3(x)
    result = 0.0
    i = 0
    while i < x.size
      ss = 0.0
      j = 0
      while j < i + 1
        ss += x[j] 
        j += 1
      end
      result += ss * ss
      i += 1
    end
    result
  end

  def lol1(x)
    health = x[0]
    armor = x[1]
    effective_hp = (health * (100.0 + armor)) / 100.0
    
    if (health * 2.5 + armor * 18) > 3600
      return 1.0
    end
    1.0 / effective_hp
  end

  def lol2(x)
    health = x[0]
    armor = x[1]
    current_health = health + 1080
    current_armor = armor + 10
    effective_hp = (current_health * (100.0 + current_armor)) / 100.0

    if (health * 2.5 + armor * 18) > 720
      return 1.0
    end
    1.0 / effective_hp
  end

  def lol3(x)
    health = x[0]
    armor = x[1]
    current_health = health + 2000
    current_armor = armor + 50
    effective_hp = (current_health * (100.0 + current_armor)) / 100.0

    if (health * 2.5 + armor * 18) > 1800
      return 1.0
    end
    1.0 / effective_hp
  end

  def find_sqrt(x)
    c = x[0]
    t = c * c - 54.0
    t.abs
  end

  def horses_and_jockeys(x)
    horses = x[0]
    jockeys = x[1]
    legs = horses * 4 + jockeys * 2
    heads = horses + jockeys
    (36 - heads).abs + (100 - legs).abs
  end
end

# --- Configuration ---
LOG_CSV = false
PRINT_INTERVAL = 1000
OPTIMIZER = :f1  # Change this symbol to switch problems (e.g., :lol1)
PARAMS = 100
BOUNDS = -10.0..10.0
GENERATIONS = 10_000
POPSIZE = 100
MUTATE_RANGE = 0.2..0.95
CROSSOVER_RANGE = 0.1..1.0

# --- Setup Logging ---
if LOG_CSV
  file = File.open("ruby_write.txt", "w")
  file.puts("step,mean")
end

# --- Timer Start ---
# Using monotonic time for accurate benchmarking
start_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)

# --- Initialization ---
# Creating population
# We use Array.new with a block to ensure distinct arrays
population = Array.new(POPSIZE) { Array.new(PARAMS) { rand(BOUNDS) } }

# Calculating initial scores
scores = population.map { |agent| Problems.send(OPTIMIZER, agent) }

# Pre-allocate the trial array to avoid creating 200 million arrays
trial = Array.new(PARAMS)

puts "Starting optimization using #{OPTIMIZER}..."

# --- Main Optimization Loop ---
g = 0
while g < GENERATIONS
  crossover = rand(CROSSOVER_RANGE)
  mutate = rand(MUTATE_RANGE)
  
  # Caching bounds for speed in the loop
  b_min = BOUNDS.min
  b_max = BOUNDS.max

  i = 0
  while i < POPSIZE
    # 1. Sampling
    # Get 3 distinct parents. .sample(3) is optimized in C.
    candidates = population.sample(3)
    x0 = candidates[0]
    x1 = candidates[1]
    x2 = candidates[2]
    xt = population[i]

    # 2. Mutation & Crossover
    j = 0
    while j < PARAMS
      if rand < crossover
        # Differential Evolution Formula
        val = x0[j] + (x1[j] - x2[j]) * mutate
        
        # Manual clamping (faster than method call)
        if val < b_min
          trial[j] = b_min
        elsif val > b_max
          trial[j] = b_max
        else
          trial[j] = val
        end
      else
        trial[j] = xt[j]
      end
      j += 1
    end

    # 3. Evaluation
    score_trial = Problems.send(OPTIMIZER, trial)

    # 4. Selection
    if score_trial < scores[i]
      # IMPORTANT: Must .dup because 'trial' is a reused reference
      population[i] = trial.dup 
      scores[i] = score_trial
    end
    
    i += 1
  end

  # Logging
  if g % PRINT_INTERVAL == 0 || g == GENERATIONS - 1
    # Manual sum is faster than .sum for simple floats
    sum_scores = 0.0
    scores.each { |s| sum_scores += s }
    mean = sum_scores / POPSIZE
    
    best = scores.min
    
    puts "generation #{g}"
    puts "generation mean #{mean}"
    puts "generation best #{best}"
    
    if LOG_CSV
      file.puts("#{g},#{mean}")
    end
  end
  
  g += 1
end

if LOG_CSV
  file.close
end

# --- Final Results ---
best_score = scores.min
best_index = scores.index(best_score)
best_agent = population[best_index]

end_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)

puts "best #{best_agent}"
puts "score #{best_score}"
puts "time taken: #{end_time - start_time}"