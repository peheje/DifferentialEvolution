require "uuid"

def generate_data()
    file1 = File.new "ids1.data", "w"
    file2 = File.new "ids2.data", "w"

    1_000_000.times do
        id1 = UUID.random
        file1.puts id1

        if rand < 0.5
            file2.puts id1
        else
            file2.puts UUID.random
        end
    end

    file1.close
    file2.close
end

def in_both(xs : Array(String), ys : Set(String))
    both = Array(String).new
    xs.each do |x|
        if ys.includes? x
            both.push x
        end
    end
    return both
end

def in_only_first(xs : Array(String), ys : Set(String))
    only_first = Array(String).new
    xs.each do |x|
        if !ys.includes? x
            only_first.push x
        end
    end
    return only_first
end

generate_data()
# puts in_both(["a", "b", "c"], ["a", "c"].to_set)
# puts in_only_first(["a", "b", "c", "d"], ["a", "b", "c"].to_set)

start_dt = Time.utc

if ARGV.size == 2
    a = File.read_lines(ARGV[0])
    a_set = a.to_set

    b = File.read_lines(ARGV[1])
    b_set = b.to_set

    end_dt = Time.utc
    puts "files read time: #{end_dt - start_dt}"

    start_dt = Time.utc

    a_and_b = in_both(a, b_set)
    only_a = in_only_first(a, b_set)
    only_b = in_only_first(b, a_set)

    puts "both #{a_and_b.size}"
    puts "only a #{only_a.size}"
    puts "only b #{only_b.size}"

    end_dt = Time.utc
    puts "compare time: #{end_dt - start_dt}"
end
