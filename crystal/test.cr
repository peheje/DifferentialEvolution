# K-genetic-clustering

# We have i data-points with n dimensions
# How do we group these data-points the best into k clusters, given we know k

# We need to find out
# 1) Where to place k centroids in the n dimensions
# 2) Which center each i datapoint fits into

# Let's generate some sample data first
require "statistics"
include Statistics
include Statistics::Distributions


def generate_sample_data()
    std = 1.0
    mean = 0.0
    k = 3
    dimensions = 2
    datapoints_per_group = 10
    
    samples = Array.new(k) do |i|
        mean += 1.0
        sample = create_samples(datapoints_per_group, dimensions, mean, std)
    end

    return samples
end

def create_samples(n, dimensions, mean, std)
    normal = Normal.new(mean, std)
    xs = Array.new(n) { Array.new(dimensions) { normal.rand } }
end

samples = generate_sample_data()

puts samples.inspect