class Op
  def initialize(@sign : String, @proc : Proc(Node?, Node?, Float64), @unary : Bool = false)
  end

  def proc
    @proc
  end

  def sign
    @sign
  end
end

class NodeWrapper
  def initialize(@node : Node?)
  end

  def setNode(node : Node?)
    @node = node
  end

  def node()
    @node
  end
end

class Node
  @left : Node?
  @right : Node?

  def initialize(@op : Op, @value : Float64 = 0.0)
  end

  def value
    @value
  end

  def eval
    if @op.sign == "val"
      @value
    else
      @op.proc.call @left, @right
    end
  end
end

def random_node(ops : Array(Op), leaf : Bool)
  if leaf
    Node.new ops[0], rand(-10.0..10.0)
  else
    Node.new ops[rand(1..ops.size - 1)]
  end
end

def random_tree(ops : Array(Op), max : Int, counter : Int = 0)
  node = random_node ops false
end

ops = Array(Op).new

ops << Op.new "val", ->(a : Node?, b : Node?) { a.not_nil!.value }
ops << Op.new "+", ->(a : Node?, b : Node?) { a.not_nil!.eval + b.not_nil!.eval }
ops << Op.new "-", ->(a : Node?, b : Node?) { a.not_nil!.eval - b.not_nil!.eval }

#puts ops

n = random_node(ops, false)

puts "main 1"
puts n.inspect

random_tree ops 2

puts "main 2"
puts n.inspect