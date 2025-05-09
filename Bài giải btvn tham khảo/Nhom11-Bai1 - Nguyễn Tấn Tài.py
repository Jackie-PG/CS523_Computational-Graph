import numpy as np

class Node:
    def __init__(self,name,operation,inputs=None,value=None):
        self.name=name
        self.operation=operation
        self.inputs=inputs if inputs is not None else []
        self.value=value
        self.gradient=0
    
class ComputationalGraph:
    def __init__(self):
        self.Nodes=[]
        self.node_dict = {}
    
    def addNode(self,Node):
        self.Nodes.append(Node)
        self.node_dict[Node.name] = Node
        return Node
   
    def Forward(self):
        for Node in self.Nodes:
            if Node.operation == 'input':
                continue
            elif Node.operation == '+':
                a = self.node_dict[Node.inputs[0]].value
                b = self.node_dict[Node.inputs[1]].value
                Node.value = a + b
            elif Node.operation == '*':
                a = self.node_dict[Node.inputs[0]].value
                b = self.node_dict[Node.inputs[1]].value
                Node.value = a * b
            elif Node.operation == '-':
                a = self.node_dict[Node.inputs[0]].value
                b = self.node_dict[Node.inputs[1]].value
                Node.value = a - b
            elif Node.operation == '^2':
                a = self.node_dict[Node.inputs[0]].value
                Node.value = a**2
            elif Node.operation == '*c':
                a = self.node_dict[Node.inputs[0]].value
                c = Node.inputs[1]
                Node.value = a * c
    
    def Backward(self):
        self.node_dict['L'].gradient = 1
        
        for node in reversed(self.Nodes):
            if node.operation == 'input':
                continue  
            elif node.operation == '*':
                a = self.node_dict[node.inputs[0]]
                b = self.node_dict[node.inputs[1]]
                a.gradient += node.gradient * b.value
                b.gradient += node.gradient * a.value
            elif node.operation == '+':
                a = self.node_dict[node.inputs[0]]
                b = self.node_dict[node.inputs[1]]
                a.gradient += node.gradient * 1
                b.gradient += node.gradient * 1
            elif node.operation == '-':
                a = self.node_dict[node.inputs[0]]
                b = self.node_dict[node.inputs[1]]
                a.gradient += node.gradient * 1
                b.gradient += node.gradient * -1
            elif node.operation == '^2':
                a = self.node_dict[node.inputs[0]]
                a.gradient += node.gradient * 2 * a.value
            elif node.operation == '*c':
                a = self.node_dict[node.inputs[0]]
                c = node.inputs[1]
                a.gradient += node.gradient * c
    
graph = ComputationalGraph()

# Add input nodes (fixed values)
graph.addNode(Node('x','input', [], value=3))
graph.addNode(Node('y', 'input', [], value=9))
graph.addNode(Node('theta', 'input', [], value=2))
graph.addNode(Node('b', 'input', [], value=1))

# Build the computational graph
graph.addNode(Node('u', '*', ['theta', 'x'])) # u = theta * x
graph.addNode(Node('v', '+', ['u', 'b']))     # v = u + b
graph.addNode(Node('w', '-', ['y', 'v']))     # w = y - v
graph.addNode(Node('z', '^2', ['w']))         # z = w^2
graph.addNode(Node('L', '*c', ['z', 0.5]))    # L = 0.5 * z

#2.Lan truyền tiến
graph.Forward()

#3.Lan truyền ngược
graph.Backward()

# Tính đạo hàm cho hàm mất mát theta và b
dL_dtheta = graph.node_dict['theta'].gradient
dL_db = graph.node_dict['b'].gradient

#4.Mô phỏng cập nhật tham số
learning_rate = 1
theta_new = graph.node_dict['theta'].value - learning_rate * dL_dtheta
b_new = graph.node_dict['b'].value - learning_rate * dL_db

# Kết quả
print("Computational Graph Results:")
print(f"Forward pass: L = {graph.node_dict['L'].value:.2f}")
print(f"Gradients: dL_dtheta = {dL_dtheta:.2f}, dL_db = {dL_db:.2f}")
print(f"Updated parameters: theta = {theta_new:.2f}, b = {b_new:.2f}")