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
        self.Nodes = []
        self.node_dict = {}
    
    def addNode(self,Node):
        self.Nodes.append(Node)
        self.node_dict[Node.name] = Node
        return Node
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
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
            elif Node.operation == 'sigmoid':
                a = self.node_dict[Node.inputs[0]].value
                Node.value = self.sigmoid(a)
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
            elif node.operation == 'sigmoid':
                a = self.node_dict[node.inputs[0]]
                a.gradient += node.gradient * self.sigmoid_derivative(a.value)
    
    def update_parameters(self, learning_rate):
        for node in self.Nodes:
            if node.name in ['w1', 'b1', 'w2', 'b2']:
                node.value -= learning_rate * node.gradient

#1.Xây dựng Computational Graph
graph = ComputationalGraph()

graph.addNode(Node('x', 'input', value=2.0))
graph.addNode(Node('y', 'input', value=3.0))

graph.addNode(Node('w1', 'input', value=0.5))
graph.addNode(Node('b1', 'input', value=0.0))
graph.addNode(Node('w2', 'input', value=-0.8))
graph.addNode(Node('b2', 'input', value=0.1))

graph.addNode(Node('w1*x', '*', ['w1', 'x']))
graph.addNode(Node('z', '+', ['w1*x', 'b1']))
graph.addNode(Node('a', 'sigmoid', ['z']))
graph.addNode(Node('w2*a', '*', ['w2', 'a']))
graph.addNode(Node('y_pred', '+', ['w2*a', 'b2']))
graph.addNode(Node('u', '-', ['y', 'y_pred']))  
graph.addNode(Node('v', '^2', ['u']))  
graph.addNode(Node('L', '*c', ['v', 0.5]))  

#2.Lan truyền tiến
graph.Forward()
print("Forward pass:")
print(f" z = {graph.node_dict['z'].value}")
print(f" a = {graph.node_dict['a'].value}")
print(f" y_pred = {graph.node_dict['y_pred'].value}")
print(f" L = {graph.node_dict['L'].value}")

#3. Tính Gradient của L theo w2,b2,w1,b1
graph.Backward()
print("Forward pass:")
print(f"Gradient cua L theo w1= {graph.node_dict['w1'].gradient}")
print(f"Gradient cua L theo w2= {graph.node_dict['w2'].gradient}")
print(f"Gradient cua L theo b1= {graph.node_dict['b1'].gradient}")
print(f"Gradient cua L theo b2= {graph.node_dict['b2'].gradient}")



# Update parameters
learning_rate = 0.2
graph.update_parameters(learning_rate)
print("\nUpdated parameters:")
print(f"w1= {graph.node_dict['w1'].value}")
print(f"w2= {graph.node_dict['w2'].value}")
print(f"b1= {graph.node_dict['b1'].value}")
print(f"b2= {graph.node_dict['b2'].value}")