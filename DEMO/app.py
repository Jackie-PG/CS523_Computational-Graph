from flask import Flask, render_template, request
import sympy as sp
from sympy.printing.dot import dotprint
from sympy import symbols, Function, Derivative
import graphviz
from graphviz import Digraph
from PIL import Image as PILImage
import io 
import base64 


from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    function_exponentiation,
)


# Ghi đè phương thức in LaTeX cho log
from sympy.printing.latex import LatexPrinter

#def _print_log_custom(self, expr):
#    arg = self._print(expr.args[0])
#    return f"log({arg})"
#LatexPrinter._print_log = _print_log_custom


def custom_label_for_graphviz(expr):
    """
    Trả về chuỗi text thuần để hiển thị trong node Graphviz.
    Nếu là log(x, base) thì hiển thị 'log<base>(x)', còn lại thì dùng str(expr).
    """
    if expr.is_Number:
        # ép thành float rồi format .3f
        return f"{float(expr):.3f}"
    
    if expr.func == sp.log and len(expr.args) == 2:
        arg_str = str(expr.args[0])
        base_str = str(expr.args[1])
        return f"log({arg_str}, {base_str})"
    else:
        return str(expr)



class CustomLatexPrinter(LatexPrinter):
    def _print_log(self, expr):
        # Kiểm tra nếu hàm log có 2 đối số: log(argument, base)
        if len(expr.args) == 2:
            arg = self._print(expr.args[0])
            base = self._print(expr.args[1])
            return r'\log_{%s}\left(%s\right)' % (base, arg)
        else:
            arg = self._print(expr.args[0])
            return r'\log\left(%s\right)' % (arg) 
        
    # bỏ “1” trong phép nhân
    def _print_Mul(self, expr):
        if len(expr.args) == 2:
            num, den = expr.as_numer_denom()
            if den != 1:
                num_str = self._print(num)
                den_str = self._print(den)
                return r'\frac{%s}{%s}' % (num_str, den_str)
        return super()._print_Mul(expr)

def custom_latex(expr):
    return CustomLatexPrinter().doprint(expr)



# transformations để parse biểu thức
transformations = standard_transformations + (
    implicit_multiplication_application,
    function_exponentiation,
)

def collect_node_info(expr, var_symbol, vals):
    """
    Duyệt qua biểu thức theo thứ tự preorder, trả về dict:
      { node_expr: (LaTeX(node), giá trị evaluate, đạo hàm evaluate theo var_symbol) }
    Nếu không có giá trị (values rỗng) thì trả về None cho giá trị và đạo hàm.
    """
    info = {}
    for node in sp.preorder_traversal(expr):
        node_tex = custom_latex(node)
        # Cố gắng evaluate giá trị của node
        try:
            node_val = float(node.subs(vals).evalf()) if vals else None
        except Exception:
            node_val = None
        # Tính đạo hàm của node theo var_symbol
        try:
            node_grad = float(sp.diff(node, var_symbol).subs(vals).evalf()) if vals else None
        except Exception:
            node_grad = None
        info[node] = (node_tex, node_val, node_grad)
    return info

def build_expr_tree(expr, node_info, graph=None, parent_id=None,
                    counter=[0], mapping=None):
    """
    Duyệt cây biểu thức và vẽ bằng Graphviz. 
    mapping: dict để đảm bảo mỗi expr chỉ có một node_id.
    """ 
    if counter is None:
        counter = [0]

    if graph is None:
        graph = Digraph(format='svg')
        graph.attr('node', shape='oval')
    if mapping is None:
        mapping = {}

    # --- MODIFIED: Bỏ qua wrapper Mul(1, x) để không vẽ node thừa
    if expr.func == sp.Mul:
        non_ones = [a for a in expr.args if not (a.is_Number and float(a) == 1)]
        if len(non_ones) == 1:
            return build_expr_tree(non_ones[0], node_info,
                                   graph, parent_id,
                                   counter, mapping)


    # Nếu đã xử lý expr này rồi, chỉ cần vẽ cạnh đến parent và return
    if expr in mapping:
        node_id = mapping[expr]
        if parent_id is not None:
            graph.edge(node_id, parent_id)
        return graph

    # Gán ID mới cho expr
    node_id = str(counter[0])
    counter[0] += 1
    mapping[expr] = node_id

    # Label cho node
    label_str = custom_label_for_graphviz(expr)
    label = f'''<<FONT FACE="Times New Roman" POINT-SIZE="12">{label_str}</FONT>>'''

    # Tạo node
    graph.node(node_id, label=label)

    # Nối edge đến parent
    if parent_id is not None:
        graph.edge(node_id, parent_id)


    # Đệ quy với các arg
    for arg in expr.args:
         # (a) Bỏ qua số 1 (1 * anything)
        if arg.is_Number and float(arg) == 1:
            continue

        # (b) Nếu expr là Pow(x, e) thì bỏ qua phần exponent e
        if expr.func == sp.Pow and arg == expr.args[1]:
            continue

        build_expr_tree(arg, node_info, graph, node_id, counter, mapping)

    return graph



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!' 

@app.route("/", methods=["GET", "POST"]) 

def index():
    forward_result = None
    error = None
    expr = None
    image = None
    variables = []
    expr_input = ""
    show_result = False # True khi đã tính toán và cần hiển thị kết quả
    show_variables = False # True khi đã chuyển đổi và cần nhập giá trị
    gradients = {}
    gradient_expressions = {}
    img_str = None  # Chuỗi Base64 của hình ảnh 
    expr_latex = None  # Biến chứa công thức LaTeX của biểu thức
    values = {}

    if request.method == "POST":
        action = request.form.get("action")
        expr_input = request.form.get("expr_input")
        # 1) Lấy biểu thức từ form
        try:
             # 1) Parse biểu thức, không evaluate để giữ nguyên cấu trúc
            expr = sp.parse_expr(expr_input, evaluate=False)
            

            expr_latex = custom_latex(expr)  # Chuyển biểu thức sang định dạng LaTeX
        except Exception as e:
            error = f"Lỗi khi phân tích biểu thức: {e}"
            return render_template("index.html", error=error, expr_input=expr_input)
       
        # 2) Lấy danh sách các biến từ biểu thức
        variables = sorted([str(v) for v in expr.free_symbols]) 

        # 3) Nếu user vừa nhấn "Xem biến" -> hiện form nhập giá trị 
        if action == "parse":
            show_variables = True

        
        # 4) Nếu bấm Tính toán thì thu giá trị, compute và vẽ đồ thị
        elif action == "compute":
            # Giữ lại các đối tượng Symbol để thay thế
            # variables = list(expr.free_symbols)
            values = {}
            for var in variables:
                # Dùng str(var) để lấy giá trị từ form
                value = request.form.get(var)
                if value is None or value.strip()=="":
                    error = "Vui long nhap day du gia tri cho cac bien." 
                    show_variables = True 
                    break
                values[var] = float(value) 


            if not error: 
                # Truyền xuôi
                result_value = expr.subs(values).evalf()
                try:
                    forward_result = format(float(result_value), '.4f')
                except Exception:
                    forward_result = str(result_value)
                
                # Tính đạo hàm cho từng biến
                for var in variables:
                    derivative_expr = Derivative(expr, var).doit()
                    # Chuyển công thức đạo hàm sang LaTeX
                    derivative_latex = custom_latex(derivative_expr)
                    gradient_expressions[var] = derivative_latex  # Lưu công thức đạo hàm
                    
                    grad_value = derivative_expr.subs(values).evalf()
                    try:
                        gradients[var] = format(float(grad_value), '.4f')
                    except Exception:
                        gradients[var] = str(grad_value)
           
                show_result = True 
        
         # Sử dụng biến đầu tiên (nếu có) cho việc tính đạo hàm của từng node trong cây biểu thức
        if variables:
            var0 = variables[0]
            var0_symbol = symbols(var0)
        else:
            var0_symbol = symbols('x')

        # Thu thập thông tin cho từng node
        node_info = collect_node_info(expr, var0_symbol, values)
        # Vẽ cây biểu thức
        graph = build_expr_tree(expr, node_info)
        svg = graph.pipe().decode('utf-8')
        img_str = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
        plot_url = img_str

  #.....        

        return render_template("index.html", 
                               expr_input=expr_input, 
                               forward_result=forward_result, 
                               error=error, 
                               pretty_expr=expr, 
                               expr_latex=expr_latex,  # Truyền biểu thức dạng LaTeX
                               plot_url = plot_url, 
                               variables = variables, 
                               gradients=gradients, 
                               gradient_expressions=gradient_expressions, 
                               show_variables = show_variables,
                               show_result = show_result)
    

    return render_template("index.html", expr_input=expr_input, error=error)

if __name__ == "__main__":
    app.run(debug=True)
