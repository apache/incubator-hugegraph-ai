# gremlin_checker.py  通过AST进行语法检查
import os
import sys
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)

from gremlin.GremlinLexer import GremlinLexer
from gremlin.GremlinParser import GremlinParser

class SyntaxErrorListener(ErrorListener):
    """
    私有错误监听器类，捕获语法错误。
    """
    def __init__(self):
        super().__init__()
        self.has_error = False
        self.error_message = ""

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        """当语法错误发生时，此方法被调用。"""
        self.has_error = True
        self.error_message = f"Syntax Error at line {line}, column {column}: {msg}"

def check_gremlin_syntax(query_string: str) -> tuple[bool, str]:
    """
    检查给定的Gremlin查询语句的语法。

    Args:
        query_string: The Gremlin query to check.

    Returns:
        A tuple containing:
        - bool: True if syntax is correct, False otherwise.
        - str: An error message if syntax is incorrect, or "Syntax OK" if correct.
    """
    input_stream = InputStream(query_string)
    lexer = GremlinLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = GremlinParser(token_stream)

    # 移除默认的控制台错误监听器
    parser.removeErrorListeners() 
    
    # 添加自定义的监听器
    error_listener = SyntaxErrorListener()
    parser.addErrorListener(error_listener)

    parser.queryList()

    if error_listener.has_error:
        return (False, error_listener.error_message)
    else:
        return (True, "Syntax OK")
if __name__ == "__main__":
    ## 正确语句
    query_string = "g.V().has('name', 'Alice').outE().inV().has('age', 30)"
    result = check_gremlin_syntax(query_string)
    print(result)
    ## 错误语句
    query_string = "g.V().has('name', 'Alice').utE().inV().has('age', 30)"
    result = check_gremlin_syntax(query_string)
    print(result)