import parser
import fractions
import sys

def compute(node):
    if node.token_type == parser.TokenType.T_NUM:
        return node.value
    
    # Handle binary operations
    if len(node.children) == 2:
        left = compute(node.children[0])
        right = compute(node.children[1])
        
        if node.token_type == parser.TokenType.T_PLUS:
            return left + right
        elif node.token_type == parser.TokenType.T_MINUS:
            return left - right
        elif node.token_type == parser.TokenType.T_MULT:
            return left * right
        elif node.token_type == parser.TokenType.T_DIV:
            return left / right
    
    # Handle unary minus (unary minus is not in the current parser, so we don't need to handle it)
    return 0

def main():
    if len(sys.argv) != 2:
        print('Error: invalid input')
        sys.exit(1)
    
    expression = sys.argv[1]
    try:
        ast = parser.parse(expression)
        result = compute(ast)
        
        # Format the result as per requirements
        if isinstance(result, int):
            print(result)
        else:
            # For fractional results, remove trailing zeros
            formatted_result = f'{result:.15f}'.rstrip('0').rstrip('.')
            print(formatted_result)
    except Exception as e:
        print('Error: invalid input')
        sys.exit(1)

if __name__ == '__main__':
    main()
