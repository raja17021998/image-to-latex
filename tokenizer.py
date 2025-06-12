import re

def tokenize_latex(expression):
    # Remove surrounding dollar signs (common in LaTeX math mode)
    expression = expression.strip()
    if expression.startswith('$') and expression.endswith('$'):
        expression = expression[1:-1].strip()

    # Add known LaTeX commands or patterns that should remain as single tokens:
    # For example: \frac, \sqrt, \alpha, \beta, \theta, \sin, \cos, \tan, \log, \left, \right, \big, \bigg, etc.
    # The pattern below matches:
    # 1) Common LaTeX commands (\\[a-zA-Z]+)
    # 2) Specific commands like \left, \right, \big, \bigg, \lbrace, \rbrace, \langle, \rangle
    # 3) Braces { and }
    # 4) Parentheses ( and ), square brackets [ and ]
    # 5) Digits (one or more), operators (+, -, ^, _), and single letters (a-z, A-Z)
    pattern = (
        r'(\\[a-zA-Z]+|\\left|\\right|\\bigg|\\big|\\lbrace|\\rbrace|\\langle|\\rangle|'
        r'\{|\}|\(|\)|\[|\]|[0-9]+|\+|\-|\^|\_|[a-zA-Z])'
    )

    tokens = re.findall(pattern, expression)

    # Filter out any empty tokens, just in case
    tokens = [t for t in tokens if t.strip()]

    return tokens

# Example usage:
# formula = "$ 3 { a ^ { 2 } } { b ^ { 3 } } + 5 { a ^ { 3 } } { b ^ { 2 } } - \\frac { { a ^ { 5 } } { b ^ { 8 } } } { 2 } $"
# print(tokenize_latex(formula))
