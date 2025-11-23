import ast

with open('/home/scnu/Gitee/EFD3D/run_enhanced_training.py', 'r', encoding='utf-8') as f:
    try:
        ast.parse(f.read())
        print('代码语法正确！')
    except SyntaxError as e:
        print(f'语法错误: {e}')
        print(f'错误位置: 行 {e.lineno}, 列 {e.offset}')
        print(f'错误信息: {e.msg}')