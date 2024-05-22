def print_latex_table(results):
    alpha_hat = results['alpha_hat']
    beta_hat = results['beta_hat']
    alpha_er = results['alpha_er']
    beta_er = results['beta_er']

    latex_table = f"""
\\begin{{table}}[H]
    \\centering
    \\begin{{tabular}}{{|c|c|c|c|}} 
        \\hline
            $\\alpha$ & $\\beta$ & Стандартная ошибка $\\alpha$ & Стандартная ошибка $\\beta$ \\\\ \\hline
            {alpha_hat:.4f} & {beta_hat:.4f} & {alpha_er:.4f} & {beta_er:.4f} \\\\ \\hline
    \\end{{tabular}}
    \\caption{{Результаты оценки коэффициентов линейной регрессии}}
    \\label{{tab:regression_results}}
\\end{{table}}
"""
    print(latex_table)
