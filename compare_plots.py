import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_eric = pd.read_csv("/Users/ericmartz/Downloads/eric_100+400.csv")
df_anna = pd.read_csv("/Users/ericmartz/Downloads/anna_100+500.csv")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(df_eric["Step"],
        df_eric["curriculum-onset-a0.1-seed42 - finetune/eval/f1"],
        label="Curriculum + onset (100k pretrain + 400k finetune)", color="orange")
ax.plot(df_anna["Step"],
        df_anna["curriculum-onset-short-a0.1-s2.0-seed42 - finetune/eval/f1"],
        label="Curriculum + onset (100k pretrain + 500k finetune)", color="blue")

ax.set_xlabel("Step")
ax.set_ylabel("$F_1$")
ax.set_title("finetune/eval/f1")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
plt.tight_layout()
plt.savefig("comparison_600k.png", dpi=150)
print("Saved comparison_600k.png")
