import numpy as np
import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')

items = []
for i in range (2201):
    items.append([str(df.values[i,j]) for j in range(5)])

final_rule = apriori(items, min_support=0.005, min_confidence=0.8)
final_results=list(final_rule)

relation = final_results[0]

survival_rules = [relation for relation in final_results if 'Yes' in relation.items or 'No' in relation.items]
sorted_survival_rules = sorted(survival_rules, key=lambda x: x.ordered_statistics[0].confidence, reverse=True)

print("Top 10 rules based on confidence for survival:")
for i in range(min(10, len(sorted_survival_rules))):
    print("Rule {}: {}".format(i+1, sorted_survival_rules[i]))

supports = [relation.support for relation in final_results]
confidences = [relation.ordered_statistics[0].confidence for relation in final_results]

print(final_results[0])

df_plot = pd.DataFrame({'support': supports, 'confidence': confidences})

plt.figure(figsize=(10, 5))
plt.bar(range(len(df_plot)), df_plot['support'], color='blue')
plt.title('Support of Apriori Rules')
plt.xlabel('Rules')
plt.ylabel('Support')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(range(len(df_plot)), df_plot['confidence'], color='green')
plt.title('Confidence of Apriori Rules')
plt.xlabel('Rules')
plt.ylabel('Confidence')
plt.show()
