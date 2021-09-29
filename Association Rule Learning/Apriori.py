import pandas as pd
from apyori import apriori

#loading dataset
data = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
print(data.head())

# creating list of items for every customer
transactions = []
for i in range(0, 7051):
    transactions.append([str(data.values[i, j]) for j in range(0, 20)])
print(transactions)


# creating model
rules = apriori(transactions=transactions, min_support = 0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
results = list(rules)

# writing rules in readable format
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    support = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lift = [result[2][0][3] for result in results]
    return zip(lhs,rhs,support,confidence,lift)

data_frame = pd.DataFrame(inspect(results), columns=['Left Side', 'Right Side', 'Support', 'Confidence', 'Lift'])
print(data_frame)

# display output based on highest lift 1st
print(data_frame.nlargest(10,"Lift"))
