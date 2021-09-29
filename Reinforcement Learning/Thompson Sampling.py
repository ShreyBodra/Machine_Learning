import pandas as pd
import random
import matplotlib.pyplot as plt
# loading dataset
data = pd.read_csv('Ads_CTR_Optimisation.csv')
print(data.head())

N = len(data)
d = len(data.columns)
ads_selected = []
number_of_reward_1 = [0] * d
number_of_reward_0 = [0] * d
total_reward = 0

for n in range(0,N):
    ad = 0
    max_random_draw = 0
    for i in range(0,d):
        draw = random.betavariate(number_of_reward_1[i] + 1, number_of_reward_0[i] + 1)

        if  draw > max_random_draw:
            max_random_draw = draw
            ad = i

    ads_selected.append(ad)
    reward = data.values[n,ad]
    if reward == 1 :
        number_of_reward_1[ad] += 1
    else:
        number_of_reward_0[ad] += 1

    total_reward += reward


plt.hist(ads_selected)
plt.title("Histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()
