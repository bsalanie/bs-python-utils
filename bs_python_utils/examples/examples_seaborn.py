"""examples using my Seaborn functions"""

import matplotlib.pyplot as plt
import seaborn as sns

from bs_python_utils.bs_seaborn import bs_regplot, bs_sns_bar_x_byf, bs_sns_bar_x_byfg

cars = sns.load_dataset("mpg")

g1 = bs_sns_bar_x_byf(
    cars,
    "horsepower",
    "cylinders",
    label_x="Horsepower",
    label_f="Number of cylinders",
    title="Mean HP by number of cylinders",
)

plt.clf()

g2 = bs_sns_bar_x_byfg(
    cars,
    "horsepower",
    "cylinders",
    "origin",
    label_x="Horsepower",
    label_f="Number of cylinders",
    label_g="Origin",
    title="Mean HP by number of cylinders and origin",
)

plt.clf()

# regression plot
g3 = bs_regplot(
    cars,
    "weight",
    "horsepower",
    title="Mean HP by weight",
    save="../Graphs/bs_regplot",
)
