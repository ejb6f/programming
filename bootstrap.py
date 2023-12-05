print("Question 2:")

print("\n2.1:")
import pandas as pd

df1 = pd.read_csv('mammogram.csv')
cross_tab = pd.crosstab(df1['treatment'], df1['breast_cancer_death'], margins = True, margins_name = 'Total')
print("Cross Tabulation")
print(cross_tab)

survival_rates = cross_tab['no'] / cross_tab['Total']
difference_in_survival_rates = survival_rates['control'] - survival_rates['mammogram']

print(f"\nDifference in 25-year survival rates between control and mammogram groups is {difference_in_survival_rates:.5f}")

print("\n2.2:")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_survival_rate(data):
    return np.sum(data == 'no') / len(data)

def bootstrap_sample(data, n_samples = 1000):
    bootstrap_samples = np.random.choice(data, size=(n_samples, len(data)), replace = True)
    survival_rates = np.apply_along_axis(calculate_survival_rate, axis = 1, arr = bootstrap_samples)
    return survival_rates

bootstrap_survival_control = bootstrap_sample(df1[df1['treatment'] == 'control']['breast_cancer_death'])
bootstrap_survival_mammogram = bootstrap_sample(df1[df1['treatment'] == 'mammogram']['breast_cancer_death'])

sns.histplot(bootstrap_survival_control, label = 'Control', kde = True)
sns.histplot(bootstrap_survival_mammogram, label = 'Mammogram', kde = True)
plt.title('Bootstrap Distributions of Survival Rates')
plt.xlabel('Survival Rate')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print("2.3:")
confidence_interval = np.percentile(bootstrap_survival_control - bootstrap_survival_mammogram, [0.5, 99.5])
zero_in_interval = confidence_interval[0] <= 0 <= confidence_interval[1]

print(f"99% Confidence Interval: [{confidence_interval[0]:.5f}, {confidence_interval[1]:.5f}]")
print("The confidence interval does in fact include zero.")

print("\n2.4")
print("This data might over/understate the conclusions I have reached because as a result of data biases. For example, if the patients in the dataset are not a representative sample, the results may not generalizable. Selection bias may also induce the over/under estimation of the conclusions, as the way patients were entered into the dataset could introduce overall bias. With only two variables, lack of detailed patient information may cause limitations in the conclusions made.")
print("Other data that I would like to have to better understand or criticize the results include information on other health conditions, lifestyle factors, and the stage and type of breast cancer. Such data would allow me to best access the the gathered results.")



print("Question 3:")

print("\n3.1:")
import pandas as pd
df2 = pd.read_csv('diabetes_hw.csv')
cross_tab2 = pd.crosstab(df2['treatment'], df2['outcome'])

print("Cross Tabulation")
print(cross_tab2)

print("\n3.2:")
proportion_success = cross_tab2['success'] / cross_tab2.sum(axis=1)

print("Treatment Proportion of Success:")
print(proportion_success)
print("\nThe treatment that appears to be the most effective is rosi, as it has the highest treatment proportion of success.")

print("\n3.3:")
def calculate_proportion_success(data):
    cross_tab = pd.crosstab(data['treatment'], data['outcome'])
    proportion_success = cross_tab['success'] / cross_tab.sum(axis = 1)
    return proportion_success.values

def bootstrap_proportions(data, num_samples = 1000):
    treatments = data['treatment'].unique()
    bootstrap_results = []

    for treatment in treatments:
        treatment_data = data[data['treatment'] == treatment]
        bootstrap_samples = []

        for value in range(num_samples):
            resample = treatment_data.sample(frac=1, replace=True)
            proportion_success = calculate_proportion_success(resample)
            bootstrap_samples.append(proportion_success)

        bootstrap_results.append((treatment, np.array(bootstrap_samples)))

    return bootstrap_results

plt.figure(figsize=(10, 6))
for treatment, samples in bootstrap_proportions(df2):
    sorted_samples = np.sort(samples.flatten())
    ecdf_values = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    plt.step(sorted_samples, ecdf_values, label=treatment)

plt.title('Empirical CDF of Proportions of Success by Treatment Type')
plt.xlabel('Proportion of Success')
plt.ylabel('Cumulative Probability')
plt.legend(title = 'Treatment Type')
plt.show()

plt.figure(figsize=(10, 6))
for treatment, samples in bootstrap_proportions(df2):
    flattened_samples = samples.flatten()
    plt.hist(flattened_samples, bins=30, density=True, alpha=0.5, label=treatment, edgecolor='black', linewidth=1)

plt.title('Kernel Density of Proportions of Success by Treatment Type')
plt.xlabel('Proportion of Success')
plt.ylabel('Density')
plt.legend(title = 'Treatment Type')
plt.show()

print("Based on the charts, rosi appers to be the most effective treatment.")

print("\n3.4:")
def calculate_difference(data1, data2):
    return data1 - data2

bootstrap_results_lifestyle_met = bootstrap_proportions(df2[df2['treatment'].isin(['lifestyle', 'met'])])
bootstrap_results_met_rosi = bootstrap_proportions(df2[df2['treatment'].isin(['met', 'rosi'])])
bootstrap_results_rosi_lifestyle = bootstrap_proportions(df2[df2['treatment'].isin(['rosi', 'lifestyle'])])

difference_lifestyle_met = calculate_difference(bootstrap_results_lifestyle_met[0][1][:, 0], bootstrap_results_lifestyle_met[1][1][:, 0])
difference_met_rosi = calculate_difference(bootstrap_results_met_rosi[0][1][:, 0], bootstrap_results_met_rosi[1][1][:, 0])
difference_rosi_lifestyle = calculate_difference(bootstrap_results_rosi_lifestyle[0][1][:, 0], bootstrap_results_rosi_lifestyle[1][1][:, 0])

confidence_interval_lifestyle_met = np.percentile(difference_lifestyle_met, [5, 95])
confidence_interval_met_rosi = np.percentile(difference_met_rosi, [5, 95])
confidence_interval_rosi_lifestyle = np.percentile(difference_rosi_lifestyle, [5, 95])

significant_lifestyle_met = confidence_interval_lifestyle_met[0] > 0 or confidence_interval_lifestyle_met[1] < 0
significant_met_rosi = confidence_interval_met_rosi[0] > 0 or confidence_interval_met_rosi[1] < 0
significant_rosi_lifestyle = confidence_interval_rosi_lifestyle[0] > 0 or confidence_interval_rosi_lifestyle[1] < 0

print("\n90% Confidence Level Pairwise Treatment Comparisons:")
print(f"Lifestyle vs. Met: {confidence_interval_lifestyle_met}, Significant: {significant_lifestyle_met}")
print(f"Met vs. Rosi: {confidence_interval_met_rosi}, Significant: {significant_met_rosi}")
print(f"Rosi vs. Lifestyle: {confidence_interval_rosi_lifestyle}, Significant: {significant_rosi_lifestyle}")

print("\nThe results indicate that, at the 90% confidence level, the outcomes for the 'Met vs. Rosi' and 'Rosi vs. Lifestyle' comparisons are significantly different.")

print("\n3.5:")
print("The treatment that appears to be the most effective is rosi. This is because it has the highest treatment proportion of success, being 0.613734. This is further presented by the charts, as in both the emperical CDF and the kernel density plot, rosi appears to hold a confidence interval at higher values for proportion of success compared to the other two treatments.")
