import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import pandas as pd

# Survival data for control and hypo groups for each day
control_data = [
    [20, 8, 8, 8],
    [20, 16, 15, 15],
    [20, 12, 12, 11],
    [20, 14, 14, 13],
    [20, 11, 11, 11],
    [50, 31, 31, 18],
    [50, 27, 23, 10],
]

hypo_data = [
    [20, 11, 11, 8],
    [20, 11, 11, 11],
    [20, 8, 8, 7],
    [20, 11, 11, 11],
    [20, 10, 9, 9],
    [50, 25, 24, 21],
    [50, 29, 28, 20],
]

# Prepare data for survival analysis
def prepare_survival_data(data):
    times = []
    events = []
    for tadpole_count in data:
        survival_days = len(tadpole_count) - 1
        for i in range(survival_days):
            times.append(i + 1)  # Day index (D1, D2, D3)
            events.append(tadpole_count[i] - tadpole_count[i + 1])  # Decrease in survival
    return np.array(times), np.array(events)

control_times, control_events = prepare_survival_data(control_data)
hypo_times, hypo_events = prepare_survival_data(hypo_data)

# 1) Log-rank test between the two groups
results = logrank_test(control_times, hypo_times, event_observed_A=control_events, event_observed_B=hypo_events)
print("Log-rank test summary:")
print(results.summary)

# 2) Kaplan-Meier estimation
kmf_control = KaplanMeierFitter()
kmf_hypo = KaplanMeierFitter()

# Fit and plot for Control group
kmf_control.fit(control_times, event_observed=control_events, label="Control")
plt.figure(figsize=(10, 6))
kmf_control.plot_survival_function(ci_show=True)
plt.title('Survival Curve for Control Group')
plt.xlabel('Days')
plt.ylabel('Survival Probability')

# Fit and plot for Hypo group
kmf_hypo.fit(hypo_times, event_observed=hypo_events, label="Hypo")
plt.figure(figsize=(10, 6))
kmf_hypo.plot_survival_function(ci_show=True)
plt.title('Survival Curve for Hypo Group')
plt.xlabel('Days')
plt.ylabel('Survival Probability')
plt.show()

# 3) Cox proportional hazards model
# Create a DataFrame for CoxPH model
df = pd.DataFrame({
    'time': np.concatenate([control_times, hypo_times]),
    'event': np.concatenate([control_events, hypo_events]),
    'group': ['control'] * len(control_times) + ['hypo'] * len(hypo_times)
})

# Convert group to binary variable
df['group'] = df['group'].map({'control': 0, 'hypo': 1})

# Fit the Cox proportional hazards model
cph = CoxPHFitter()
cph.fit(df, 'time', event_col='event')
cph.print_summary()

# Plot the Cox model
cph.plot()
plt.title('Cox Proportional Hazards Model')
plt.show()
